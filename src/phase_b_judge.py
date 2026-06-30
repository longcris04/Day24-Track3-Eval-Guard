from __future__ import annotations

"""Phase B: LLM-as-Judge — pairwise, swap-and-average, Cohen κ, bias analysis."""

import json
import os
import sys
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (OPENAI_API_KEY, JUDGE_MODEL, HUMAN_LABELS_PATH,
                    OPENROUTER_API_KEY, OPENROUTER_BASE_URL, OPENROUTER_MODEL)


def _judge_client():
    """Return (client, model) for the judge — OpenAI nếu có key, ngược lại OpenRouter."""
    from openai import OpenAI
    if OPENAI_API_KEY:
        return OpenAI(api_key=OPENAI_API_KEY), JUDGE_MODEL
    return OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL), OPENROUTER_MODEL


def _parse_judge_json(raw: str) -> dict:
    """Parse JSON từ LLM, chịu được markdown fences và winner viết hoa/thường.

    Luôn trả về dict với winner ∈ {"A","B","tie"} và scores ∈ [0,1].
    """
    text = (raw or "").strip()
    if "```" in text:                       # strip ```json ... ``` fences
        text = text.split("```")[1] if text.count("```") >= 2 else text.replace("```", "")
        if text.lstrip().lower().startswith("json"):
            text = text.lstrip()[4:]
    try:
        data = json.loads(text)
    except Exception:
        start, end = text.find("{"), text.rfind("}")
        try:
            data = json.loads(text[start:end + 1])
        except Exception:
            return {"winner": "tie", "reasoning": raw.strip()[:200], "scores": {"A": 0.5, "B": 0.5}}

    w = str(data.get("winner", "tie")).strip().upper()
    winner = "A" if w.startswith("A") else "B" if w.startswith("B") else "tie"

    scores = data.get("scores", {}) or {}
    def _clamp(v):
        try:
            return max(0.0, min(1.0, float(v)))
        except (TypeError, ValueError):
            return 0.5
    return {
        "winner":    winner,
        "reasoning": str(data.get("reasoning", "")),
        "scores":    {"A": _clamp(scores.get("A", 0.5)), "B": _clamp(scores.get("B", 0.5))},
    }


@dataclass
class JudgeResult:
    question: str
    answer_a: str
    answer_b: str
    winner_pass1: str       # "A" | "B" | "tie"  (original order)
    winner_pass2: str       # "A" | "B" | "tie"  (after swap, ALREADY converted back)
    final_winner: str       # consensus after swap-and-average
    reasoning_pass1: str
    reasoning_pass2: str
    position_consistent: bool  # True if both passes agree on same answer
    scores_pass1: dict = field(default_factory=dict)  # {"A": float, "B": float}
    scores_pass2: dict = field(default_factory=dict)


# ─── Task 5: Pairwise Judge ───────────────────────────────────────────────────

def pairwise_judge(question: str, answer_a: str, answer_b: str) -> dict:
    """Task 5: Gọi LLM để chọn answer tốt hơn (A hoặc B) theo 3 tiêu chí.

    Tiêu chí đánh giá:
        - Độ chính xác (accuracy): có khớp với thực tế chính sách không?
        - Độ đầy đủ (completeness): có trả lời đủ câu hỏi không?
        - Tính súc tích (conciseness): có thừa / thiếu thông tin không?

    Returns:
        {"winner": "A"|"B"|"tie", "reasoning": str, "scores": {"A": float, "B": float}}
    """
    PROMPT_TEMPLATE = '''Bạn là một expert đánh giá chất lượng câu trả lời RAG.

Câu hỏi: {question}

Answer A:
{answer_a}

Answer B:
{answer_b}

Đánh giá dựa trên 3 tiêu chí: độ chính xác, đầy đủ, súc tích.
Trả lời JSON (chỉ JSON, không text khác):
{{"winner": "A" hoặc "B" hoặc "tie", "reasoning": "giải thích ngắn gọn", "scores": {{"A": 0.0-1.0, "B": 0.0-1.0}}}}
'''
    try:
        client, model = _judge_client()
        kwargs = dict(
            model=model,
            messages=[
                {"role": "system", "content": "Bạn là expert đánh giá RAG. Chỉ trả lời JSON."},
                {"role": "user",   "content": PROMPT_TEMPLATE.format(
                    question=question, answer_a=answer_a, answer_b=answer_b)},
            ],
            temperature=0,
        )
        try:                                   # JSON mode nếu model hỗ trợ
            resp = client.chat.completions.create(
                response_format={"type": "json_object"}, **kwargs)
        except Exception:
            resp = client.chat.completions.create(**kwargs)
        return _parse_judge_json(resp.choices[0].message.content)
    except Exception as e:
        print(f"  ⚠️  pairwise_judge LLM call failed: {e}")
        return {"winner": "tie", "reasoning": "", "scores": {"A": 0.5, "B": 0.5}}


# ─── Task 6: Swap-and-Average ─────────────────────────────────────────────────

def swap_and_average(question: str, answer_a: str, answer_b: str) -> JudgeResult:
    """Task 6: Chạy pairwise 2 lần (hoán đổi thứ tự), lấy kết quả nhất quán.

    Lý do: LLM thường có position bias (ưu tiên answer xuất hiện trước).
    Bằng cách swap, ta phát hiện và giảm bias này.

    Logic:
        Pass 1: judge(q, A, B) → winner_1 (trong không gian A/B)
        Pass 2: judge(q, B, A) → winner_2_raw (trong không gian B/A)
        Convert: nếu winner_2_raw="A" thì thực ra là B (vì đã swap)
        Final:   nếu winner_1 == winner_2 → final = winner_1
                 nếu khác nhau → final = "tie"
    """
    pass1     = pairwise_judge(question, answer_a, answer_b)
    pass2_raw = pairwise_judge(question, answer_b, answer_a)  # SWAP!

    # Convert pass2 back to original A/B space
    swap_map = {"A": "B", "B": "A", "tie": "tie"}
    winner_pass2 = swap_map[pass2_raw["winner"]]

    # Average: consensus only if both agree
    position_consistent = (pass1["winner"] == winner_pass2)
    final = pass1["winner"] if position_consistent else "tie"  # disagreement = inconclusive

    return JudgeResult(
        question=question, answer_a=answer_a, answer_b=answer_b,
        winner_pass1=pass1["winner"], winner_pass2=winner_pass2,
        final_winner=final,
        reasoning_pass1=pass1["reasoning"], reasoning_pass2=pass2_raw["reasoning"],
        position_consistent=position_consistent,
        scores_pass1=pass1["scores"],
        scores_pass2={"A": pass2_raw["scores"]["B"], "B": pass2_raw["scores"]["A"]},
    )


# ─── Task 7: Cohen's κ ────────────────────────────────────────────────────────

def cohen_kappa(judge_labels: list[int], human_labels: list[int]) -> float:
    """Task 7: Tính Cohen's κ giữa LLM judge và human labels.

    Args:
        judge_labels:  nhãn từ LLM judge (0 = bad answer, 1 = good answer)
        human_labels:  nhãn từ human_labels_10q.json

    Returns:
        κ ∈ [-1, 1]
        Thang đo Landis-Koch: <0=poor, 0-0.2=slight, 0.2-0.4=fair,
                               0.4-0.6=moderate, 0.6-0.8=substantial, 0.8-1=almost perfect

    Gợi ý A — dùng scikit-learn:
        from sklearn.metrics import cohen_kappa_score
        return cohen_kappa_score(human_labels, judge_labels)

    Gợi ý B — tính tay:
        n = len(judge_labels)
        p_o = sum(j == h for j, h in zip(judge_labels, human_labels)) / n
        p_e = (judge_labels.count(1)/n * human_labels.count(1)/n +
               judge_labels.count(0)/n * human_labels.count(0)/n)
        κ = (p_o - p_e) / (1 - p_e) if p_e != 1 else 0
        return κ
    """
    if not judge_labels or len(judge_labels) != len(human_labels):
        return 0.0
    try:
        from sklearn.metrics import cohen_kappa_score
        kappa = cohen_kappa_score(human_labels, judge_labels)
    except Exception:
        n = len(judge_labels)
        p_o = sum(j == h for j, h in zip(judge_labels, human_labels)) / n
        p_e = (judge_labels.count(1) / n * human_labels.count(1) / n +
               judge_labels.count(0) / n * human_labels.count(0) / n)
        kappa = (p_o - p_e) / (1 - p_e) if p_e != 1 else 0.0
    # sklearn trả về nan khi một bên không có biến thiên → quy về 0.0
    return 0.0 if kappa != kappa else float(kappa)


# ─── Task 8: Bias Report ──────────────────────────────────────────────────────

def bias_report(judge_results: list[JudgeResult]) -> dict:
    """Task 8: Đo lường position bias và verbosity bias.

    Position bias: LLM chọn answer theo vị trí (A hay B) thay vì chất lượng.
        → Đo bằng % cases where position_consistent = False

    Verbosity bias: LLM ưu tiên answer dài hơn dù không chính xác hơn.
        → Đo bằng: trong các case A thắng, A có dài hơn B không? Tương tự cho B.

    Returns:
        {
          "total_judged": int,
          "position_bias_rate": float,        # 0-1, cao = bias nhiều
          "position_bias_count": int,
          "verbosity_bias": float,            # 0-1, > 0.6 = đáng lo ngại
          "verbosity_details": {
            "a_wins_a_longer": int,           # A thắng VÀ A dài hơn
            "b_wins_b_longer": int,           # B thắng VÀ B dài hơn
            "total_decisive": int,            # tổng case có winner rõ ràng
          },
          "interpretation": str,
        }
    """
    total = len(judge_results)
    if total == 0:
        return {"total_judged": 0, "position_bias_rate": 0.0, "verbosity_bias": 0.0,
                "position_bias_count": 0, "verbosity_details": {}, "interpretation": ""}

    position_bias_count = sum(1 for r in judge_results if not r.position_consistent)
    position_bias_rate  = position_bias_count / total

    a_wins_a_longer = sum(
        1 for r in judge_results
        if r.final_winner == "A" and len(r.answer_a) > len(r.answer_b)
    )
    b_wins_b_longer = sum(
        1 for r in judge_results
        if r.final_winner == "B" and len(r.answer_b) > len(r.answer_a)
    )
    decisive = sum(1 for r in judge_results if r.final_winner != "tie")
    verbosity_bias = (a_wins_a_longer + b_wins_b_longer) / decisive if decisive > 0 else 0.0

    interpretation = ("Position bias cao — nên dùng swap-and-average."
                      if position_bias_rate > 0.3 else "Position bias thấp — judge ổn định.")
    return {
        "total_judged": total, "position_bias_rate": round(position_bias_rate, 3),
        "position_bias_count": position_bias_count,
        "verbosity_bias": round(verbosity_bias, 3),
        "verbosity_details": {"a_wins_a_longer": a_wins_a_longer,
                              "b_wins_b_longer": b_wins_b_longer,
                              "total_decisive": decisive},
        "interpretation": interpretation,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def _kappa_interpretation(k: float) -> str:
    if k < 0:    return "poor (worse than chance)"
    if k < 0.2:  return "slight"
    if k < 0.4:  return "fair"
    if k < 0.6:  return "moderate"
    if k < 0.8:  return "substantial"
    return "almost perfect"


if __name__ == "__main__":
    # --- Demo pairwise + swap trên một cặp answer ---
    q   = "Nhân viên được nghỉ bao nhiêu ngày phép năm?"
    a_a = "Nhân viên được nghỉ 15 ngày phép năm theo chính sách v2024 hiện hành."
    a_b = "Theo quy định, nhân viên có 12 ngày phép hàng năm."

    print("Running swap-and-average judge (demo)...")
    demo = swap_and_average(q, a_a, a_b)
    print(f"  Pass 1: {demo.winner_pass1} | Pass 2: {demo.winner_pass2} | "
          f"Final: {demo.final_winner} | consistent: {demo.position_consistent}")

    # --- Cohen's κ vs human labels (10 câu) ---
    # Judge label được sinh bằng cách so model_answer (A) với một baseline "không có thông tin" (B):
    # nếu model answer thắng → judge cho là answer tốt (1), ngược lại (0).
    with open(HUMAN_LABELS_PATH, encoding="utf-8") as f:
        human_data = json.load(f)
    print(f"\nJudging {len(human_data)} human-labeled answers...")

    BASELINE = "Xin lỗi, tôi không tìm thấy thông tin liên quan trong tài liệu."
    judge_results: list[JudgeResult] = []
    judge_labels:  list[int] = []
    per_question_labels: list[dict] = []

    for item in human_data:
        jr = swap_and_average(item["question"], item["model_answer"], BASELINE)
        judge_results.append(jr)
        label = 1 if jr.final_winner == "A" else 0   # A = model answer
        judge_labels.append(label)
        per_question_labels.append({
            "question_id": item["question_id"],
            "human_label": item["human_label"],
            "judge_label": label,
            "agree":       label == item["human_label"],
        })

    human_labels = [item["human_label"] for item in human_data]
    kappa = cohen_kappa(judge_labels, human_labels)
    bias  = bias_report(judge_results)

    print(f"Cohen's κ: {kappa:.3f} ({_kappa_interpretation(kappa)})")
    print(f"Agreement: {sum(p['agree'] for p in per_question_labels)}/{len(per_question_labels)}")
    print(f"Position bias rate: {bias['position_bias_rate']:.0%} | "
          f"Verbosity bias: {bias['verbosity_bias']:.0%}")

    # --- Save report ---
    os.makedirs("reports", exist_ok=True)
    report = {
        "judge_model": JUDGE_MODEL if OPENAI_API_KEY else OPENROUTER_MODEL,
        "cohen_kappa": round(kappa, 4),
        "kappa_interpretation": _kappa_interpretation(kappa),
        "agreement_count": sum(p["agree"] for p in per_question_labels),
        "per_question_labels": per_question_labels,
        "bias_report": bias,
        "demo_pair": {
            "question": q,
            "winner_pass1": demo.winner_pass1,
            "winner_pass2": demo.winner_pass2,
            "final_winner": demo.final_winner,
            "position_consistent": demo.position_consistent,
            "reasoning_pass1": demo.reasoning_pass1,
        },
    }
    with open("reports/judge_results.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("\nPhase B report saved → reports/judge_results.json")
