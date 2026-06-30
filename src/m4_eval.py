from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH, OPENROUTER_API_KEY, OPENROUTER_BASE_URL, OPENROUTER_MODEL, OPENAI_API_KEY

# Import ragas at module level so Python caches it correctly in sys.modules.
# Lazy imports inside functions can leave ragas in a broken state after a partial failure,
# causing "No module named 'ragas'" on every subsequent call in the same process.
try:
    from ragas import evaluate as _ragas_evaluate
    from ragas.metrics import (faithfulness as _faithfulness,
                               answer_relevancy as _answer_relevancy,
                               context_precision as _context_precision,
                               context_recall as _context_recall)
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from langchain_huggingface import HuggingFaceEmbeddings
    from datasets import Dataset as HFDataset
    _RAGAS_AVAILABLE = True
except Exception as _ragas_import_err:
    _RAGAS_AVAILABLE = False
    import traceback as _tb
    print(f"  ⚠️  ragas import failed: {type(_ragas_import_err).__name__}: {_ragas_import_err}")
    _tb.print_exc()


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation using OpenRouter (or OpenAI if available)."""
    _zero = {"faithfulness": 0.0, "answer_relevancy": 0.0,
             "context_precision": 0.0, "context_recall": 0.0, "per_question": []}

    if not _RAGAS_AVAILABLE:
        print("  ⚠️  ragas not available. Skipping evaluation.")
        return _zero

    try:
        api_key = OPENAI_API_KEY or OPENROUTER_API_KEY
        if not api_key:
            print("  ⚠️  No API key found. Skipping RAGAS evaluation.")
            return _zero

        if OPENAI_API_KEY:
            ragas_llm = LangchainLLMWrapper(
                ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY, temperature=0)
            )
            ragas_embeddings = LangchainEmbeddingsWrapper(
                OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)
            )
        else:
            # max_tokens cao để judge không bị cắt giữa chừng → tránh
            # LLMDidNotFinishException khiến RAGAS trả NaN cho metric đó.
            ragas_llm = LangchainLLMWrapper(
                ChatOpenAI(model=OPENROUTER_MODEL, api_key=OPENROUTER_API_KEY,
                           base_url=OPENROUTER_BASE_URL, temperature=0, max_tokens=4096)
            )
            ragas_embeddings = LangchainEmbeddingsWrapper(
                HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            )

        metrics = [_faithfulness, _answer_relevancy, _context_precision, _context_recall]
        for m in metrics:
            m.llm = ragas_llm
        _answer_relevancy.embeddings = ragas_embeddings

        dataset = HFDataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        })
        result = _ragas_evaluate(dataset, metrics=metrics)
        df = result.to_pandas()

        # ragas 0.2.x renamed columns: question→user_input, answer→response,
        # contexts→retrieved_contexts, ground_truth→reference
        def _col(row, *names, default=""):
            for n in names:
                if n in row.index and row[n] is not None:
                    return row[n]
            return default

        per_question = [
            EvalResult(
                question=_col(row, "question", "user_input"),
                answer=_col(row, "answer", "response"),
                contexts=_col(row, "contexts", "retrieved_contexts", default=[]),
                ground_truth=_col(row, "ground_truth", "reference"),
                faithfulness=float(_col(row, "faithfulness", default=0.0) or 0.0),
                answer_relevancy=float(_col(row, "answer_relevancy", default=0.0) or 0.0),
                context_precision=float(_col(row, "context_precision", default=0.0) or 0.0),
                context_recall=float(_col(row, "context_recall", default=0.0) or 0.0),
            )
            for _, row in df.iterrows()
        ]

        def _mean(col, *aliases):
            for name in (col, *aliases):
                if name in df.columns:
                    return float(df[name].mean())
            return 0.0

        return {
            "faithfulness": _mean("faithfulness"),
            "answer_relevancy": _mean("answer_relevancy"),
            "context_precision": _mean("context_precision"),
            "context_recall": _mean("context_recall"),
            "per_question": per_question,
        }
    except Exception as e:
        # Use ascii-safe print to avoid UnicodeEncodeError on Windows cp1252 console
        msg = str(e).encode("ascii", errors="replace").decode("ascii")
        print(f"  [WARN] RAGAS evaluation failed: {msg}")
        import traceback; traceback.print_exc()
        return _zero


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    if not eval_results:
        return []

    diagnostic_tree = {
        "faithfulness": ("LLM hallucinating", "Tighten prompt, lower temperature"),
        "context_recall": ("Missing relevant chunks", "Improve chunking or add BM25"),
        "context_precision": ("Too many irrelevant chunks", "Add reranking or metadata filter"),
        "answer_relevancy": ("Answer doesn't match question", "Improve prompt template"),
    }

    analyzed = []
    for r in eval_results:
        scores = {
            "faithfulness": r.faithfulness,
            "answer_relevancy": r.answer_relevancy,
            "context_precision": r.context_precision,
            "context_recall": r.context_recall,
        }
        avg = sum(scores.values()) / len(scores)
        worst_metric = min(scores, key=lambda m: scores[m])
        diagnosis, suggested_fix = diagnostic_tree[worst_metric]
        analyzed.append({
            "question": r.question,
            "avg_score": avg,
            "worst_metric": worst_metric,
            "score": scores[worst_metric],
            "diagnosis": diagnosis,
            "suggested_fix": suggested_fix,
        })

    analyzed.sort(key=lambda x: x["avg_score"])
    return analyzed[:bottom_n]


def save_report(results: dict, failures: list[dict], path: str = "ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
