# CI/CD Blueprint: RAG Eval + Guardrail Stack

**Sinh viên:** longnh
**Ngày:** 2026-06-30
**Nguồn số liệu:** `reports/ragas_50q.json` (A), `reports/judge_results.json` (B), `reports/guard_results.json` (C).
**Môi trường chạy:** LLM = OpenRouter `google/gemini-2.5-flash-lite`; Phase C chạy trong env riêng `lab24nemo` (nemoguardrails 0.10 + langchain<0.3 + Presidio).

---

## Guard Stack Architecture

```
User Input
    │
    ▼ (Presidio P95 ≈ 7.5ms)
[Presidio PII Scan]
    │ block if: VN_CCCD / VN_PHONE / EMAIL detected
    │ action:   return 400 + "PII detected in query"
    ▼ (NeMo input rail P95 ≈ 2843ms — LLM self_check_input)
[NeMo Input Rail]
    │ block if: jailbreak / off-topic / prompt injection / đòi PII người khác
    │ action:   refuse message tiếng Việt
    ▼
[RAG Pipeline (Day 18)]
    │ M1 Chunk → M2 Search → M3 Rerank → LLM (OpenRouter)
    ▼
[NeMo Output Rail (self_check_output)]
    │ flag if:  PII / mật khẩu / dữ liệu bí mật trong response
    │ action:   thay bằng refuse message
    ▼
User Response
```

---

## Latency Budget (đo thực tế — Task 12, n=10)

| Layer | P50 (ms) | P95 (ms) | P99 (ms) | Budget | Đạt? |
|---|---|---|---|---|---|
| Presidio PII | 6.61 | **7.52** | 7.52 | <10ms | ✅ |
| NeMo Input Rail | 663.26 | **2842.70** | 2842.70 | <300ms | ❌ |
| RAG Pipeline | — | — | — | <2000ms | (không đo ở Task 12) |
| NeMo Output Rail | — | — | — | <300ms | (đo gộp trong generation) |
| **Total Guard (Presidio + NeMo input)** | 670.78 | **2849.18** | 2849.18 | **<500ms** | **❌** |

**Budget OK?** [ ] Yes / [x] **No**
**Comment:** Bottleneck rõ ràng là **NeMo Input Rail** — nó gọi LLM (`self_check_input`) qua OpenRouter nên P95 ≈ 2.8s, vượt xa ngân sách 300ms và kéo tổng vượt 500ms. Presidio (regex cục bộ) chỉ ~7.5ms, gần như miễn phí. Hướng tối ưu: (1) cache kết quả self-check theo hash input; (2) dùng model nhỏ/nhanh hơn hoặc self-hosted cho rail; (3) thay LLM self-check bằng classifier nhẹ (heuristics/embedding) cho các pattern phổ biến, chỉ gọi LLM cho ca mơ hồ; (4) chạy Presidio song song với NeMo.

---

## CI/CD Gates (phải pass trước khi merge to main)

```yaml
# .github/workflows/rag_eval.yml
- name: RAGAS Quality Gate
  run: python src/phase_a_ragas.py
  env:
    MIN_FAITHFULNESS: 0.75
    MIN_AVG_SCORE: 0.65

- name: Guardrail Gate
  run: pytest tests/test_phase_c.py -k "test_adversarial_suite_pass_rate"
  # phải ≥ 15/20 (75%) — thực tế đạt 20/20

- name: Latency Gate
  run: python -c "from src.phase_c_guard import measure_p95_latency; ..."
  # P95 total < 500ms — HIỆN ĐANG FAIL (2849ms) → cần tối ưu NeMo rail trước khi bật gate này
```

---

## Monitoring Dashboard (production)

| Metric | Alert Threshold | Action |
|---|---|---|
| RAGAS faithfulness (daily sample) | < 0.70 | Page on-call |
| Adversarial block rate | < 80% | Review new attack patterns |
| Guard P95 latency | > 600ms | Scale/optimize NeMo rail |
| PII detected count | spike >10/hour | Security alert |

---

## Kết quả thực tế từ Lab

| | Kết quả |
|---|---|
| RAGAS avg_score (50q) | factual **0.778**, adversarial **0.637**, multi_hop **0.463** (yếu nhất) |
| Worst metric | **answer_relevancy** (31/50 câu, ~0.18–0.33) |
| Dominant failure distribution | **factual** (theo công thức lab); thực chất **multi_hop** yếu retrieval nhất (ctx_precision 0.417) |
| Cohen's κ | **0.091** (slight) — judge baseline-method chưa đáng tin |
| Adversarial pass rate | **20 / 20** (100%) — Presidio chặn 4, NeMo chặn 16 |
| Guard P95 latency | **2849ms** (Presidio 7.5ms + NeMo 2843ms) → vượt budget 500ms |

**Bonus đạt được:** pass rate ≥18/20 ✅ · adversarial avg (0.637) < factual avg (0.778) ✅ · (κ>0.6 ❌).

---

## Nhận xét & Cải tiến

> - **Hoạt động tốt:** Lớp Presidio bắt PII tiếng Việt (CCCD/SĐT/email) chính xác 4/4 và cực nhanh (~7.5ms) — gần như miễn phí về latency. NeMo self-check (LLM Yes/No) chặn đúng 16/16 ca jailbreak / off-topic / prompt-injection / đòi PII, kể cả các injection nhúng trong câu hỏi HR hợp lệ (id 16–20).
> - **Cần cải thiện:** (1) **Latency** — NeMo rail gọi LLM nên P95 ~2.8s, không thể đáp ứng budget 500ms; production phải cache hoặc thay bằng classifier nhẹ. (2) **Chất lượng RAG** — answer_relevancy thấp toàn cục do answers quá cộc lốc; cần sửa prompt sinh đáp án. (3) **LLM-judge** κ chỉ 0.091 → không dùng làm cổng tự động khi chưa cải tiến phương pháp.
> - **Nếu deploy production:** đặt Presidio làm lớp chặn cứng đầu tiên (rẻ, nhanh), đưa NeMo self-check sang chế độ async + cache, log mọi lần block PII để audit, và chạy RAGAS hằng ngày trên sample để phát hiện drift. Chỉ bật Latency Gate sau khi đã tối ưu NeMo.
