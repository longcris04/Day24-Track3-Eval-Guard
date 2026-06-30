# LLM Judge Bias Report — Phase B

**Sinh viên:** longnh
**Ngày:** 2026-06-30
**Judge model:** google/gemini-2.5-flash-lite (qua OpenRouter)
**Nguồn:** `reports/judge_results.json`

> **Phương pháp tạo judge label:** với mỗi câu trong `human_labels_10q.json`, so `model_answer` (A)
> với một baseline "không có thông tin" (B) bằng `swap_and_average()`. Nếu model_answer thắng →
> judge_label = 1 (answer tốt), ngược lại = 0. Sau đó đối chiếu với `human_label`.

---

## 1. Pairwise Judge — ví dụ minh hoạ (demo pair)

| Câu hỏi | Answer A | Answer B | Winner pass1 | Winner pass2 | Final |
|---|---|---|---|---|---|
| Nhân viên được nghỉ bao nhiêu ngày phép năm? | "15 ngày (v2024)" | "12 ngày" | A | tie | **tie** |

**Reasoning (pass1):** *"Answer A is more precise by mentioning the specific policy version (v2024) and a higher number of days (15)… Answer B is less specific and potentially outdated."*

→ Pass 1 chọn A, nhưng khi **đảo thứ tự** (pass 2) judge lại hoà → đây chính là một ca **position bias**: kết luận thay đổi theo vị trí chứ không thuần theo chất lượng. `swap_and_average` quy về `tie` để tránh kết luận sai.

---

## 2. Swap-and-Average — Position Bias

| | Giá trị |
|---|---|
| Tổng số cặp đánh giá | 10 |
| Số ca **không nhất quán** khi swap | 2 |
| **Position bias rate** | **20%** |
| Diễn giải | Position bias thấp (<30%) — judge tương đối ổn định, swap-and-average xử lý tốt |

---

## 3. Cohen's κ Analysis

**Human labels:** `human_labels_10q.json` (10 câu, 5 label=1, 5 label=0)
**Judge labels:** sinh từ swap-and-average (model_answer vs baseline)

| Question ID | Human Label | Judge Label | Agree? |
|---|---|---|---|
| 1 | 1 | 1 | ✅ |
| 5 | 0 | 1 | ❌ |
| 12 | 1 | 1 | ✅ |
| 21 | 1 | 1 | ✅ |
| 23 | 1 | 1 | ✅ |
| 29 | 0 | 1 | ❌ |
| 33 | 1 | 1 | ✅ |
| 41 | 0 | 1 | ❌ |
| 46 | 1 | 0 | ❌ |
| 50 | 0 | 0 | ✅ |

**Agreement:** 6/10
**Cohen's κ:** **0.091**
**Interpretation:** **slight** (hầu như chỉ nhỉnh hơn ngẫu nhiên)

**Phân tích:** Judge gán "tốt" (1) cho **8/10** câu, trong khi human chỉ 5/10. Ba ca sai then chốt là q5, q29, q41 — human chấm **0 (answer dở)** nhưng judge chấm **1**. Nguyên nhân: baseline là câu "không có thông tin", nên *bất kỳ* câu trả lời thật nào — kể cả sai-nhưng-tự-tin — vẫn "thắng" baseline. Đây là **leniency bias**: phép so với baseline rỗng không phát hiện được lỗi factual. κ thấp ⇒ **không nên dùng judge kiểu này làm cổng chất lượng tự động** nếu chưa cải tiến (nên dùng pointwise rubric scoring hoặc so 2 candidate thật).

---

## 4. Verbosity Bias

Trong các ca có winner rõ ràng (không tie): **8 ca decisive**
- A thắng + A dài hơn B: **1 / 8**
- B thắng + B dài hơn A: **0 / 8**
- **Verbosity bias rate:** **12.5%**

**Kết luận:** Verbosity bias thấp. Lý do đặc thù: nhiều `model_answer` rất ngắn (vd "3 ngày làm việc") còn ngắn hơn cả baseline, nên judge không hề thiên vị câu dài. Trong môi trường thật (so 2 câu trả lời đầy đủ), cần đo lại verbosity bias vì LLM judge nổi tiếng hay ưu ái câu dài.

---

## 5. Nhận xét chung

> - **κ = 0.091 (slight)** → LLM judge với phương pháp baseline **chưa đáng tin** làm thước đo độc lập; cần human-in-the-loop hoặc rubric chặt hơn. (Chưa đạt mốc bonus κ>0.6.)
> - **Position bias 20%** — đáng chú ý nhưng chưa nghiêm trọng; **swap-and-average thực sự có ích** (ví dụ demo pair được cứu khỏi kết luận sai).
> - **Leniency bias** là vấn đề lớn nhất ở đây: judge over-rate vì so với baseline rỗng. Production nên: (1) chấm điểm theo rubric tuyệt đối thay vì so baseline, (2) luôn swap order, (3) chỉ dùng judge để *sàng lọc sơ bộ* rồi để người review các ca biên.
