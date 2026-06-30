# Failure Cluster Analysis — Phase A

**Sinh viên:** longnh
**Ngày:** 2026-06-30
**Nguồn:** `reports/ragas_50q.json` (RAGAS 4 metrics × 50 câu, judge = OpenRouter gemini-2.5-flash-lite `max_tokens=4096`, embeddings = all-MiniLM-L6-v2)

---

## 1. Aggregate RAGAS Scores theo Distribution

| Metric | factual | multi_hop | adversarial |
|---|---|---|---|
| faithfulness | 1.000 | 0.605 | 0.700 |
| answer_relevancy | 0.254 | 0.176 | 0.329 |
| context_precision | 0.958 | 0.417 | 0.933 |
| context_recall | 0.900 | 0.654 | 0.583 |
| **avg_score** | **0.778** | **0.463** | **0.637** |

**Quan sát chính:**
- Thứ hạng chất lượng: **factual (0.778) > adversarial (0.637) > multi_hop (0.463)**. `multi_hop` yếu nhất rõ rệt.
- `answer_relevancy` **thấp ở MỌI distribution** (0.18–0.33) → điểm yếu hệ thống. Câu trả lời gốc quá cộc lốc (vd "3 ngày làm việc") nên embedding của answer lệch xa embedding của question.
- `factual` grounded rất tốt (faithfulness 1.0, context_precision 0.958) nhưng vẫn rớt answer_relevancy → vấn đề ở khâu **generation/prompt**, không phải retrieval.
- `multi_hop` yếu cả retrieval (context_precision **0.417**) lẫn faithfulness (0.605): câu cần ghép nhiều tài liệu + tính toán khiến hệ thống lấy thiếu/sai chunk và dễ bịa số.

---

## 2. Bottom 10 Questions

| Rank | Distribution | id | Question | avg_score | worst_metric |
|---|---|---|---|---|---|
| 1 | multi_hop | 21 | Senior 9 năm thâm niên — số ngày phép + khoảng lương | 0.000 | faithfulness |
| 2 | multi_hop | 39 | So sánh mật khẩu policy v1.0 vs v2.0 (độ dài/thời hạn/MFA) | 0.000 | faithfulness |
| 3 | multi_hop | 33 | Manager 12 năm — tổng phụ cấp hàng tháng | 0.125 | answer_relevancy |
| 4 | multi_hop | 35 | Junior P1 lương 12 triệu vừa vào thử việc | 0.322 | context_precision |
| 5 | multi_hop | 22 | Mua laptop 30 triệu — ai duyệt + cần gì từ CNTT? | 0.333 | faithfulness |
| 6 | multi_hop | 24 | Tạm ứng 15 triệu, thanh toán trễ 20 ngày — phạt bao nhiêu? | 0.375 | faithfulness |
| 7 | multi_hop | 40 | Thử việc tháng 3 phát hiện vi phạm bảo mật — nên/không nên làm gì? | 0.375 | faithfulness |
| 8 | adversarial | 50 | Manager có được dùng VPN cá nhân (NordVPN) khi WFH? | 0.375 | faithfulness |
| 9 | multi_hop | 37 | Tự ý xóa malware + chia sẻ thông tin sự cố | 0.395 | context_precision |
| 10 | adversarial | 46 | Nhân viên thử việc có được nghỉ phép năm không? | 0.417 | faithfulness |

→ **8/10 câu tệ nhất là `multi_hop`** (2 adversarial, 0 factual). Các câu cần tính toán/ghép tài liệu (lương, phạt tạm ứng, phụ cấp) là khó nhất.

---

## 3. Failure Cluster Matrix

*(Mỗi ô = số câu có worst_metric = row, thuộc distribution = col)*

| worst_metric | factual | multi_hop | adversarial | Total |
|---|---|---|---|---|
| faithfulness | 0 | 6 | 3 | 9 |
| answer_relevancy | 19 | 6 | 6 | 31 |
| context_precision | 1 | 8 | 0 | 9 |
| context_recall | 0 | 0 | 1 | 1 |
| **Total** | **20** | **20** | **10** | **50** |

---

## 4. Dominant Failure Analysis

**Dominant distribution:** factual (theo công thức của lab — đếm worst_metric của *mọi* câu nên tổng mỗi nhóm = số câu; factual & multi_hop hòa 20, hàm `max` chọn factual)
**Dominant metric:** answer_relevancy (31/50 câu có đây là điểm yếu nhất)

**Lý do phân tích:**

> `answer_relevancy` là nút thắt rõ rệt nhất: 31/50 câu coi đây là metric tệ nhất, điểm trung bình chỉ 0.18–0.33 ở cả 3 nhóm. Nguyên nhân: bộ `answers_50q.json` chứa câu trả lời rất cộc lốc — đúng dữ kiện nhưng không diễn đạt thành câu bám sát câu hỏi, nên embedding answer lệch khỏi embedding question. Riêng `multi_hop` còn yếu cả retrieval (context_precision **0.417** — thấp nhất) vì một câu cần gộp nhiều chính sách + phép tính, mà rerank chỉ giữ 3 chunk nên dễ thiếu mảnh thông tin → kéo theo faithfulness 0.605 (LLM bịa số khi thiếu ngữ cảnh). Vì vậy multi_hop có avg thấp nhất (0.463) dù không phải "dominant" theo định nghĩa đếm-toàn-bộ của lab.

---

## 5. Suggested Fixes

| Metric yếu | Root cause | Suggested fix |
|---|---|---|
| answer_relevancy | Answer cộc lốc, không bám câu hỏi | Sửa prompt sinh đáp án: trả lời thành câu đầy đủ, nhắc lại chủ thể câu hỏi; tăng độ dài tối thiểu |
| faithfulness | LLM bịa khi thiếu ngữ cảnh (multi_hop) | Hạ temperature, ép "chỉ trả lời từ context"; (đã tăng max_tokens judge để tránh NaN) |
| context_precision | Giữ chunk thừa/nhiễu (multi_hop 0.417) | Cải thiện rerank, thêm metadata filter theo phiên bản policy (v2023/v2024, v1/v2) |
| context_recall | Thiếu chunk liên quan (adversarial 0.583) | Tăng RERANK_TOP_K cho câu multi_hop, cải thiện hybrid BM25+dense |

---

## 6. Nhận xét về Adversarial Distribution

> avg_score: **factual 0.778 > adversarial 0.637 > multi_hop 0.463**. Pipeline **bị adversarial làm khó** (thấp hơn factual ~0.14) — đạt tiêu chí bonus "adversarial avg < factual avg". Adversarial có context_recall thấp (0.583): các bẫy version-conflict (hỏi policy cũ v2023) khiến hệ thống lấy nhầm/thiếu đúng phiên bản. Trong bottom-10 có 2 câu adversarial: #8 "VPN cá nhân NordVPN khi WFH" và #10 "thử việc có được nghỉ phép năm" — đúng kiểu bẫy điều kiện/phủ định mà pipeline hay trả lời sai (cả hai worst_metric = faithfulness → bịa đáp án). Khắc phục bằng metadata filter theo phiên bản + version-awareness trong prompt.
