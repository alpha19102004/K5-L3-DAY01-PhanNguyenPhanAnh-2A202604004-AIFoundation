# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **MSSV:** 2A202604004
- **Họ và tên:** Phan Nguyễn Phan Anh
- **SEED:** 2910

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, có output | ☑ Đã kiểm tra |
| [`lab1_report.json`](lab1_report.json) — ô "Báo cáo cuối" sinh ra | ☑ Đã kiểm tra|
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☑ Đã kiểm tra |



## 2. Kết quả chính

| Phương pháp | F1 test | F1 SUB (40 tin) | p95 (ms/tin) | USD/1k tin |
| --- | ---: | ---: | ---: | ---: |
| 1. BoW + Logistic Regression | 0.872 | 0.889 | 0.590 | 0.000 |
| 2. BoW + MLP | 0.942 | 0.947 | 1.138 | 0.000 |
| 3a. MiniLM + kNN | 0.873 | 0.927 | 73.308 | 0.000 |
| 3b. MiniLM + Logistic Regression | **0.955** | **1.000** | 41.168 | 0.000 |
| 3c. MiniLM + MLP head | 0.945 | **1.000** | 39.103 | 0.000 |
| 4a. LLM nhỏ zero-shot | —* | 0.000 | 1130.211 | 0.006 |
| 4b. LLM lớn zero-shot | —* | 0.000 | 1436.724 | 0.059 |
| 4c. LLM nhỏ few-shot | —* | 0.000 | 1353.838 | 0.018 |

*LLM chỉ được đánh giá trên 40 tin SUB, không có phép đo F1 trên toàn bộ tập test. Trong lần chạy này, cả ba cấu hình LLM đều có **40/40 đầu ra không hợp lệ (invalid)**; F1 bằng 0 phản ánh sự cố đầu ra/parser trong pipeline, không đủ để kết luận năng lực phân loại của bản thân LLM bằng 0.*

**Nhận xét:** MiniLM + LogReg đạt F1 test cao nhất (0.955). BoW + MLP đạt F1 0.942 với p95 chỉ 1.138 ms/tin. Encoder + kNN vượt SLA 50 ms/tin, còn hai bộ phân loại trên MiniLM đạt SLA theo phép đo từng tin trong notebook. Chi phí USD/1k bằng 0 cho các mô hình chạy cục bộ nghĩa là **không ghi nhận phí API**, không có nghĩa là không tốn tài nguyên tính toán.

## 3. Dự đoán và thực tế

1. **Learning rate (`ml_lr`):** Tôi dự đoán `lr=50` có thể khiến loss dao động hoặc tăng. Thực tế, loss dao động mạnh ở đầu nhưng vẫn giảm rất nhanh và đạt khoảng **0.006** sau 300 epoch; `lr=0.5` đạt khoảng **0.084**, `lr=0.001` khoảng **0.627**. Như vậy, learning rate lớn không nhất thiết gây phân kỳ trên bài toán cụ thể này, dù vẫn tiềm ẩn rủi ro ở dữ liệu khác.
2. **Bỏ ReLU (`dl_relu`):** Tôi dự đoán F1 có thể giảm hoặc xấp xỉ khi bỏ ReLU. Kết quả F1 **không đổi: 0.942**, so với MLP có ReLU cũng là 0.942. Với đầu vào BoW và bài toán này, lợi ích của phi tuyến chưa thể hiện rõ; mô hình tuyến tính có thể đã đủ tốt.
3. **LLM lớn so với nhỏ (`llm_models`):** Tôi dự đoán LLM lớn có thể có F1 tốt hơn nhưng chậm và đắt hơn. Thực tế, cả LLM nhỏ và lớn đều cho F1 SUB **0.000**, đồng thời có **40 đầu ra invalid**; LLM lớn chậm hơn (p95 1436.728 so với 1130.211 ms) và phí cao hơn (0.059 so với 0.006 USD/1k tin). Vì parser không nhận được nhãn hợp lệ, thí nghiệm này **chưa xác định được** LLM nào phân loại tốt hơn.

## 4. Phân tích 50 case khó

| Phương pháp | F1 SUB thường | F1 trên 50 case khó | Độ giảm F1 |
| --- | ---: | ---: | ---: |
| BoW + LogReg | 0.889 | 0.324 | **0.565** |
| BoW + MLP | 0.947 | 0.462 | 0.485 |
| MiniLM + kNN | 0.927 | 0.724 | 0.203 |
| MiniLM + LogReg | 1.000 | 0.717 | 0.283 |
| MiniLM + MLP head | 1.000 | 0.731 | 0.269 |
| LLM nhỏ zero-shot | 0.000 | 0.000 | 0.000* |
| LLM nhỏ + prompt phòng thủ | — | 0.000 | —* |

**Tụt nhiều nhất:** BoW + LogReg giảm **0.565 điểm F1** (0.889 → 0.324). Nhóm **"Spam giọng tự nhiên"** khó với BoW nhất: tỷ lệ đúng chỉ **10%**. Các mô hình embedding vẫn gặp khó ở nhóm **"Nói về spam"**, trong đó MiniLM + LogReg và MiniLM + MLP head đều đúng 25%.

**Hai case cụ thể:**
- **#14 — “Hi, I'm working from home and making about £300 a day…” (spam):** Câu được viết như trải nghiệm cá nhân nên thiếu dấu hiệu spam rõ ràng. Cả các mô hình trong stress test đều dự đoán sai; BoW không hiểu được ý đồ mời chào công việc đáng ngờ.
- **#16 — “Long time no see! I've started a little side business with skincare…” (spam):** Tin giả làm bạn cũ rồi gợi ý bán hàng. Cách diễn đạt thân mật làm nó giống tin ham; cả các mô hình đều sai trong thử nghiệm.

**Prompt injection:** LLM nhỏ zero-shot và prompt phòng thủ đều đạt 42% accuracy tổng thể, nhóm prompt injection đều 0% chính xác trong báo cáo. Không thể khẳng định prompt phòng thủ đã giúp ích trong lần chạy này; cần sửa hoặc xác nhận cơ chế parse nhãn trước khi so sánh khả năng chống tấn công.

*Ghi chú: F1 của các mô hình LLM trong notebook bị ảnh hưởng bởi đầu ra invalid; không diễn giải chênh lệch F1 = 0 là bằng chứng LLM hoàn toàn không có năng lực.*

## 5. Thử thách cá nhân

**Đề #1:** Thay ngưỡng quyết định của BoW + Logistic Regression từ 0.5 sang 0.2 và 0.8; báo Precision, Recall, F1; chọn ngưỡng phù hợp nếu ngân hàng cần tránh chặn nhầm tin OTP.

**Cách làm:** Dùng `predict_proba(X_te, w, b)` tạo xác suất spam, rồi chuyển thành nhãn theo công thức `y_pred = (probs >= threshold).astype(int)`. Tính `precision_score`, `recall_score`, `f1_score` trên cùng tập test.

```python
from sklearn.metrics import precision_score, recall_score, f1_score

probs = predict_proba(X_te, w, b)
for threshold in [0.2, 0.5, 0.8]:
    y_pred = (probs >= threshold).astype(int)
    print(threshold,
          precision_score(y_te, y_pred, zero_division=0),
          recall_score(y_te, y_pred, zero_division=0),
          f1_score(y_te, y_pred, zero_division=0))
```

| Threshold | Precision | Recall | F1 |
| ---: | ---: | ---: | ---: |
| 0.2 | 0.8911 | 0.8911 | **0.8911** |
| 0.5 | **1.0000** | 0.7723 | 0.8715 |
| 0.8 | **1.0000** | 0.6535 | 0.7904 |

**Giải thích:** Ngưỡng 0.2 bắt được nhiều spam hơn nhưng tăng nguy cơ chặn nhầm ham. Ngưỡng 0.8 bỏ sót nhiều spam hơn, đổi lại đòi hỏi xác suất cao mới chặn. Nếu ưu tiên tránh chặn nhầm OTP, tôi nghiêng về **0.8** và bổ sung cơ chế nhận diện/bảo vệ OTP riêng; tuy nhiên trong tập test này cả 0.5 và 0.8 đều đạt precision 1.0, nên chưa có bằng chứng ngưỡng 0.8 thực sự giảm false positive so với 0.5 trên dữ liệu hiện tại.

## 6. Phương án cho sản phẩm thật

Yêu cầu là **10 triệu SMS/ngày**, khoảng **116 SMS/giây trung bình**, và **p95 < 50 ms/tin**.

Tôi ưu tiên **BoW + MLP** cho lớp phân loại mặc định: F1 test **0.942**, p95 **1.138 ms/tin**, tốc độ benchmark cả lô **30,205 tin/giây**. BoW + LogReg còn nhanh hơn (p95 0.590 ms) nhưng F1 thấp hơn (0.872). MiniLM + LogReg có F1 cao hơn (0.955) và p95 41.168 ms, nhưng biên độ dự phòng với SLA nhỏ hơn và chi phí tính toán cao hơn.

Tôi đề xuất **router hai tầng**: BoW + MLP xử lý tin có xác suất rõ ràng; những tin có xác suất spam trong khoảng **[0.3, 0.7]** được chuyển sang MiniLM + LogReg. Trường hợp nghi ngờ nhưng liên quan OTP, tài khoản ngân hàng hoặc giao dịch quan trọng nên đưa vào luồng kiểm tra riêng thay vì tự động chặn. Ngưỡng router chỉ là đề xuất và cần hiệu chỉnh trên validation; p95 của toàn hệ thống phải được benchmark lại ở tải thực tế, không suy trực tiếp từ p95 từng mô hình.

LLM API trong notebook không phù hợp với đường xử lý chính ở SLA này: p95 khoảng **1.13–1.44 giây/tin**, cao hơn nhiều so với 50 ms. Theo chi phí USD/1k tin đã đo trong bảng, nếu gọi cho toàn bộ 10 triệu SMS/ngày trong 30 ngày, ước tính lần lượt **$1,800/tháng** (LLM nhỏ zero-shot), **$17,700/tháng** (LLM lớn zero-shot) hoặc **$5,400/tháng** (LLM nhỏ few-shot), chưa kể chi phí phụ trợ. Các số này chỉ là phép ngoại suy từ lần chạy notebook; hơn nữa pipeline LLM hiện đang có lỗi đầu ra invalid cần được khắc phục trước khi sử dụng.

## 7. AI log và điều đã học

| Pareto, SLA, stress test và prompt injection | Hỗ trợ diễn giải đánh đổi F1–độ trễ và nhóm lỗi | Đối chiếu bảng Phần 5 và 50 case khó; lưu ý F1 không đồng nghĩa accuracy |
| Báo cáo cuối và rà soát API key | Hỗ trợ tổ chức báo cáo, lưu ý bảo mật | Kiểm tra notebook đã chạy và output; cần tự kiểm tra lại các tệp bổ trợ trước khi nộp |

**Điều tôi học được:** Ban đầu tôi chưa hiểu Binary Bag-of-Words biến tin nhắn thành vector như thế nào; giờ tôi hiểu mỗi hàng là một tin nhắn và mỗi cột là một từ trong từ điển. Tôi cũng nhận ra mô hình phức tạp hơn không nhất thiết tốt hơn, đặc biệt khi stress test hoặc gặp vấn đề định dạng đầu ra LLM. Cuối cùng, việc chọn mô hình cho sản phẩm thật phải cân bằng F1, độ trễ, nguy cơ chặn nhầm và chi phí, thay vì chỉ chọn mô hình có điểm test cao nhất.
