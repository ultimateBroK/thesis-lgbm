# Chương 7. Kết luận

## 7.1. Tổng kết

Đồ án đã xây dựng pipeline học máy dự báo tín hiệu XAU/USD H1 theo hướng đánh giá có kiểm soát cho chuỗi thời gian tài chính. Pipeline gồm 6 stage: chuẩn bị dữ liệu, tạo feature, gán nhãn, huấn luyện, backtest minh họa và báo cáo. Thiết kế này giúp tách biệt rõ dữ liệu, mô hình và đánh giá, đồng thời giảm nguy cơ rò rỉ thông tin.

## 7.2. Kết quả đạt được

Các kết quả chính:

1. Xây dựng dữ liệu OHLCV H1 và kiểm tra chất lượng dữ liệu.
2. Tạo feature causal từ OHLCV, tập trung vào trend, momentum, volatility, price position và session.
3. Áp dụng triple-barrier labeling với TP/SL theo ATR và horizon 24 bars.
4. Áp dụng walk-forward validation với purge/embargo để giảm leakage.
5. Triển khai Classic Hybrid Stacking gồm Logistic Regression, Random Forest, LightGBM và meta Logistic Regression.
6. So sánh với baseline và base models.
7. Sinh báo cáo classification metrics, confusion matrix, model comparison và backtest demo.

## 7.3. Kết quả thực nghiệm chính

Trong lần chạy mới nhất:

```text
Hybrid Stacking Accuracy  0.3416
Hybrid Stacking Macro F1  0.3152
LightGBM Accuracy         0.3738
LightGBM Macro F1         0.3265
Backtest Win Rate         47.17%
Backtest Total Return     1.92%
```

Hybrid Stacking không vượt LightGBM đơn lẻ. Đây là kết quả cần trình bày trung thực: trên dữ liệu tài chính nhiễu cao, tăng độ phức tạp mô hình không luôn tạo cải thiện ngoài mẫu. Đóng góp của đồ án không nằm ở việc chứng minh chiến lược giao dịch sinh lời, mà ở việc xây dựng quy trình đánh giá có kiểm soát và minh bạch.

## 7.4. Đóng góp học thuật và kỹ thuật

Đồ án có các đóng góp sau:

- Áp dụng triple-barrier labeling thay vì nhãn tăng/giảm đơn giản.
- Áp dụng walk-forward validation thay vì random split.
- Sử dụng purge/embargo để giảm leakage do event horizon.
- So sánh nhiều baseline và mô hình thay vì chỉ báo cáo mô hình chính.
- Phân tích per-class metrics để phát hiện điểm yếu lớp Hold.
- Tách backtest khỏi bằng chứng chính, tránh overclaim về profitability.
- Ghi nhận kết quả âm hoặc chưa vượt baseline như một phần hợp lệ của nghiên cứu.

## 7.5. Hạn chế

Các hạn chế chính:

1. Feature chỉ dựa trên OHLCV, chưa có dữ liệu vĩ mô, tin tức, sentiment hoặc order book.
2. Lớp Hold thấp, làm bài toán ba lớp khó cân bằng.
3. Directional Accuracy chưa vượt baseline rõ ràng.
4. Hybrid Stacking chưa cải thiện so với LightGBM.
5. Backtest demo chưa tính đầy đủ mọi điều kiện thực thi thực tế.
6. Chưa có phân tích SHAP chi tiết theo từng giai đoạn thị trường.

## 7.6. Hướng phát triển

Các hướng phát triển hợp lý:

1. Thiết kế lại nhãn để tăng tỷ lệ Hold hợp lý hơn, ví dụ điều chỉnh barrier hoặc định nghĩa no-trade zone.
2. Calibration xác suất để dùng confidence threshold tốt hơn [23].
3. Phân tích SHAP theo regime để hiểu mô hình dựa vào feature nào [22].
4. Thử thêm dữ liệu vĩ mô như lãi suất, DXY, yield, CPI hoặc biến động thị trường.
5. Kiểm tra robustness theo chi phí giao dịch và giai đoạn thị trường.
6. Giữ LightGBM-only như baseline mạnh; chỉ thêm mô hình phức tạp nếu có giả thuyết rõ ràng.
7. Chỉ xem xét deep sequence models sau khi pipeline classical đã ổn định và có thêm dữ liệu/kiểm định phù hợp.

## 7.7. Kết luận cuối

Đề tài đã hoàn thành mục tiêu xây dựng một pipeline học máy có kiểm soát cho dự báo tín hiệu XAU/USD H1. Kết quả thực nghiệm cho thấy bài toán khó và mô hình phức tạp không tự động vượt baseline mạnh. Đây là kết luận phù hợp với đặc thù tài chính: tín hiệu yếu, dữ liệu nhiễu và nguy cơ overfitting cao. Giá trị chính của đồ án là phương pháp luận: dữ liệu causal, nhãn có ý nghĩa giao dịch, validation đúng theo thời gian, so sánh baseline và báo cáo trung thực.

## 7.8. Trả lời câu hỏi nghiên cứu

### Câu hỏi 1: Pipeline có tránh được thông tin tương lai không?

Pipeline đã áp dụng nhiều lớp kiểm soát: feature causal, loại bỏ cột label/barrier khỏi feature, walk-forward split, event-time purge và embargo. Điều này không đảm bảo tuyệt đối không có mọi dạng leakage, nhưng tốt hơn đáng kể so với random split thông thường.

### Câu hỏi 2: Triple-barrier có phù hợp hơn nhãn tăng/giảm đơn giản không?

Triple-barrier phù hợp hơn vì nhãn gắn với TP, SL và horizon. Nó phản ánh câu hỏi giao dịch thực tế: trong một khoảng thời gian hữu hạn, giá chạm mục tiêu lợi nhuận hay rủi ro trước. Tuy nhiên, phân phối Hold thấp cho thấy thiết kế barrier vẫn cần cải thiện.

### Câu hỏi 3: Classic Hybrid Stacking có vượt mô hình đơn lẻ không?

Trong lần chạy hiện tại, không. LightGBM có Macro F1 cao hơn Hybrid Stacking. Đây là kết quả quan trọng vì nó cho thấy không nên mặc định mô hình phức tạp hơn sẽ tốt hơn. Kết quả này cũng củng cố vai trò của baseline comparison.

### Câu hỏi 4: Backtest có chứng minh chiến lược sinh lời không?

Không. Backtest có return dương nhẹ (1.92%) nhưng profit factor chỉ vừa trên 1 và Sharpe vẫn thấp. Nó chỉ chứng minh pipeline có thể tạo tín hiệu và mô phỏng giao dịch, không chứng minh lợi thế giao dịch thực tế.

## 7.9. Bài học phương pháp luận

Các bài học chính:

1. Trong tài chính, validation quan trọng ngang mô hình.
2. Label design quyết định mô hình học bài toán gì.
3. Baseline mạnh là bắt buộc để tránh overclaim.
4. Accuracy có thể gây hiểu lầm khi class imbalance.
5. Backtest đẹp không đủ nếu không kiểm soát overfitting.
6. Mô hình phức tạp không tự động tạo hiệu quả ngoài mẫu.
7. Kết quả không như kỳ vọng vẫn có giá trị nếu được phân tích đúng.

## 7.10. Khuyến nghị cho bản bảo vệ

Khi bảo vệ, nên nhấn mạnh:

- "Hybrid" ở đây là stacking nhiều họ mô hình: tuyến tính, bagging tree, boosting tree.
- Đóng góp chính là pipeline đánh giá có kiểm soát.
- Kết quả LightGBM vượt Stacking được trình bày trung thực.
- Backtest là demo ứng dụng, không phải claim lợi nhuận.
- Hạn chế lớp Hold thấp đã được nhận diện và có hướng cải thiện.

Nếu hội đồng hỏi vì sao kết quả chưa cao, có thể trả lời:

```text
Dữ liệu tài chính có tín hiệu yếu và non-stationary. Đề tài không cố tối ưu để có backtest đẹp, mà ưu tiên đánh giá đúng, tránh leakage và so sánh baseline. Việc LightGBM vượt Stacking là kết quả thực nghiệm có ý nghĩa: tăng độ phức tạp không đảm bảo cải thiện ngoài mẫu.
```

## 7.11. Công việc cần làm nếu có thêm thời gian

Ưu tiên tiếp theo nên là:

1. Thiết kế lại vùng Hold bằng no-trade zone hoặc barrier/horizon khác.
2. Tạo thêm feature vĩ mô: DXY, US10Y yield, real yield proxy, CPI/FOMC event flags.
3. Calibration xác suất trước khi dùng confidence threshold.
4. SHAP analysis cho LightGBM và Stacking.
5. Robustness test với nhiều mức chi phí giao dịch.
6. Kiểm tra rolling retrain schedule.
7. Tạo out-of-time test sau 2026-04-30.
8. So sánh với rule-based technical strategies đơn giản.

## 7.12. Kết luận bảo vệ cuối cùng

Luận văn nên được bảo vệ như một nghiên cứu xây dựng và kiểm định pipeline học máy tài chính. Thành công của đồ án nằm ở việc biến một bài toán dễ bị overfit thành một quy trình có kiểm soát: dữ liệu có contract, feature causal, label event-based, validation theo thời gian, purge/embargo, baseline comparison, metrics đa chiều và backtest minh họa. Kết quả thực nghiệm hiện tại chưa chứng minh lợi thế giao dịch, nhưng cung cấp nền tảng rõ ràng để tiếp tục cải thiện một cách khoa học.
