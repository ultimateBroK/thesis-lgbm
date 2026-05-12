# Chương 6. Minh họa ứng dụng tín hiệu

## 6.1. Vai trò của backtest

Backtest trong đồ án có vai trò minh họa cách tín hiệu Short/Hold/Long có thể được chuyển thành lệnh giao dịch giả lập. Trong nghiên cứu tài chính, backtest rất dễ bị overfit nếu nhà nghiên cứu thử nhiều tham số và chỉ chọn kết quả tốt nhất [7]. Vì vậy, đồ án không dùng backtest làm bằng chứng chính cho chất lượng mô hình.

Bằng chứng chính là kết quả classification ngoài mẫu trong walk-forward validation. Backtest chỉ trả lời câu hỏi phụ: nếu dùng tín hiệu dự báo để giao dịch theo một bộ quy tắc đơn giản, kết quả mô phỏng sẽ như thế nào?

## 6.2. Quy tắc minh họa

Quy tắc backtest tổng quát:

1. Nếu mô hình dự báo Long và vượt điều kiện confidence, mở hoặc giữ vị thế Long.
2. Nếu mô hình dự báo Short và vượt điều kiện confidence, mở hoặc giữ vị thế Short.
3. Nếu mô hình dự báo Hold, không vào lệnh mới.
4. Thoát lệnh theo TP/SL, tín hiệu đảo chiều hoặc điều kiện quản trị rủi ro.
5. Tính spread, slippage và commission.

TP/SL của backtest cần khớp logic label. Nếu label dùng TP/SL 2.0 ATR nhưng backtest dùng rule khác hoàn toàn, kết quả giao dịch sẽ không còn đánh giá đúng mục tiêu mà mô hình học.

## 6.3. Kết quả minh họa mới nhất

```text
Session: results/XAUUSD_1H_20260513_023811/
Period: 2022-01-27 -> 2026-04-29
Initial equity: 10,000
Total return: 1.92%
Max drawdown: -2.72%
Sharpe ratio: 0.384
Sortino ratio: 0.637
Calmar ratio: 0.138
Profit factor: 1.109
Win rate: 47.17%
Trades: 159
```

## 6.4. Ý nghĩa kết quả

Kết quả dương nhẹ (return 1.92%, profit factor 1.109) cho thấy tín hiệu có thể được đưa qua simulator nhưng chưa chứng minh lợi thế thực tế. Sharpe 0.384 và Calmar 0.138 vẫn ở mức thấp. Max drawdown khoảng 2.72% trong mô phỏng này chưa quá lớn, nhưng không đủ để khẳng định tính ổn định.

Backtest cũng phụ thuộc mạnh vào giả định thực thi. Trong thực tế, spread của XAU/USD thay đổi theo phiên và sự kiện tin tức; slippage có thể tăng khi biến động mạnh; thanh khoản và điều kiện broker cũng ảnh hưởng kết quả. Vì vậy, mọi kết luận triển khai cần thận trọng.

## 6.5. Điều kiện để tiến tới ứng dụng thực tế

Muốn nâng cấp từ demo sang nghiên cứu triển khai, cần thêm:

- Kiểm định ngoài mẫu trên dữ liệu mới chưa từng dùng khi phát triển.
- Phân tích độ nhạy theo spread, slippage, commission và lot size.
- Calibration xác suất và thresholding.
- Kiểm tra hiệu quả theo phiên giao dịch và regime biến động.
- Risk management độc lập với mô hình dự báo.
- Giới hạn số lần thử nghiệm để giảm backtest overfitting.

## 6.6. Kết luận chương

Backtest minh họa hoàn thành vai trò chứng minh pipeline có thể biến dự báo thành hành động giao dịch giả lập. Tuy nhiên, kết quả hiện tại chưa đủ để khẳng định chiến lược sinh lời. Luận văn nên trình bày backtest như phần ứng dụng phụ, còn trọng tâm học thuật nằm ở quy trình labeling, validation, model comparison và phân tích lỗi.

## 6.7. Thiết kế tín hiệu giao dịch từ xác suất mô hình

Mô hình classification không trực tiếp xuất lệnh giao dịch; nó xuất xác suất cho các lớp. Một lớp chuyển đổi tín hiệu cần quyết định:

```text
Nếu P(Long) lớn nhất và confidence đủ cao -> Long
Nếu P(Short) lớn nhất và confidence đủ cao -> Short
Nếu P(Hold) lớn nhất hoặc confidence thấp -> Không vào lệnh
```

Confidence có thể được định nghĩa là xác suất lớn nhất trong ba lớp:

```text
confidence = max(P(Short), P(Hold), P(Long))
```

Nếu threshold quá thấp, hệ thống giao dịch nhiều và dễ nhiễu. Nếu threshold quá cao, hệ thống giao dịch ít, có thể bỏ lỡ cơ hội. Kết quả high-confidence ở Chương 5 cho thấy threshold 0.7 tạo rất ít mẫu, vì vậy cần calibration trước khi dùng threshold như một quyết định thực tế.

## 6.8. Chi phí giao dịch và giả định thực thi

Backtest chỉ có ý nghĩa khi giả định chi phí được nêu rõ. Với XAU/USD CFD, chi phí có thể gồm:

- Spread giữa bid và ask.
- Commission theo lot.
- Slippage khi thị trường biến động mạnh.
- Swap/overnight fee nếu giữ qua ngày.
- Giới hạn margin và leverage.
- Khác biệt giá giữa broker và nguồn dữ liệu.

Nếu bỏ qua chi phí, backtest thường lạc quan. Trong kết quả hiện tại, profit factor chỉ vừa trên 1 và Sharpe vẫn thấp, vì vậy chỉ cần chi phí thực tế tăng nhẹ cũng có thể làm chiến lược xấu đi.

## 6.9. Quản trị rủi ro tối thiểu

Một ứng dụng thực tế không nên chỉ dựa vào tín hiệu mô hình. Cần thêm quản trị rủi ro:

| Thành phần | Vai trò |
|---|---|
| Max position size | Giới hạn rủi ro mỗi lệnh |
| Daily loss limit | Dừng giao dịch khi lỗ trong ngày vượt ngưỡng |
| Cooldown | Tránh vào lệnh liên tục sau tín hiệu nhiễu |
| Volatility filter | Giảm giao dịch khi biến động bất thường |
| Session filter | Chỉ giao dịch phiên có thanh khoản phù hợp |
| Confidence threshold | Chỉ vào lệnh khi xác suất đủ rõ |

Các thành phần này chưa phải trọng tâm đồ án, nhưng cần nêu để tránh hiểu nhầm rằng mô hình classification là đủ cho hệ thống giao dịch thực.

## 6.10. Phân biệt nghiên cứu và triển khai

Bảng sau phân biệt phạm vi của luận văn và yêu cầu triển khai thật:

| Hạng mục | Trong luận văn | Triển khai thật |
|---|---|---|
| Dữ liệu | Historical OHLCV | Realtime feed ổn định |
| Mô hình | Train offline | Retrain/monitor định kỳ |
| Đánh giá | Walk-forward + backtest demo | Paper trading + live monitoring |
| Chi phí | Giả định mô phỏng | Chi phí broker thực tế |
| Rủi ro | Mô tả cơ bản | Risk engine độc lập |
| Vận hành | Script/pipeline | Hệ thống giám sát lỗi |

Vì vậy, kết luận của Chương 6 chỉ nên nói: pipeline có thể minh họa cách tín hiệu được dùng trong giao dịch giả lập. Không nên nói: hệ thống đã sẵn sàng giao dịch thật.

## 6.11. Kịch bản cải thiện backtest

Nếu tiếp tục nghiên cứu, các thí nghiệm hợp lý gồm:

1. Dùng confidence threshold sau khi calibration.
2. Chỉ giao dịch khi LightGBM và Stacking đồng thuận.
3. Lọc theo phiên London/New York.
4. Lọc theo volatility regime.
5. Tối ưu position sizing ngoài tập test.
6. Kiểm tra sensitivity với spread/slippage.
7. Tách riêng giai đoạn validation cho rule giao dịch, không dùng test set để chọn threshold.

Điểm quan trọng là mọi cải thiện backtest phải có validation riêng. Nếu chọn rule dựa trên cùng kết quả backtest đã báo cáo, nguy cơ overfit rất cao.

## 6.12. Kết luận mở rộng chương ứng dụng

Chương ứng dụng cho thấy khoảng cách giữa mô hình dự báo và hệ thống giao dịch. Một mô hình có thể có metric classification tốt nhưng backtest kém do chi phí, timing và risk management. Ngược lại, một backtest đẹp cũng không đủ nếu classification evaluation bị leakage. Vì vậy luận văn đặt trọng tâm vào quy trình học máy trước, rồi dùng backtest như minh họa có kiểm soát.
