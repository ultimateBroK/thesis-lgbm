# Chương 5. Thực nghiệm và kết quả

## 5.1. Thiết lập thực nghiệm

Thiết lập chính:

| Thành phần | Giá trị |
|---|---|
| Dữ liệu | XAU/USD H1 |
| Giai đoạn test/backtest | 2022-01-27 đến 2026-04-29 |
| Nhãn | Short/Hold/Long theo triple-barrier |
| TP/SL labeling | 2.0 ATR / 2.0 ATR |
| Horizon | 24 bars H1 |
| Validation | Walk-forward có purge/embargo |
| Runtime chính | Classic Hybrid Stacking |
| Phiên kết quả | `results/XAUUSD_1H_20260513_023811/` |

Các mô hình được so sánh:

1. Naive Direction baseline.
2. Majority baseline.
3. Random baseline.
4. Logistic Regression.
5. Random Forest.
6. LightGBM.
7. Hybrid Stacking.

## 5.2. Phân phối nhãn

Phân phối nhãn trong tập đánh giá:

```text
Short: 43.0%
Hold :  8.5%
Long : 48.5%
```

Lớp Hold thấp hơn mức lý tưởng. Tuy nhiên, thử `horizon_bars = 48` làm Hold giảm còn khoảng 1.5%, do đó cấu hình 48 bị loại. Điều này cho thấy trên dữ liệu XAU/USD H1, barrier 2.0 ATR thường bị chạm trước khi hết horizon. Đây là đặc điểm cần được thảo luận thay vì che giấu.

Mất cân bằng lớp ảnh hưởng đến cách diễn giải metrics. Accuracy có thể bị chi phối bởi Short/Long, trong khi Macro F1 phản ánh tốt hơn hiệu quả trên cả ba lớp.

## 5.3. Kết quả Hybrid Stacking

Kết quả chính:

```text
Total samples         23,910
Accuracy              0.3416
Balanced Accuracy     0.3675
Directional Accuracy  0.4929
Directional Baseline  0.5000
Majority Baseline     0.4850
Macro F1              0.3152
Weighted F1           0.3674
```

Per-class metrics:

| Lớp | True count | Pred count | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Short | 10,280 | 7,374 | 0.436 | 0.313 | 0.364 |
| Hold | 2,033 | 8,013 | 0.112 | 0.440 | 0.178 |
| Long | 11,597 | 8,523 | 0.476 | 0.350 | 0.404 |

Nhận xét:

- Mô hình có F1 tốt hơn ở Short và Long so với Hold.
- Hold có precision thấp vì mô hình dự báo Hold nhiều hơn số Hold thật.
- Directional Accuracy 0.4929 gần baseline 0.5, cho thấy tín hiệu hướng chưa đủ mạnh.
- Macro F1 0.3152 phản ánh bài toán ba lớp còn khó và chưa đạt mức triển khai thực tế.

## 5.4. Confusion matrix

Confusion matrix ba lớp:

| True \ Pred | Short | Hold | Long |
|---|---:|---:|---:|
| Short | 3,213 | 3,234 | 3,833 |
| Hold | 510 | 894 | 629 |
| Long | 3,651 | 3,885 | 4,061 |

Kết quả cho thấy mô hình nhầm khá nhiều giữa Short và Long, đồng thời cũng đẩy nhiều mẫu directional sang Hold. Điều này phù hợp với đặc thù dữ liệu tài chính nhiễu cao: tín hiệu kỹ thuật không ổn định qua mọi regime.

## 5.5. So sánh mô hình

Bảng so sánh:

| Model | Accuracy | Macro F1 | Long F1 | Short F1 | Ghi chú |
|---|---:|---:|---:|---:|---|
| Hybrid Stacking | 0.3416 | 0.3152 | 0.404 | 0.364 | Runtime chính |
| Naive Direction | 0.4574 | 0.3178 | - | - | Baseline direction |
| Majority Baseline | 0.4850 | 0.2177 | - | - | Dự báo lớp đa số |
| Random Baseline | 0.3361 | 0.3056 | - | - | Baseline ngẫu nhiên |
| Logistic Regression | 0.3568 | 0.3173 | - | - | Base model tuyến tính |
| Random Forest | 0.3596 | 0.3280 | - | - | Bagging tree |
| LightGBM | 0.3738 | 0.3265 | - | - | Boosting tree |

LightGBM đơn lẻ đạt Accuracy và Macro F1 cao hơn Hybrid Stacking trong lần chạy này. Đây không nên được xem là lỗi triển khai mặc định. Theo nguyên lý ensemble, stacking có thể cải thiện khi các base learners có lỗi bổ sung cho nhau; nhưng nếu base models học tín hiệu yếu hoặc lỗi tương quan cao, meta-model không nhất thiết vượt mô hình tốt nhất [18]. Với dữ liệu tài chính nhiễu cao, tăng độ phức tạp còn có thể làm tăng rủi ro overfit [7].

## 5.6. Kết quả high-confidence

Với threshold confidence 0.7:

```text
Số mẫu high-confidence: 182
Tỷ lệ trên tổng mẫu: 0.76%
Accuracy: 0.242
Directional Accuracy: 0.559
```

Directional Accuracy high-confidence cao hơn tổng thể, nhưng số mẫu chỉ khoảng 0.76%, chưa đủ để kết luận chắc chắn. Đây là hướng có thể phát triển thêm: calibration và thresholding theo confidence.

## 5.7. Backtest minh họa

Backtest demo cho kết quả:

```text
Initial equity  10,000
Total return    1.92%
Max drawdown   -2.72%
Sharpe ratio    0.384
Sortino ratio   0.637
Calmar ratio    0.138
Profit factor   1.109
Win rate        47.17%
Trades          159
```

Kết quả có return dương nhẹ, profit factor trên 1 và Sharpe thấp nhưng dương. Tuy nhiên Sharpe 0.384 vẫn ở mức chưa đủ mạnh để kết luận chiến lược có lợi thế giao dịch thực tế. Kết quả này chỉ chứng minh pipeline có thể chuyển tín hiệu ML thành giao dịch giả lập và đo lường rủi ro/lợi nhuận.

## 5.8. Diễn giải kết quả

Các kết luận chính:

1. Pipeline chạy được đầy đủ từ dữ liệu đến báo cáo.
2. Labeling và validation đã được thiết kế theo chuẩn phù hợp hơn cho dữ liệu tài chính.
3. LightGBM là baseline mạnh nhất trong lần chạy hiện tại.
4. Hybrid Stacking không vượt LightGBM, cho thấy ensemble phức tạp không đảm bảo cải thiện ngoài mẫu.
5. Lớp Hold là điểm yếu lớn nhất do phân phối thấp và precision kém.
6. Backtest demo cho kết quả dương nhẹ nhưng chưa đủ mạnh để khẳng định profitability.

Kết quả này vẫn có giá trị học thuật vì nó minh họa đầy đủ quy trình kiểm định có kiểm soát. Trong tài chính, một kết quả trung thực cho thấy mô hình chưa vượt baseline cũng quan trọng vì nó tránh overclaim và phù hợp với cảnh báo về backtest overfitting [7], [9].

## 5.9. Hạn chế thực nghiệm

- Chỉ sử dụng OHLCV và feature kỹ thuật, chưa có dữ liệu vĩ mô/tin tức/sentiment.
- Hold class thấp, làm bài toán ba lớp khó cân bằng.
- Chưa có calibration xác suất.
- Chưa kiểm tra robustness qua nhiều cấu hình chi phí giao dịch.
- Chưa phân tích SHAP theo từng regime.
- Chưa có out-of-time test trên dữ liệu sau 2026-04-30.

## 5.10. Môi trường thực nghiệm chi tiết

Môi trường thực nghiệm:

| Thành phần | Vai trò |
|---|---|
| Python | Ngôn ngữ triển khai pipeline |
| Pixi | Quản lý môi trường và command workflow |
| Polars | Xử lý dữ liệu dạng bảng/parquet hiệu năng cao |
| NumPy | Tính toán numeric |
| scikit-learn | Logistic Regression, Random Forest, metrics |
| LightGBM | Gradient boosting tree |
| Ruff | Kiểm tra style/lint |
| compileall | Kiểm tra lỗi cú pháp Python |

Các bước validation đã chạy trước khi viết báo cáo:

```text
pixi run ruff check src
pixi run python -m compileall -q src tests
pixi run python main.py --stage 2 --force
```

Stage 2 trở đi đã tạo session kết quả mới nhất dùng trong báo cáo.

## 5.11. Artifact thực nghiệm

Các artifact chính:

| File | Nội dung |
|---|---|
| `reports/model_metrics.json` | Metrics tổng, per-class, confusion matrix |
| `reports/model_comparison.md` | So sánh baseline/base models/stacking |
| `reports/feature_importance.json` | Feature importance sau training |
| `reports/walk_forward_history.json` | Lịch sử từng cửa sổ walk-forward |
| `backtest/backtest_results.json` | Kết quả backtest demo |
| `backtest/backtest_chart.html` | Biểu đồ backtest |

Việc lưu artifact giúp báo cáo không chỉ dựa trên mô tả miệng. Mỗi số liệu trong chương này có thể kiểm tra lại từ file kết quả.

## 5.12. Phân tích lỗi theo confusion matrix

Confusion matrix hiện tại:

| True \ Pred | Short | Hold | Long | Tổng true |
|---|---:|---:|---:|---:|
| Short | 3,213 | 3,234 | 3,833 | 10,280 |
| Hold | 510 | 894 | 629 | 2,033 |
| Long | 3,651 | 3,885 | 4,061 | 11,597 |

Các điểm chính:

1. True Short bị chia gần đều sang Short/Hold/Long, cho thấy mô hình chưa phân biệt tốt điều kiện giảm.
2. True Long cũng bị chia mạnh sang Short/Hold/Long, thể hiện tín hiệu hướng chưa ổn định.
3. Hold recall đạt 0.440 nhưng precision chỉ 0.112, nghĩa là mô hình dự báo Hold quá rộng so với số Hold thật.
4. Sai nhầm Short thành Long và Long thành Short là lỗi nghiêm trọng nhất nếu dùng giao dịch.
5. Sai nhầm directional thành Hold chủ yếu là bỏ lỡ cơ hội, ít nguy hiểm hơn vào sai hướng.

## 5.13. So sánh với baseline đa số

Majority Baseline có accuracy 0.4850 nhưng macro F1 chỉ 0.2177. Điều này minh họa vì sao accuracy không đủ. Dự báo lớp phổ biến nhất có thể đạt accuracy cao hơn Hybrid Stacking, nhưng không nhận diện cân bằng cả ba lớp. Macro F1 của Hybrid Stacking cao hơn Majority Baseline, dù accuracy thấp hơn.

Diễn giải nên viết như sau:

```text
Accuracy của Majority Baseline cao do phân phối lớp lệch. Tuy nhiên Macro F1 thấp cho thấy baseline này bỏ qua các lớp còn lại. Vì vậy báo cáo ưu tiên Macro F1 và per-class F1 khi đánh giá mô hình ba lớp.
```

## 5.14. So sánh với LightGBM

LightGBM đạt:

```text
Accuracy  0.3738
Macro F1  0.3265
```

Hybrid Stacking đạt:

```text
Accuracy  0.3416
Macro F1  0.3152
Long F1   0.404
Short F1  0.364
```

LightGBM tốt hơn về accuracy và macro F1; Hybrid Stacking có Long F1 và Short F1 khá nhưng không đủ bù đắp. Điều này cho thấy stacking không hoàn toàn vô ích, nhưng meta-model chưa tạo cải thiện tổng thể. Có thể lỗi giữa base learners tương quan cao, hoặc meta split làm giảm lượng dữ liệu train base models.

## 5.15. High-confidence analysis

High-confidence threshold 0.7 chỉ tạo 182 mẫu, chiếm 0.76% tổng số mẫu. Directional Accuracy tăng lên 0.559 nhưng mẫu quá ít. Có hai cách diễn giải:

- Tích cực: khi mô hình tự tin, tín hiệu hướng có vẻ tốt hơn.
- Thận trọng: số mẫu quá ít nên kết luận chưa chắc chắn, cần thêm kiểm định.

Hướng phát triển là calibration và threshold tuning, nhưng phải tránh overfit threshold trên cùng test set.

## 5.16. Feature importance sau pruning

Sau khi bỏ 4 feature importance thấp, feature set còn 21 feature. Mục tiêu không phải tạo kết quả tốt hơn ngay lập tức, mà là giảm nhiễu và giúp mô hình dễ giải thích. Báo cáo nên trình bày feature pruning như một quyết định kỹ thuật có kiểm soát:

1. Dựa trên kết quả feature importance thật.
2. Chỉ bỏ feature ít đóng góp.
3. Không thêm mô hình mới khi label/feature chưa ổn.
4. Rerun pipeline để kiểm chứng.

## 5.17. Phân tích backtest demo

Backtest có 159 giao dịch, win rate 47.17%, return 1.92%, profit factor 1.109. Đây là kết quả dương nhẹ nhưng Sharpe 0.384 vẫn ở mức thấp.

Một chiến lược thực tế cần:

- Profit factor lớn hơn 1 một cách ổn định.
- Sharpe đủ cao sau chi phí.
- Drawdown trong ngưỡng chấp nhận.
- Kết quả ổn định qua nhiều giai đoạn.
- Robustness với spread/slippage.

Kết quả hiện tại chưa đạt các điều kiện đó. Vì vậy backtest chỉ nên nằm ở vai trò minh họa.

## 5.18. Bài học từ kết quả âm

Một điểm quan trọng của nghiên cứu là không phải mọi mô hình đề xuất đều thắng. Kết quả Hybrid Stacking không vượt LightGBM cho thấy:

1. Dữ liệu tài chính có tín hiệu yếu.
2. Ensemble phức tạp có thể không giúp nếu base models không bổ sung lỗi cho nhau.
3. LightGBM là baseline rất mạnh cho feature tabular.
4. Label Hold thấp là nút thắt cần xử lý trước khi tăng complexity.
5. Báo cáo trung thực giúp tránh overclaim.

Đây là kết quả có thể bảo vệ được nếu luận văn nhấn mạnh phương pháp luận thay vì lợi nhuận.

## 5.19. Đề xuất bảng/biểu đồ nên đưa vào bản Word

Khi chuyển sang báo cáo Word/PDF, nên có:

1. Sơ đồ pipeline 6 stage.
2. Bảng cấu hình label.
3. Bảng danh sách feature groups.
4. Bảng phân phối nhãn.
5. Bảng metrics tổng.
6. Bảng per-class precision/recall/F1.
7. Confusion matrix.
8. Bảng model comparison.
9. Biểu đồ equity curve backtest.
10. Bảng hạn chế và hướng phát triển.

## 5.20. Kết luận thực nghiệm chi tiết

Thực nghiệm chứng minh pipeline hoạt động end-to-end và tạo được kết quả đo lường đầy đủ. Tuy nhiên mô hình chưa đạt hiệu quả đủ mạnh để triển khai giao dịch. Điểm mạnh của đồ án là quy trình đánh giá và báo cáo minh bạch; điểm yếu chính là chất lượng tín hiệu hướng và lớp Hold. Kết quả này dẫn trực tiếp đến hướng phát triển: cải thiện label design, calibration, thêm dữ liệu vĩ mô và phân tích SHAP theo regime.
