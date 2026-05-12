# Chương 4. Phương pháp

## 4.1. Tổng quan pipeline

Pipeline gồm 6 stage:

1. Data Preparation: tạo OHLCV H1.
2. Feature Engineering: sinh đặc trưng causal.
3. Label Generation: tạo nhãn triple-barrier.
4. Model Training: walk-forward Classic Hybrid Stacking.
5. Backtest: minh họa ứng dụng tín hiệu.
6. Reporting: tổng hợp metrics, comparison và biểu đồ.

Thiết kế theo stage giúp mỗi bước có input/output rõ ràng. Khi sửa label, pipeline chạy lại từ Stage 3; khi sửa feature, pipeline chạy lại từ Stage 2. Cách này giảm nguy cơ kết quả cũ bị trộn với cấu hình mới.

## 4.2. Công thức bài toán

Với mỗi thời điểm t, mô hình nhận vector đặc trưng X_t và dự báo:

```text
y_t ∈ {-1, 0, +1}
```

Trong đó:

- -1 là Short.
- 0 là Hold.
- +1 là Long.

Mục tiêu học máy là xấp xỉ xác suất có điều kiện:

```text
P(y_t = k | X_t), k ∈ {-1, 0, +1}
```

Tín hiệu cuối cùng lấy từ lớp có xác suất cao nhất. Trong backtest, có thể thêm confidence threshold để tránh giao dịch khi xác suất quá thấp.

## 4.3. Thiết kế nhãn triple-barrier

Tại mỗi thời điểm t, pipeline đặt ba barrier:

```text
TP_long  = close_t + 2.0 * ATR_t
SL_long  = close_t - 2.0 * ATR_t
horizon  = t + 24 bars
```

Logic nhãn:

- Nếu giá chạm upper barrier trước lower barrier: Long.
- Nếu giá chạm lower barrier trước upper barrier: Short.
- Nếu hết horizon chưa chạm: Hold.

Triple-barrier giúp nhãn phản ánh bài toán giao dịch có rủi ro thay vì chỉ phản ánh dấu của return sau N nến [5]. Vì ATR thay đổi theo biến động, khoảng TP/SL cũng thích nghi với điều kiện thị trường.

## 4.4. Walk-forward với purge/embargo

Dữ liệu được chia thành nhiều cửa sổ train/test theo thời gian. Ở mỗi cửa sổ:

1. Train set nằm trước test set.
2. Purge loại bỏ các mẫu train có event_end overlap với test.
3. Embargo tạo khoảng đệm sau test để giảm leakage.
4. Mô hình được huấn luyện lại và dự báo test window.
5. Dự báo ngoài mẫu được ghép lại thành OOF predictions.

Cách này phù hợp với khuyến nghị trong tài chính vì random K-fold có thể làm rò rỉ thông tin khi nhãn có horizon [5]. Nó cũng giảm rủi ro backtest overfitting được Bailey và cộng sự cảnh báo [7].

## 4.5. Base learners

### 4.5.1. Logistic Regression

Logistic Regression đóng vai trò baseline tuyến tính. Nếu Logistic Regression đã đạt kết quả gần với mô hình phức tạp, điều đó cho thấy tín hiệu có thể chủ yếu tuyến tính hoặc mô hình phức tạp chưa khai thác thêm được cấu trúc phi tuyến.

### 4.5.2. Random Forest

Random Forest học nhiều cây trên các mẫu bootstrap và feature subset [15]. Mô hình này có thể bắt quan hệ phi tuyến và tương tác feature, đồng thời giảm phương sai so với decision tree đơn.

### 4.5.3. LightGBM

LightGBM là mô hình boosting tree hiệu quả cho dữ liệu tabular, dùng GOSS và EFB để tăng tốc [17]. Trong đồ án, LightGBM vừa là base learner trong stacking vừa là ablation/baseline mạnh.

## 4.6. Classic Hybrid Stacking

Trong mỗi train window, pipeline thực hiện:

1. Chia train window theo thời gian thành base-train và meta-train.
2. Train Logistic Regression, Random Forest và LightGBM trên base-train.
3. Dự báo xác suất Short/Hold/Long trên meta-train.
4. Ghép xác suất thành meta-features:

```text
[p_lr_short, p_lr_hold, p_lr_long,
 p_rf_short, p_rf_hold, p_rf_long,
 p_lgbm_short, p_lgbm_hold, p_lgbm_long]
```

5. Train Logistic Regression meta-model trên meta-train.
6. Dự báo xác suất test window bằng base learners, sau đó qua meta-model.

Thiết kế này tuân theo stacked generalization [18] nhưng tránh lỗi phổ biến là train meta-model trên dự báo in-sample. Split base/meta theo thời gian giúp mô phỏng điều kiện dự báo ngoài mẫu tốt hơn.

## 4.7. Feature pruning

Sau lần chạy trước, feature importance cho thấy một số feature có đóng góp thấp. Pipeline đã giảm whitelist từ 25 xuống 21 feature, bỏ:

```text
regime_strength
upper_wick_ratio
lower_wick_ratio
volume_zscore_20
```

Mục tiêu của feature pruning không phải tối ưu quá mức, mà là giảm nhiễu, giảm độ phức tạp báo cáo và giữ tập feature dễ giải thích. Trong tài chính, thêm feature không luôn cải thiện kết quả vì có thể tăng overfitting và multiple testing risk [7], [9].

## 4.8. Backtest minh họa

Backtest nhận tín hiệu dự báo và mô phỏng giao dịch. Các giả định gồm:

- Vào lệnh theo tín hiệu Long/Short.
- Bỏ qua Hold.
- Có spread/slippage/commission.
- Có lot size và rule thoát lệnh.
- TP/SL trong backtest phải khớp logic labeling để tránh train một mục tiêu nhưng đánh giá một mục tiêu khác.

Backtest không phải bằng chứng chính vì kết quả phụ thuộc mạnh vào giả định thực thi. Classification metrics trong walk-forward mới là cơ sở đánh giá mô hình.

## 4.9. Quy trình thực nghiệm

Quy trình thực nghiệm chuẩn:

1. Chạy Stage 2 để tạo feature.
2. Chạy Stage 3 và kiểm tra phân phối nhãn.
3. Nếu nhãn quá lệch, chỉ chỉnh label trước.
4. Chạy Stage 4 để huấn luyện stacking và baseline.
5. Đọc `model_metrics.json` và `model_comparison.md`.
6. Nếu kết quả xấu, xem feature importance rồi giảm feature có cơ sở.
7. Chạy Stage 5/6 để tạo backtest demo và report.
8. Viết kết luận trung thực, không chọn kết quả đẹp bằng cách thử quá nhiều cấu hình.

## 4.10. Thiết kế validation chi tiết

Thiết kế validation kế thừa khung cũ nhưng được chỉnh cho runtime hiện tại là Classic Hybrid Stacking.

| Tham số | Ý nghĩa | Vai trò |
|---|---|---|
| `train_window_bars` | Độ dài cửa sổ train | Đảm bảo đủ dữ liệu quá khứ |
| `test_window_bars` | Độ dài cửa sổ test | Tạo đánh giá ngoài mẫu |
| `step_bars` | Bước trượt | Tránh test overlap quá nhiều |
| `purge_bars` | Khoảng loại mẫu có thể overlap label | Chống leakage do horizon |
| `embargo_bars` | Khoảng đệm sau test | Giảm ảnh hưởng lan truyền thông tin |
| `min_train_bars` | Số mẫu train tối thiểu | Tránh train cửa sổ quá nhỏ |

Ví dụ trực quan:

```text
Train window      Purge/Embargo      Test window
[ quá khứ ...... ][ khoảng đệm ][ tương lai cần dự báo ]
```

Trong mỗi cửa sổ, mô hình chỉ được học từ dữ liệu trước test window. Điều này mô phỏng tình huống triển khai: tại thời điểm ra quyết định, tương lai chưa tồn tại.

## 4.11. Event-time purge

Purge cố định theo số bar là cách đơn giản nhưng không hoàn hảo. Với triple-barrier, mỗi mẫu có `event_end` khác nhau: có mẫu chạm barrier sau 2 giờ, có mẫu sau 20 giờ, có mẫu đến tận horizon. Vì vậy event-time purge chính xác hơn:

```text
Giữ mẫu train nếu event_end_train < test_start
Loại mẫu train nếu event_end_train >= test_start
```

Cách này loại bỏ đúng các mẫu mà nhãn của chúng có thể dùng thông tin trùng với test window. Nó cũng tránh loại bỏ quá nhiều mẫu đã kết thúc sự kiện từ sớm.

## 4.12. Base/meta split trong stacking

Một lỗi phổ biến khi stacking là train base models và meta-model trên cùng một tập dự báo in-sample. Khi đó meta-model học từ dự báo quá đẹp vì base models đã thấy chính dữ liệu đó. Để tránh lỗi này, pipeline chia train window thành hai phần theo thời gian:

```text
Train window = [ base-train 80% ][ meta-train 20% ]
```

Quy trình:

1. Base learners học trên base-train.
2. Base learners dự báo xác suất trên meta-train.
3. Meta learner học trên xác suất meta-train.
4. Khi test, base learners dự báo test rồi meta learner kết hợp.

Split theo thời gian quan trọng hơn random split vì meta-train phải xảy ra sau base-train, gần với điều kiện dự báo tương lai.

## 4.13. Khung baseline chi tiết

Khung baseline hiện tại gồm:

| Nhóm | Mô hình | Mục đích |
|---|---|---|
| Không học | Naive Direction | Kiểm tra persistence hướng giá |
| Không học | Majority Baseline | Kiểm tra bias lớp đa số |
| Không học | Random Baseline | Floor ngẫu nhiên |
| Học máy tuyến tính | Logistic Regression | Baseline đơn giản, dễ giải thích |
| Học máy cây bagging | Random Forest | Bắt phi tuyến và interaction |
| Học máy boosting | LightGBM | Baseline tabular mạnh |
| Ensemble stacking | Hybrid Stacking | Mô hình đề xuất |

Báo cáo cần nhấn mạnh rằng mô hình đề xuất không được đánh giá một mình. Nó phải được đặt cạnh baseline. Nếu không vượt LightGBM, kết quả vẫn có giá trị vì chỉ ra rằng kiến trúc đơn giản hơn đang phù hợp hơn với dữ liệu hiện tại.

## 4.14. Cấu hình mô hình hiện tại

Cấu hình chính:

```toml
[model]
architecture = "stacking"
objective = "multiclass"
num_leaves = 15
max_depth = 4
learning_rate = 0.03
n_estimators = 300
min_child_samples = 80
feature_fraction = 0.70
reg_lambda = 10.0
stacking_base_models = ["logistic_regression", "random_forest", "lightgbm"]
stacking_meta_model = "logistic_regression"
stacking_meta_fraction = 0.20
```

Các tham số này thiên về regularization. Với dữ liệu tài chính nhiễu cao, việc giảm capacity thường an toàn hơn tăng complexity. `num_leaves=15`, `max_depth=4`, `min_child_samples=80` và `reg_lambda=10.0` đều nhằm hạn chế mô hình học nhiễu.

## 4.15. LightGBM trong vai trò baseline mạnh

LightGBM được giữ như baseline mạnh vì:

- Phù hợp dữ liệu tabular.
- Học được quan hệ phi tuyến.
- Xử lý interaction feature tốt.
- Huấn luyện nhanh hơn nhiều mô hình phức tạp.
- Có feature importance để giải thích.

Trong kết quả hiện tại, LightGBM vượt Hybrid Stacking. Vì vậy, khi viết báo cáo, không nên cố trình bày stacking như “tốt nhất tuyệt đối”. Thay vào đó, nên viết rằng stacking là kiến trúc đề xuất và được kiểm định công bằng; kết quả cho thấy LightGBM là ablation mạnh hơn trong lần chạy này.

## 4.16. Quy trình feature pruning

Feature pruning được thực hiện sau khi có kết quả feature importance. Các feature bị bỏ:

```text
regime_strength
upper_wick_ratio
lower_wick_ratio
volume_zscore_20
```

Quy trình đúng:

1. Chạy Stage 4 để có feature importance.
2. Xác định feature importance thấp hoặc khó bảo vệ.
3. Chỉ sửa whitelist model-facing.
4. Rerun từ Stage 2 vì feature output thay đổi.
5. So sánh metrics trước/sau.

Không nên xóa indicator khỏi code ngay vì indicator vẫn có thể hữu ích cho phân tích, chart hoặc thí nghiệm sau.

## 4.17. Backtest barrier guard

Một điểm thiết kế quan trọng là backtest TP/SL phải khớp label TP/SL. Nếu mô hình được train để dự báo sự kiện TP/SL 2.0 ATR nhưng backtest lại thoát lệnh theo rule khác, kết quả backtest sẽ không đánh giá đúng mục tiêu học.

Pipeline có guard:

```text
labels.tp == backtest.tp
labels.sl == backtest.sl
```

Nếu không khớp, pipeline raise error. Đây là lựa chọn tốt vì nó buộc nghiên cứu nhất quán giữa target học máy và mô phỏng ứng dụng.

## 4.18. Reproducibility

Để kết quả có thể lặp lại, pipeline cố định:

- Random seed.
- Config TOML.
- Feature whitelist.
- Window split.
- Output artifacts theo session timestamp.
- Metrics lưu thành JSON/Markdown.

Khi báo cáo kết quả, cần nêu rõ session artifact:

```text
results/XAUUSD_1H_20260513_023811/
```

Điều này giúp người đọc truy ngược từ số liệu trong luận văn về file kết quả cụ thể.

## 4.19. Tổng kết phương pháp

Phương pháp đề xuất không chỉ là một mô hình stacking. Nó là toàn bộ quy trình gồm label design, leakage control, model comparison và reporting. Trong tài chính, pipeline quan trọng vì một mô hình tốt nhưng validation sai sẽ cho kết luận sai. Thiết kế của đồ án ưu tiên tính kiểm soát, giải thích và khả năng bảo vệ học thuật hơn việc tối ưu một metric ngắn hạn.

## 4.20. Pseudo-code thuật toán tổng thể

Quy trình tổng thể có thể mô tả bằng pseudo-code:

```text
Input: OHLCV H1, config
Output: OOF predictions, metrics, backtest demo, report

1. Load OHLCV và kiểm tra timestamp
2. Tạo feature causal
3. Loại warm-up rows
4. Tạo triple-barrier labels
5. Loại censored rows
6. Tạo walk-forward windows
7. For each window:
   a. Tách train/test theo thời gian
   b. Purge train samples có event_end overlap test
   c. Chia train thành base-train và meta-train
   d. Train Logistic Regression, Random Forest, LightGBM trên base-train
   e. Dự báo xác suất meta-train
   f. Train Logistic Regression meta-model
   g. Dự báo test window
   h. Lưu OOF predictions và window metrics
8. Gộp OOF predictions
9. Tính metrics tổng và per-class
10. Chạy backtest demo
11. Sinh báo cáo
```

Pseudo-code này giúp hội đồng thấy rõ mô hình đề xuất không được train/test tùy tiện. Mọi dự báo test đều đến từ mô hình chỉ học từ quá khứ.

## 4.21. Kiểm soát lỗi triển khai

Pipeline có một số kiểm soát kỹ thuật:

- Config dataclass báo lỗi nếu có key lạ.
- Feature registry xác định output schema.
- EXCLUDE_COLS loại cột không được làm feature.
- Barrier guard đảm bảo label/backtest TP/SL khớp nhau.
- Ruff và compileall kiểm tra lỗi code trước khi chạy pipeline.
- Artifacts được ghi theo session để tránh ghi đè kết quả.

Các kiểm soát này không phải phần mô hình, nhưng rất quan trọng trong đồ án phần mềm. Chúng giúp kết quả có thể tái lập và giảm lỗi do thay đổi code/config.

## 4.22. Tổng kết mở rộng chương phương pháp

Chương phương pháp cần cho thấy ba điểm: mô hình được thiết kế có lý do, validation phù hợp tài chính, và pipeline có kiểm soát kỹ thuật. Nếu chỉ mô tả thuật toán stacking mà không mô tả label, purge/embargo và artifact, báo cáo sẽ thiếu phần quan trọng nhất của financial ML. Vì vậy bản cuối nên giữ đầy đủ các mục từ thiết kế nhãn đến reproducibility.
