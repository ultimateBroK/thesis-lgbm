# Báo cáo luận văn

## Xây dựng pipeline học máy dự báo tín hiệu giao dịch XAU/USD H1 bằng Classic Hybrid Stacking

Thư mục này chứa bản thảo báo cáo học thuật cho đồ án. Nội dung được viết theo hướng nghiên cứu ứng dụng học máy trong dữ liệu tài chính, nhấn mạnh quy trình đánh giá đúng hơn là cam kết lợi nhuận giao dịch.

Kiến trúc runtime hiện tại:

```text
Logistic Regression + Random Forest + LightGBM
-> meta-model Logistic Regression
-> Short / Hold / Long
```

Trọng tâm báo cáo:

1. Cơ sở lý thuyết về dữ liệu tài chính: nhiễu, non-stationarity, fat tails, leakage và backtest overfitting.
2. Dữ liệu XAU/USD H1 và kiểm soát chất lượng dữ liệu.
3. Feature engineering causal, không dùng thông tin tương lai.
4. Triple-barrier labeling cho ba nhãn Short/Hold/Long.
5. Walk-forward validation với purge/embargo chống leakage.
6. So sánh Naive/Majority/Random baseline, Logistic Regression, Random Forest, LightGBM và Hybrid Stacking.
7. Backtest chỉ minh họa ứng dụng tín hiệu, không phải bằng chứng chính.

Kết quả gần nhất dùng để viết chương thực nghiệm:

```text
Session: results/XAUUSD_1H_20260513_023811/
Accuracy: 0.3416
Balanced Accuracy: 0.3675
Directional Accuracy: 0.4929
Macro F1: 0.3152
Backtest demo win rate: 47.17%
```

Các nguồn học thuật chính gồm Fama (1970), Cont (2001), Lo et al. (2000), López de Prado (2018), Bailey et al. (2017), Wolpert (1992), Breiman (2001), Friedman (2001), Ke et al. (2017), Lundberg & Lee (2017). Danh mục đầy đủ nằm trong `tai_lieu_tham_khao.md`.

## Khung chi tiết theo chương

Bản báo cáo nên được triển khai theo khung dài hơn bản tóm tắt kỹ thuật:

1. Chương 1 trình bày bối cảnh, động cơ chọn XAU/USD, bài toán nghiên cứu, câu hỏi nghiên cứu, mục tiêu, phạm vi, đóng góp và cấu trúc luận văn.
2. Chương 2 trình bày nền tảng học thuật: đặc tính dữ liệu tài chính, phân tích kỹ thuật dưới góc nhìn thống kê, triple-barrier labeling, walk-forward validation, purge/embargo, các mô hình học máy, metrics, SHAP và rủi ro backtest overfitting.
3. Chương 3 trình bày dữ liệu: nguồn, biến OHLCV, múi giờ, kiểm tra chất lượng, gap, warm-up, feature engineering causal, whitelist feature, gán nhãn và phân phối nhãn.
4. Chương 4 trình bày phương pháp đề xuất: pipeline 6 stage, thiết kế nhãn, thiết kế validation, baseline, base learners, Classic Hybrid Stacking, feature pruning, backtest guard và reproducibility.
5. Chương 5 trình bày thực nghiệm: môi trường, cấu hình, dữ liệu, mô hình so sánh, metrics, kết quả OOF, confusion matrix, high-confidence, backtest demo, phân tích lỗi và hạn chế.
6. Chương 6 trình bày minh họa ứng dụng: logic biến dự báo thành tín hiệu, giả định giao dịch, kết quả, caveats và điều kiện triển khai thực tế.
7. Chương 7 kết luận: tóm tắt đóng góp, kết quả thực nghiệm, hạn chế, hướng phát triển và thông điệp bảo vệ.

Nguyên tắc viết:

- Không viết theo hướng “mô hình chắc chắn kiếm tiền”.
- Viết theo hướng “pipeline đánh giá học máy tài chính có kiểm soát”.
- Kết quả không thắng baseline vẫn có giá trị nếu được phân tích trung thực.
- Mọi claim học thuật nên gắn với tài liệu tham khảo.
- Mọi claim thực nghiệm nên gắn với artifact trong `results/XAUUSD_1H_20260513_023811/`.

## Checklist trước khi nộp

- [ ] Các chương thống nhất runtime là Classic Hybrid Stacking.
- [ ] Không còn mô tả GRU là runtime chính.
- [ ] Chương 5 dùng số liệu thật mới nhất.
- [ ] Chương 6 nói rõ backtest là demo.
- [ ] Tài liệu tham khảo đủ nguồn nền tảng, không phụ thuộc vào blog/preprint yếu.
- [ ] Có phần hạn chế và hướng phát triển trung thực.
