# Bảng thăm dò của Lộc bé — bàn giao 08/11/2026

**Nguồn:** team-reproduced exploratory evidence, run
`sage_pcdfusion_08nov_20261106_170517`. Chỉ có 10 ID FMB/test repackaged, 640×512;
chưa phải benchmark chính thức 280 ảnh. Xem [báo cáo](../../weekly/loc_be_2026-11-08.md)
và [hướng dẫn tái lập](../../../notebooks/loc_be/README.md).

| File | Nội dung / cách đọc |
|---|---|
| `pcdfusion_per_image.csv` | 40 dòng: 4 phương pháp × 10 ID; giữ `image_id` dạng chuỗi; 40 valid, 0 loại |
| `pcdfusion_summary.csv` | 32 dòng dạng phẳng: `method,metric,mean,std,median,count`; std mẫu, ddof=1 |
| `pcdfusion_paired_comparison.csv` | 10 cặp PCDFusion − SAGE; giữ nguyên số liệu gốc |
| `paired_summary_verified.csv` | Mean, paired median delta, số giá trị cao/thấp hơn; cao hơn không luôn tốt hơn |
| `paired_key_deltas.csv` | Một số cột paired delta để đọc nhanh |
| `roi_primary.csv` | 9 ROI thủ công, mean, ring 24 px |
| `roi_sensitivity_all.csv` | 54 dòng = 9 ROI × 3 ring (12/24/36 px) × 2 thống kê (mean/median) |
| `roi_robustness_flags.csv` | 9 dòng, kết quả qua cả 6 cấu hình mỗi ROI |
| `roi_near_white_clipping.csv` | Y trung bình và % gần trắng từng ROI; **`over240` nghĩa là Y ≥ 240, `over250` nghĩa là Y ≥ 250**, xác minh 36/36 ô ngưỡng trên PNG gốc |
| `roi_summary.json` | Tóm tắt audit gốc, giữ nguyên |
| `roi_validation_checks.json` | Kiểm tra từ gói bàn giao gốc; không phải log mới của phiên tích hợp |
| `file_provenance.csv` | SHA256 byte file của 40 PNG; đường dẫn tương đối theo tên hai gói kết quả |
| `manifest.csv` | 10 ID, shape, SHA256 IR/VIS/PCDFusion; SAGE là SHA256 pixel RGB, khác hash byte PNG |
| `run_config.json` | Config run gốc đã bỏ đường dẫn tài khoản/local; công thức và checkpoint giữ nguyên |
| `sage_environment.json`, `pcdfusion_environment.json` | Môi trường GPU gốc, không phải môi trường validation local |
| `source_provenance.json` | SHA256 nguồn và bản import; ghi rõ mọi thay đổi notebook/config/header |
| `offline_validation.json` | Log kiểm chứng CPU khi tích hợp: checksum, replay ROI, clipping, notebook schema |

Header pandas nhiều tầng trong `pcdfusion_summary.csv` gốc được đọc theo **cả hai
hàng header và hàng tên index**, rồi chuẩn hóa. 128 giá trị mean/std/median/count
khớp CSV per-image trong sai số làm tròn 2e-9. Không đổi công thức evaluation.

EN/SD dùng fused RGB→GRAY 8-bit; SCD/MS-SSIM và heuristic-mask dùng GRAY fused,
IR grayscale, VIS_Y. SD là độ lệch chuẩn dân số của pixel (ddof=0); `std` trong
bảng summary là độ lệch chuẩn mẫu giữa 10 ảnh (ddof=1). Audit **hình chữ nhật thủ
công** dùng Y của VIS/SAGE/PCDFusion và IR grayscale. Hai phép đo local dùng mask,
background và kênh khác nhau; không trộn chúng thành một metric.

Để kiểm tra bảng, chạy `python -m pytest tests/test_locbe_evidence.py`. Chạy lại
audit trên PNG ngoài Git theo README notebook. Không có ảnh/checkpoint/raw data
trong thư mục này. Kaggle Output URL: **link pending**.
