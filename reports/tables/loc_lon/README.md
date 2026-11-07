# Bảng kiểm chứng Metric — Lộc lớn (Evaluation)

Thư mục này lưu trữ các bảng tra cứu, kiểm chứng toán học và đối chuẩn cho các chỉ số đánh giá
hợp nhất ảnh hồng ngoại – khả kiến (IVIF): **EN**, **SD**, **SCD**, và **MS-SSIM**.

## 1. Danh sách bảng

| File | Nội dung chính |
|---|---|
| [`metric_definitions_reference.csv`](metric_definitions_reference.csv) | Định nghĩa toán học, citation gốc, miền giá trị lý thuyết và trạng thái kiểm chứng. |
| [`reference_cases_verification.csv`](reference_cases_verification.csv) | Kết quả kiểm chứng trên 8 ca đối chứng nhân tạo (nhiễu đều, ảnh hằng, F=IR, F=255-IR, dịch độ sáng, v.v.). |
| [`aggregation_comparison.csv`](aggregation_comparison.csv) | So sánh 3 phương thức tổng hợp MS-SSIM đa nguồn: Equal Mean (0.5/0.5), Direct Sum (1.0/1.0), và Variance-Weighted. |

## 2. Kết luận kiểm chứng toán học

1. **Entropy (EN):**
   - Đạt $0.0$ bit/pixel trên ảnh hằng (zero information).
   - Đạt đúng $\log_2(256) = 8.0$ bit/pixel trên phân phối đều rời rạc 256 mức xám.
2. **Standard Deviation (SD):**
   - Chuẩn hoá theo quy ước `ddof=0` (population standard deviation) theo thông lệ CVPR fusion benchmarks.
   - Luôn ép kiểu `float64` trước khi trừ nhằm chống tràn số kiểu `uint8`.
3. **Sum of Correlation Differences (SCD):**
   - Công thức: $SCD = r(F - VIS, IR) + r(F - IR, VIS)$.
   - Đã chứng minh tính bất biến khi cộng hằng số vào ảnh fused: $SCD(F + c, IR, VIS) = SCD(F, IR, VIS)$.
   - Miền giá trị lý thuyết: $[-2.0, 2.0]$. Khi $F = 255 - IR$ (ngược pha), $SCD$ chuyển sang âm ($-0.713$).
4. **MS-SSIM & Source Aggregation:**
   - Hoàn toàn tái lập $MS\text{-}SSIM(A, A) = 1.0$ trên ảnh đồng nhất.
   - Phân biệt rõ `MS_SSIM_mean` (chuẩn $[0, 1]$) và `MS_SSIM_sum` (chuẩn $[0, 2]$ thường dùng trong một số script benchmark của PCDFusion / VIFB). Cả hai đều được báo cáo minh bạch trong `per_image.csv`.
