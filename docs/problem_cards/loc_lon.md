# Problem–Solution Card — Lộc lớn

> Trạng thái: Evidence verified / PR ready. Bảng kiểm chứng và mã nguồn đã được tích hợp trong repo.

Theo [template của nhóm](TEMPLATE.md). Card mô tả vấn đề độ nhạy giao thức đánh giá và phương án chuẩn hóa evaluator cho nhóm.

## 1. Problem

**Tên problem:** Metric Discrepancy & Protocol Sensitivity in Multi-Modality Image Fusion Evaluation (Độ nhạy giao thức tiền xử lý và bất đồng cơ chế tổng hợp chỉ số trong đánh giá hợp nhất IVIF).

**Failure mode cụ thể:**
1. **Thiếu nhất quán về không gian màu / kênh:** Các bài báo và script mã nguồn mở thường không ghi rõ đánh giá trên kênh Y (YCbCr BT.601), Grayscale chuyển đổi, hay trung bình 3 kênh RGB, gây sai lệch số liệu tái lập lên tới 5–15%.
2. **Bất đồng cơ chế tổng hợp đa nguồn trong MS-SSIM:** MS-SSIM ban đầu được thiết kế cho so sánh ảnh đơn lẻ (1 ảnh gốc, 1 ảnh méo). Trong IVIF có 2 ảnh nguồn (IR và VIS). Một số benchmark (PCDFusion, VIFB) cộng trực tiếp (`MS_SSIM_sum = MS_IR + MS_VIS` $\in [0, 2]$), trong khi các tài liệu chuẩn hóa khác lấy trung bình (`MS_SSIM_mean` $\in [0, 1]$). Nếu báo cáo thiếu nhãn định danh cụ thể, kết quả sẽ bị nhầm lẫn là vượt ngưỡng lý thuyết 1.0.
3. **Rủi ro tràn số `uint8` khi trừ ma trận:** Khi tính các bản đồ sai phân $D_1 = F - B$ và $D_2 = F - A$ trong SCD, nếu không ép kiểu `float64` trước khi thực hiện phép trừ, các giá trị âm bị cuộn số (modular wrap-around) thành các giá trị dương lớn (ví dụ $10 - 200 = 66$ trong uint8), làm hỏng hoàn toàn tương quan Pearson.
4. **Bất đồng giữa thống kê toàn cục (EN, SD) và mục tiêu nhiệt:** EN và SD tính trên toàn ảnh nên có xu hướng thiên vị nhiễu nền tần số cao của ảnh VIS, che lấp sự suy giảm tương phản của mục tiêu hồng ngoại (người/xe) trong vùng tối.

## 2. Evidence

- **Team-reproduced evidence:**
  - Bảng kiểm chứng 8 ca đối chứng toán học: [`reports/tables/loc_lon/reference_cases_verification.csv`](../../reports/tables/loc_lon/reference_cases_verification.csv).
  - So sánh cơ chế tổng hợp MS-SSIM (Mean vs Sum vs Variance-weighted): [`reports/tables/loc_lon/aggregation_comparison.csv`](../../reports/tables/loc_lon/aggregation_comparison.csv).
  - Bảng định nghĩa và nguồn tra cứu metric: [`reports/tables/loc_lon/metric_definitions_reference.csv`](../../reports/tables/loc_lon/metric_definitions_reference.csv).
  - Đối chiếu số liệu SAGE v3 trên 10 ảnh của Lộc bé: SAGE đạt `MS_SSIM_sum` = 1.5544, tương ứng `MS_SSIM_mean` = 0.7772; SCD đạt 1.6790.
  - Phân loại evidence: **team-reproduced** (toàn bộ kiểm nghiệm CPU unit tests đã pass trong repo).

## 3. Giả thuyết nguyên nhân

Chỉ số thống kê toàn cục (EN, SD) xem mọi pixel có trọng số như nhau, không phản ánh cấu trúc ngữ nghĩa phân cấp của SAM/SAGE. Ngoài ra, việc thiếu hợp đồng kiểm soát dữ liệu đầu vào (input contract: ép kiểu `float64`, bắt buộc khớp shape/ID) là nguồn gốc gây ra sự bất đồng giữa các báo cáo thực nghiệm độc lập.

## 4. Related works

1. Wang, Z., Simoncelli, E. P., & Bovik, A. C. (2003). *Multiscale structural similarity for image quality assessment*. In IEEE Asilomar Conference on Signals, Systems and Computers, pp. 1398–1402. DOI: 10.1109/ACSSC.2003.1292216.
2. Hou, J., et al. (2014) / Aslan, M. S., et al. (2015). *A novel method for image fusion performance evaluation based on sum of correlations of differences*. Optik / Signal Processing.
3. Zhang, X. (2021). *Benchmarking spatial and transform domain image fusion algorithms*. Information Fusion, 76, pp. 268–282. (Bộ benchmark VIFB).
4. Wu, H., et al. (2025). *Every SAM Drop Counts: Embracing Semantic Priors for Multi-Modality Image Fusion and Beyond*. CVPR 2025, pp. 17882–17891.

## 5. Solution đề xuất

Xây dựng bộ công cụ đánh giá chuẩn hóa, khép kín và có kiểm thử CPU trong `src/evaluation/`:
1. **Ép kiểu `float64` cưỡng bức:** Mọi ma trận được cast sang `float64` trước khi tính hiệu và phương sai.
2. **Chuẩn hóa kênh Y:** Hỗ trợ trích xuất kênh độ sáng $Y$ theo ITU-R BT.601 ($Y = 0.299R + 0.587G + 0.114B$) cho ảnh hợp nhất và Visible RGB, tương thích chuẩn IVIF.
3. **Minh bạch hóa tổng hợp MS-SSIM:** Báo cáo song song cả 4 cột: `MS_SSIM_mean` ($[0, 1]$), `MS_SSIM_sum` ($[0, 2]$), `MS_SSIM_IR`, và `MS_SSIM_VIS`.
4. **Tuân thủ Output Contract:** Hỗ trợ xuất `per_image.csv`, `summary.json` (mean, std, median, valid_count) và `paired_comparison.csv` phục vụ so sánh baseline với cải tiến.

## 6. Vị trí thay đổi trong SAGE

Không can thiệp vào mô hình Student của SAGE. Thay đổi tập trung ở module đánh giá độc lập:
- `src/evaluation/metrics.py`: Triển khai các hàm toán học thuần CPU.
- `src/evaluation/image_io.py`: Tiền xử lý, trích xuất kênh, kiểm tra khớp kích thước.
- `src/evaluation/evaluator.py`: Động cơ chấm điểm dataset và sinh output contract.
- `scripts/run_evaluation.py`: CLI chạy đánh giá thực tế.

## 7. Yêu cầu training/inference

- **Môi trường:** CPU thuần (NumPy + Pillow), không cần PyTorch, CUDA hay GPU.
- **Tương thích:** 100% qua các bước kiểm tra trong CI GitHub Actions của repository.

## 8. Chi phí GPU

- **GPU:** 0 MiB CUDA memory.
- **Thời gian chạy CPU:** ~0.04s cho một cặp ảnh kích thước 800×600 trên CPU thông thường.

## 9. Baseline và đối chứng đơn giản

- **Baseline:** Output SAGE Student trên FMB test 280 ảnh (từ PR #1 của Quân).
- **Đối chứng đơn giản:** Phép ghép trung bình Y_AVG ($0.5 IR + 0.5 VIS$) và phép ghép cực đại Y_MAX ($\max(IR, VIS)$).

## 10. Metrics

- **Primary:** SCD (Sum of Correlation Differences) và MS-SSIM_mean.
- **Secondary:** EN (Entropy), SD (Standard Deviation), MS-SSIM_IR, MS_SSIM_VIS, MS-SSIM_sum.

## 11. Tiêu chí quyết định

- **Tiêu chí thành công:**
  1. Toàn bộ unit tests CPU kiểm tra bất biến toán học và trường hợp biên pass 100%.
  2. Bất biến dịch chuyển độ sáng của SCD được bảo toàn ($SCD(F+c) = SCD(F)$).
  3. Xuất đầy đủ schema `per_image.csv`, `summary.json`, và `paired_comparison.csv` theo `docs/OUTPUT_CONTRACT.md`.
- **Tiêu chí bác bỏ:**
  - Nếu công thức pure numpy sai khác vượt quá $10^{-4}$ so với reference implementation hoặc phát sinh lỗi chia cho 0 trên ảnh phẳng/suy biến.

## 12. Novelty risk và fallback

- **Rủi ro:** Đóng góp nằm ở tính chuẩn tắc của giao thức thực nghiệm (rigorous protocol & verification), không phải đề xuất kiến trúc fusion mới.
- **Fallback:** Hoàn thiện evaluator đóng vai trò hạ tầng kiểm định chung cho toàn nhóm, phục vụ nghiệm thu baseline của Quân (FMB 280 ảnh), phân tích ROI của Lộc bé, và kiểm định lệch tâm của Mân.

## Review

- **Owner:** Lộc lớn (`kouun-rokku`).
- **Reviewers:** Quân (`quan14333`), Lộc bé (`ThangChimBe`), Mân (`dthanhman`).
- **Ngày hoàn thành bàn giao:** 07/11/2026 (trước mốc nghiệm thu 08/11/2026).
- **Liên kết:** Module `src/evaluation/`, test suite `tests/`, và bảng tra cứu `reports/tables/loc_lon/`.
