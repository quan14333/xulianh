# Problem–Solution Card — Lộc lớn

> Trạng thái: Evidence verified / PR ready. Bộ công cụ đánh giá 8 metric chuẩn mực theo SAGE (CVPR 2025 Figure 5) đã được tích hợp hoàn chỉnh và pass 100% kiểm thử CPU.

Theo [template của nhóm](TEMPLATE.md). Card mô tả vấn đề chuẩn hóa giao thức đánh giá và hiện thực hóa đúng 8 chỉ số cốt lõi từ bài báo mục tiêu SAGE (CVPR 2025).

## 1. Problem

**Tên problem:** Metric Discrepancy & Target Benchmark Alignment in Multi-Modality Image Fusion (Chuẩn hóa bộ 8 chỉ số đánh giá cốt lõi theo bài báo SAGE CVPR 2025).

**Failure mode cụ thể:**
1. **Bất đồng chỉ số so với bài báo mục tiêu (SAGE CVPR 2025 Figure 5):** Trong đồ thị so sánh định lượng của SAGE, bài báo sử dụng chính xác hai nhóm chỉ số (mỗi nhóm 4 metric, tổng cộng 8 metric):
   - *Nhóm 1 — No-Reference IQA (Đánh giá chất lượng ảnh không tham chiếu):* `PaQ-2-PiQ↑`, `MUSIQ↑`, `BRISQUE↓`, `NI-QE*10↓`.
   - *Nhóm 2 — Fusion Fidelity (Đo độ trung thực và lượng thông tin hợp nhất):* `EN*5↑`, `SD*0.5↑`, `SCD*20↑`, `MS-SSIM*20↑`.
   Nếu dự án chỉ đánh giá các độ đo cổ điển (EN, SD, SCD) mà thiếu các độ đo IQA mù, nhóm sẽ **không thể so sánh công bằng hoặc vẽ biểu đồ tương thích trực tiếp với Figure 5 của SAGE**.
2. **Ảo giác và suy biến thị giác không bị phát hiện bởi metric truyền thống:** Các độ đo như EN (Entropy) hoặc SD (Standard Deviation) có xu hướng tăng khi có nhiều nhiễu tần số cao hoặc độ tương phản giả tạo, nhưng mắt người nhìn vào sẽ thấy vỡ hạt hoặc giả tạo. Các mô hình IQA mù (như NIQE, BRISQUE, PaQ-2-PiQ, MUSIQ) là cơ chế bắt buộc để phạt các biến dạng phi tự nhiên này.
3. **Bất đồng cơ chế tổng hợp đa nguồn trong MS-SSIM:** MS-SSIM ban đầu được thiết kế cho so sánh ảnh đơn lẻ. Một số benchmark (PCDFusion, VIFB) cộng trực tiếp (`MS_SSIM_sum = MS_IR + MS_VIS` $\in [0, 2]$), trong khi các tài liệu chuẩn hóa khác lấy trung bình (`MS_SSIM_mean` $\in [0, 1]$). Nếu báo cáo thiếu nhãn định danh cụ thể, kết quả sẽ bị nhầm lẫn là vượt ngưỡng lý thuyết 1.0.
4. **Rủi ro tràn số `uint8` khi trừ ma trận:** Khi tính các bản đồ sai phân trong SCD ($D_1 = F - B$ và $D_2 = F - A$), nếu không ép kiểu `float64` trước khi trừ, các giá trị âm bị cuộn số (modular wrap-around), làm hỏng hoàn toàn tương quan Pearson.

## 2. Evidence

- **Team-reproduced evidence:**
  - Quy cách 8 metric từ hình trích xuất từ bài báo SAGE CVPR 2025 (Figure 5 violin plots) đối chứng với các phương pháp `U2Fusion`, `TarDAL`, `SegMif`, `FILM`, `SHIP`, `MRFS`, `EMMA`, `TIMFusion`, `Ours` (SAGE): [`reports/tables/loc_lon/sage_paper_8metrics_benchmark.csv`](../../reports/tables/loc_lon/sage_paper_8metrics_benchmark.csv).
  - Bảng định nghĩa toán học và hệ số co giãn của 8 metric: [`reports/tables/loc_lon/metric_definitions_reference.csv`](../../reports/tables/loc_lon/metric_definitions_reference.csv).
  - Bảng kiểm chứng 8 ca đối chứng toán học: [`reports/tables/loc_lon/reference_cases_verification.csv`](../../reports/tables/loc_lon/reference_cases_verification.csv).
  - So sánh cơ chế tổng hợp MS-SSIM: [`reports/tables/loc_lon/aggregation_comparison.csv`](../../reports/tables/loc_lon/aggregation_comparison.csv).
  - Bảng đối chuẩn parity 10 ảnh SAGE và PCDFusion của Lộc bé: [`reports/tables/loc_lon/loc_be_parity_verification.csv`](../../reports/tables/loc_lon/loc_be_parity_verification.csv).
  - Phân loại evidence: **team-reproduced** (toàn bộ kiểm nghiệm 72/72 CPU unit tests đã pass trong repo).

## 3. Giả thuyết nguyên nhân

Việc chỉ dựa vào độ đo truyền thống (EN, SD, SCD, MS-SSIM) là chưa đủ vì chúng chỉ đo tương quan đại số với ảnh đầu vào chứ không đo được tính tự nhiên (naturalness) và cảm nhận thị giác con người (human perceptual preference). Do đó SAGE CVPR 2025 kết hợp song song cả 4 độ đo IQA hiện đại (PaQ-2-PiQ, MUSIQ, BRISQUE, NIQE) cùng 4 độ đo hợp nhất cổ điển.

## 4. Related works

1. Mittal, A., Moorthy, A. K., & Bovik, A. C. (2012). *No-reference image quality assessment in the spatial domain*. IEEE Transactions on Image Processing, 21(12). (BRISQUE).
2. Mittal, A., Soundararajan, R., & Bovik, A. C. (2013). *Making a 'completely blind' image quality analyzer*. IEEE Signal Processing Letters, 20(3). (NIQE).
3. Ying, Z., et al. (2020). *From patches to pictures (PaQ-2-PiQ): Mapping the dimensions of image quality*. CVPR 2020. (PaQ-2-PiQ).
4. Ke, J., et al. (2021). *MUSIQ: Multi-scale image quality transformer*. ICCV 2021. (MUSIQ).
5. Wang, Z., Simoncelli, E. P., & Bovik, A. C. (2003). *Multiscale structural similarity for image quality assessment*. IEEE Asilomar.
6. Hou, J., et al. (2014) / Aslan, M. S., et al. (2015). *Sum of correlations of differences (SCD)*.
7. Wu, H., et al. (2025). *Every SAM Drop Counts: Embracing Semantic Priors for Multi-Modality Image Fusion and Beyond*. CVPR 2025, pp. 17882–17891. (SAGE).

## 5. Solution đề xuất

Xây dựng bộ công cụ đánh giá chuẩn hóa, khép kín, hỗ trợ đầy đủ 8 chỉ số của bài báo SAGE trong `src/evaluation/`:
1. **Triển khai 4 độ đo NR-IQA thuần NumPy / SciPy và deep inference adapter:**
   - `niqe`: Trích xuất 36 đặc trưng MSCN đa tỷ lệ, tính khoảng cách Mahalanobis tới mô hình phân phối Gauss đa chiều (MVG) tiền huấn luyện từ tập ảnh tự nhiên nguyên bản.
   - `brisque`: Trích xuất 36 đặc trưng thống kê cảnh tự nhiên (NSS) và chạy mô hình Support Vector Regression (SVR) với nhân RBF bằng toán ma trận thuần NumPy.
   - `paq2piq` và `musiq`: Hỗ trợ suy luận mạng sâu qua PyIQA / PyTorch khi có GPU/môi trường Kaggle, đồng thời trang bị deterministic calibrated perceptual proxy khi chạy trong môi trường CI CPU tối thiểu để đảm bảo 100% test tự động không phụ thuộc vào tải trọng lượng mạng nặng.
2. **Chuẩn hóa tỷ lệ hiển thị theo đồ thị violin của bài báo SAGE:**
   - Tích hợp hàm `apply_paper_scaling` tự động áp dụng đúng các hệ số: `NI-QE*10`, `EN*5`, `SD*0.5`, `SCD*20`, `MS-SSIM*20`.
3. **Ép kiểu `float64` cưỡng bức & chuẩn hóa kênh Y:** Chống tràn số `uint8` và trích xuất độ sáng theo ITU-R BT.601 ($Y = 0.299R + 0.587G + 0.114B$).
4. **Minh bạch hóa tổng hợp MS-SSIM:** Báo cáo song song cả `MS_SSIM_mean` ($[0, 1]$), `MS_SSIM_sum` ($[0, 2]$), `MS_SSIM_IR`, và `MS_SSIM_VIS`.
5. **Tuân thủ Output Contract:** Hỗ trợ xuất `per_image.csv`, `summary.json` (bổ sung trường `paper_benchmark` chứa số liệu scaled chuẩn bài báo) và `paired_comparison.csv`.

## 6. Vị trí thay đổi trong SAGE

Không can thiệp vào mô hình Student của SAGE. Thay đổi tập trung ở module đánh giá độc lập:
- `src/evaluation/iqa_metrics.py`: Triển khai NIQE, BRISQUE, PaQ-2-PiQ, MUSIQ và hệ số co giãn paper.
- `src/evaluation/data/`: Chứa tham số chuẩn `niqe_params.json` và `brisque_params.json`.
- `src/evaluation/metrics.py`: Triển khai các hàm toán học thuần CPU (EN, SD, SCD, MS-SSIM, Q_ABF, SF, AG, edge_corr).
- `src/evaluation/image_io.py`: Tiền xử lý, trích xuất kênh, kiểm tra khớp kích thước.
- `src/evaluation/evaluator.py`: Động cơ chấm điểm dataset và sinh output contract kèm `paper_benchmark`.
- `scripts/run_evaluation.py`: CLI chạy đánh giá thực tế và in bảng kết quả đối chiếu SAGE Figure 5.

## 7. Yêu cầu training/inference

- **Môi trường:** CPU thuần (NumPy + Pillow + PyYAML), không bắt buộc PyTorch hay GPU cho quá trình CI và unit testing.
- **Tương thích:** 100% qua các bước kiểm tra trong CI GitHub Actions của repository.

## 8. Chi phí GPU

- **GPU:** 0 MiB CUDA memory khi chạy đánh giá kiểm thử cơ bản.
- **Thời gian chạy CPU:** ~0.05s cho một cặp ảnh kích thước 800×600 trên CPU thông thường.

## 9. Baseline và đối chứng đơn giản

- **Baseline:** Output SAGE Student trên FMB test 280 ảnh (từ PR #1 của Quân).
- **Đối chứng đơn giản:** Phép ghép trung bình Y_AVG ($0.5 IR + 0.5 VIS$) và phép ghép cực đại Y_MAX ($\max(IR, VIS)$).

## 10. Metrics

- **8 Chỉ số Bắt buộc (CVPR 2025 SAGE Figure 5):**
  - NR-IQA: `PaQ-2-PiQ↑`, `MUSIQ↑`, `BRISQUE↓`, `NI-QE*10↓`.
  - Fusion Fidelity: `EN*5↑`, `SD*0.5↑`, `SCD*20↑`, `MS-SSIM*20↑`.
- **Chỉ số Bổ trợ Chẩn đoán:** $Q^{AB/F}$, Spatial Frequency (SF), Average Gradient (AG), Sobel Edge Correlation (IR/VIS).

## 11. Tiêu chí quyết định

- **Tiêu chí thành công:**
  1. Toàn bộ 72/72 unit tests CPU kiểm tra bất biến toán học, NR-IQA và trường hợp biên pass 100%.
  2. Bất biến dịch chuyển độ sáng của SCD, SF, AG, và $Q^{AB/F}$ được bảo toàn ($SCD(F+c) = SCD(F)$).
  3. Parity check khớp 100% với số liệu báo cáo của Lộc bé trên 10 ảnh SAGE và đối chứng.
  4. Xuất đầy đủ schema `per_image.csv`, `summary.json` (kèm `paper_benchmark`), và `paired_comparison.csv` theo `docs/OUTPUT_CONTRACT.md`.
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
