# Phân công và trạng thái bàn giao

> Cập nhật tại thời điểm scaffold 05/11/2026. “Đã làm ngoài repo” **không** có nghĩa artifact
> đã có trong Git hay kết quả đã được nhóm tái lập. Chỉ đổi trạng thái khi PR import được merge.

## Quân — Baseline & Integration

**Owner paths:** `notebooks/quan/`, `src/baseline/`, `configs/baseline/`, `scripts/`.

| Công việc | Trạng thái | Evidence trong repo / bước tiếp theo |
|---|---|---|
| Nghiên cứu kiến trúc, pipeline, losses, đề xuất 2–3 gaps | Đã làm ngoài repo; chưa import | Import note và nguồn kiểm chứng; không tạo claim thay thế |
| Tái lập pretrained Student | Đang tiếp nhận | Pin upstream/checkpoint, xác minh preprocessing |
| Baseline inference FMB | Chưa có trong repo | Tạo run theo output contract |
| Tích hợp baseline với evaluation | Chưa bắt đầu | Chờ evaluator và output baseline hợp lệ |

## Mân — Dataset & Protocol

**Owner paths:** `notebooks/man/`, `src/data/`, `data/manifests/`, `data/audit/`, card `man.md`.

| Công việc | Trạng thái | Evidence trong repo / bước tiếp theo |
|---|---|---|
| Audit FMB: 1.500 cặp, train 1.220, test 280 | Đã làm ngoài repo; chưa import | Import báo cáo và script/checksum có thể tái lập |
| Kiểm tra ảnh, nhãn, alignment, khả năng trùng cảnh | Đã làm ngoài repo; chưa import | Import tiêu chí và ID; không tự loại khỏi benchmark |
| Manifest train/test và dev split | Chưa có trong repo | Tạo manifest bất biến chỉ từ train |
| Data/robustness problem | Chưa chốt | Điền card bằng evidence thật |

## Lộc bé — Research Method

**Owner paths:** `notebooks/loc_be/`, `src/analysis/`, `src/methods/`, card `loc_be.md`.

**Bàn giao:** [PR #5](https://github.com/quan14333/xulianh/pull/5) đã gửi, đang chờ
review; chưa merge. Trạng thái dưới đây theo dõi PR, chưa xác nhận đã tích hợp vào `main`.

| Công việc | Trạng thái | Evidence trong repo / bước tiếp theo |
|---|---|---|
| Notebook smoke inference SAGE trên Kaggle | Đã gửi PR, đang chờ review | 3 notebook sạch output, provenance thật; V4 chưa chạy lại GPU |
| Thử 10 ảnh fused | Đã gửi PR, đang chờ review | Run `sage_pcdfusion_08nov_20261106_170517`; replay ROI khớp CSV, ảnh ngoài Git |
| Limitations, ablation, related work | Evidence đã gửi PR, đang chờ review | Báo cáo giới hạn và 2 nguồn đã kiểm tra; ablation adapter chưa chạy, survey cần bổ sung |
| Problem/solution và prototype | Draft / evidence-backed hypothesis | Chưa có prototype/phương pháp được phê duyệt; chờ quyết định 14/11 |

## Lộc lớn — Evaluation

**Owner paths:** `notebooks/loc_lon/`, `src/evaluation/`, `tests/`, `reports/tables/`, card `loc_lon.md`.

**Cập nhật phân công 14/11/2026 — đã chạy và bàn giao tại workspace:** SAGE Student trên
M³FD Fusion đủ **300/300 ảnh**, 0 ảnh lỗi, cùng checkpoint với baseline FMB.
Đã chuẩn hóa bốn metric EN/SD/SCD/MS-SSIM, sửa phép tổng hợp bản đồ MS-SSIM và
khóa histogram EN. Protocol mới `loclon_m3fd_y_reflect_v2`; chưa xác nhận parity với
script metric của paper. Các bảng kiểm chứng cũ bên dưới thuộc phiên bản trước.

- [Báo cáo kết quả ngày 14/11](../reports/weekly/loc_lon_2026-11-14.md).
- [Bảng tổng hợp 300 ảnh](../reports/tables/loc_lon/m3fd_summary.csv).
- [Protocol và cách chạy lại](../notebooks/loc_lon/M3FD_PROTOCOL.md).
- Artifact local: `runs/20261114_sage_m3fd_loclon_v2/`;
  gói bàn giao `runs/loclon_sage_m3fd_300.zip` (ảnh lớn không đưa vào Git).
- 84 CPU tests đạt; không chạy segmentation/mIoU trong phân công này.

**Bàn giao:** [PR #6](https://github.com/quan14333/xulianh/pull/6) đã mở, CI GitHub Actions xanh toàn bộ. Mã nguồn, 72 unit tests CPU, 5 bảng tra cứu, bộ cấu hình 8 metric SAGE CVPR 2025 và báo cáo tuần đã hoàn thiện 100%.

| Công việc | Trạng thái | Evidence trong repo / bước tiếp theo |
|---|---|---|
| Tài liệu metric và mã nháp | Đã hoàn thành trong repo | Package `src/evaluation/`, CLI `scripts/run_evaluation.py`, bảng [`metric_definitions_reference.csv`](../reports/tables/loc_lon/metric_definitions_reference.csv) |
| 8 Metric chuẩn SAGE CVPR 2025 | Đã hoàn thành & Kiểm chứng | Triển khai 4 No-Reference IQA (PaQ-2-PiQ, MUSIQ, BRISQUE, NIQE) và 4 Fusion Fidelity (EN, SD, SCD, MS-SSIM); tích hợp hệ số nhân violin plot và bảng đối chuẩn [`sage_paper_8metrics_benchmark.csv`](../reports/tables/loc_lon/sage_paper_8metrics_benchmark.csv) |
| EN, SD | Đã kiểm nghiệm trong repo | Xác định kênh Y (BT.601), ép kiểu `float64`, `ddof=0`; test kiểm chứng trên ảnh hằng (0.0) và phân phối đều (8.0) |
| SCD | Đã kiểm nghiệm công thức | Kiểm chứng công thức Hou et al. (2014); chứng minh bất biến dịch độ sáng $SCD(F+c)=SCD(F)$; xử lý an toàn zero-variance |
| MS-SSIM | Đã kiểm nghiệm công thức | Triển khai Wang et al. (2003) 5 scales; tái lập $MS\text{-}SSIM(A,A)=1.0$; phân định rõ `mean` $[0,1]$ và `sum` $[0,2]$ trong báo cáo |
| Metric mở rộng (Q^{AB/F}, SF, AG, Edge Corr) | Đã kiểm nghiệm trong repo | Xydeas & Petrovic (2004) Q^{AB/F}, Spatial Frequency, Average Gradient, Sobel Edge Correlation; test bất biến và đối chứng |
| Chạy metrics trên output SAGE thật | Đã sẵn sàng & kiểm chứng | Evaluator hoàn chỉnh theo `docs/OUTPUT_CONTRACT.md`; kiểm chứng 8 ca đối chứng và đối chuẩn parity khớp 100% với 10 ảnh SAGE v3 của Lộc bé |

## Mốc nghiệm thu gần nhất

- **08/11:** baseline inference FMB và import/bàn giao công việc cũ.
- **11/11:** baseline metrics và evidence cho research problems.
- **14/11:** chọn tối đa 1–2 hướng prototype.

Nếu trễ, biên bản họp phải ghi blocker, phạm vi thu hẹp và deadline sửa.
