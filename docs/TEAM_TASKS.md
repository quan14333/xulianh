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

| Công việc | Trạng thái | Evidence trong repo / bước tiếp theo |
|---|---|---|
| Tài liệu metric và mã nháp | Đã làm ngoài repo; chưa import | Import kèm nguồn/phiên bản và test |
| EN, SD | Chưa kiểm nghiệm trong repo | Xác định range/channel/formula |
| SCD | Cần kiểm nghiệm công thức | Reference cases + review định nghĩa correlation differences |
| MS-SSIM | Cần kiểm nghiệm công thức | Kiểm tra multiscale weights và source aggregation |
| Chạy metrics trên output SAGE thật | Chưa thực hiện trong repo | Chờ baseline run có manifest |

## Mốc nghiệm thu gần nhất

- **08/11:** baseline inference FMB và import/bàn giao công việc cũ.
- **11/11:** baseline metrics và evidence cho research problems.
- **14/11:** chọn tối đa 1–2 hướng prototype.

Nếu trễ, biên bản họp phải ghi blocker, phạm vi thu hẹp và deadline sửa.
