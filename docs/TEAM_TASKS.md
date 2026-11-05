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

| Công việc | Trạng thái | Evidence trong repo / bước tiếp theo |
|---|---|---|
| Notebook smoke inference SAGE trên Kaggle | Đã làm ngoài repo; chưa import | Import notebook đã dọn output, config và log thật |
| Thử 10 ảnh fused | Đã làm ngoài repo; chưa import | Gắn manifest/run ID; không tự tạo ảnh hay số liệu thay thế |
| Limitations, ablation, related work | Chờ bàn giao | Kiểm tra citation và phân biệt paper/team evidence |
| Problem/solution và prototype | Chưa chốt | Hoàn thiện card, chờ quyết định 14/11 |

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
