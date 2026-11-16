# Workspace Đánh giá & Kiểm nghiệm Metric — Lộc lớn (Evaluation)

**Phân công 14/11/2026:** chạy SAGE Student trên **M³FD Fusion, 300 cặp**, bàn giao
ảnh fused và **EN/SD/SCD/MS-SSIM**. Xem [protocol mới](M3FD_PROTOCOL.md) và
[notebook chạy baseline](SAGE_M3FD_Baseline.ipynb). CLI:
`python scripts/run_m3fd_baseline.py --device cpu`.

Protocol mới `loclon_m3fd_y_reflect_v2` sửa phép tổng hợp bản đồ SSIM/MS-SSIM và
khóa histogram EN. Các bảng kiểm chứng trước 14/11 bên dưới thuộc phiên bản cũ;
không dùng chúng làm bằng chứng parity cho v2 hoặc cho script của paper.
Phạm vi bàn giao mới chỉ gồm bốn fusion metric, không chạy NR-IQA.

Workspace này cung cấp notebook và tài liệu kiểm nghiệm cho **8 thước đo chất lượng chuẩn mực của bài báo SAGE (CVPR 2025 Figure 5)**:
- **4 No-Reference IQA:** `PaQ-2-PiQ↑`, `MUSIQ↑`, `BRISQUE↓`, `NI-QE*10↓`
- **4 Fusion Fidelity:** `EN*5↑`, `SD*0.5↑`, `SCD*20↑`, `MS-SSIM*20↑`

## 1. Danh mục Notebook

| Notebook | Mục đích & Phạm vi |
|---|---|
| [`SAGE_Evaluation_Metrics_Verification.ipynb`](SAGE_Evaluation_Metrics_Verification.ipynb) | Phân tích toán học, kiểm nghiệm trên các ca đối chứng nhân tạo, đánh giá độ nhạy kênh (Y vs Gray vs RGB), kiểm chứng 8 chỉ số theo đồ thị violin của SAGE CVPR 2025 Figure 5 và xuất báo cáo chuẩn `OUTPUT_CONTRACT.md`. |

## 2. Hướng dẫn chạy Notebook

### Chạy cục bộ với môi trường của repo:
```bash
# Kích hoạt môi trường
source .venv/bin/activate

# Cài đặt jupyter nếu cần
pip install ipykernel jupyterlab

# Chạy notebook
jupyter lab notebooks/loc_lon/SAGE_Evaluation_Metrics_Verification.ipynb
```

Notebook được thiết kế tuân thủ nghiêm ngặt quy tắc nhóm:
- Không chứa secret, API key hay dữ liệu nhạy cảm.
- Code cells sạch output hoặc đã sanitize, có thể thực thi 100% bằng CPU thuần (dùng numpy/Pillow), không bắt buộc GPU hay PyTorch nặng.
- Kết quả kiểm chứng được liên kết với các bảng tại `reports/tables/loc_lon/`.
