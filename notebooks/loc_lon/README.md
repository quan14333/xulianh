# Workspace Đánh giá & Kiểm nghiệm Metric — Lộc lớn (Evaluation)

Workspace này cung cấp notebook và tài liệu kiểm nghiệm cho các thước đo chất lượng ảnh hợp nhất
hồng ngoại – khả kiến (IVIF): **EN**, **SD**, **SCD**, và **MS-SSIM**.

## 1. Danh mục Notebook

| Notebook | Mục đích & Phạm vi |
|---|---|
| [`SAGE_Evaluation_Metrics_Verification.ipynb`](SAGE_Evaluation_Metrics_Verification.ipynb) | Phân tích toán học, kiểm nghiệm trên các ca đối chứng nhân tạo, đánh giá độ nhạy kênh (Y vs Gray vs RGB) và so sánh các phương thức tổng hợp MS-SSIM đa nguồn. Xuất báo cáo chuẩn `OUTPUT_CONTRACT.md`. |

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
