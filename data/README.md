# Dữ liệu

Dataset chính là FMB nhưng **không được commit vào Git**.

Đặt dữ liệu cục bộ dưới `data/raw/FMB/`; dữ liệu đã xử lý dưới `data/processed/`.
Chỉ commit manifest, báo cáo audit nhỏ và ví dụ được phép chia sẻ. Mỗi manifest phải có
image ID, split, đường dẫn tương đối, kích thước và checksum khi khả thi. Xem
[`docs/DATA_POLICY.md`](../docs/DATA_POLICY.md) trước khi tạo split.
