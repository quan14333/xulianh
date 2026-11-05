# Chính sách dữ liệu FMB

## Split chính thức và mục đích

- Tổng theo báo cáo audit chưa import: **1.500 cặp**.
- Train chính thức: **1.220 cặp**.
- Test chính thức: **280 cặp**.
- Development split phải được tạo chỉ từ train, có seed/quy tắc và manifest bất biến.
- Test không được dùng để chọn hyperparameter, điều chỉnh solution hoặc chọn model tốt nhất.

Có thể chạy pretrained baseline trên test để tái lập benchmark, nhưng phải tách khỏi vòng phát
triển phương pháp. Chạy pretrained Student không phải full training SAGE. Không claim 61.2% mIoU
khi chưa có đúng downstream segmentation evaluator và protocol.

## Lưu trữ và manifest

FMB cục bộ đặt tại `data/raw/FMB/` và bị Git ignore. Manifest commit được nên chứa tối thiểu:
`image_id`, `split`, đường dẫn IR/VIS tương đối, shape, và checksum khi khả thi. Dữ liệu dẫn xuất
đặt ở `data/processed/`; mỗi phiên bản cần liên kết về source manifest và preprocessing config.

## Preprocessing cần xác minh

Audit gốc ghi kích thước **800×600**, còn notebook smoke ban đầu có thể dùng **640×512**. Đây là
một điểm chưa giải quyết, không được xem hai thiết lập tương đương. Trước baseline chính thức phải
ghi và kiểm tra:

- thứ tự width/height và cách resize/crop/pad;
- VIS RGB hay chuyển YCbCr, kênh nào được fusion và cách phục hồi màu;
- IR grayscale, bit depth và channel expansion (nếu có);
- normalization/range/dtype, interpolation và alignment;
- output có trở về đúng kích thước benchmark hay không.

## Leakage và cảnh có khả năng trùng

Audit ngoài repo đã ghi nhận một số cặp có khả năng trùng cảnh giữa train/test. Không tự động
loại chúng khỏi benchmark chuẩn. Mân cần import danh sách ID và tiêu chí kiểm tra. Báo cáo chính
giữ test chuẩn; có thể thêm báo cáo sensitivity loại nghi vấn nếu công bố đầy đủ ID, tiêu chí và
không thay nó thành benchmark chính.

## Bảo mật và chia sẻ

Không commit dataset/credential. Chỉ dùng nguồn có quyền truy cập hợp lệ và ghi hướng dẫn lấy
dữ liệu, phiên bản, checksum. Ví dụ ảnh chỉ được commit khi quyền sử dụng cho phép và phải ghi nguồn.
