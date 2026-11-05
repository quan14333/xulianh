# Protocol thí nghiệm và đánh giá

## 1. Đăng ký trước khi chạy

Mỗi experiment có issue, `run_id`, commit, config, manifest, checkpoint ID, preprocessing, seed,
môi trường, baseline, thay đổi duy nhất và tiêu chí thành công/bác bỏ. Dùng dev split để phát triển;
không nhìn test để tinh chỉnh. Primary metric/protocol được khóa trước evaluation chính thức.

## 2. Input contract

- IR, VIS và Fused khớp chính xác `image_id`; không ghép theo thứ tự file hệ thống.
- Shape/registration khớp hoặc sai lệch phải làm run thất bại có thông báo.
- Ghi rõ đánh giá kênh Y, grayscale hay RGB và range (`[0,1]`/`[0,255]`).
- Cast sang floating point trước phép trừ/chênh lệch để tránh `uint8` overflow.
- Ghi cách xử lý NaN, ảnh constant, border, window và ảnh không hợp lệ.

## 3. Metrics cơ bản

- **EN:** entropy; xác định bin/range và công thức log.
- **SD:** standard deviation; xác định population/sample convention và channel.
- **SCD:** phải xác minh theo định nghĩa correlation differences bằng reference implementation,
  công thức viết tay và test trường hợp biết trước. “Script chạy được” không chứng minh đúng paper.
- **MS-SSIM:** xác minh công thức đa tỷ lệ, weights/downsampling/range và cách tổng hợp score
  Fused–IR với Fused–VIS. Unit test cơ bản chưa đủ chứng minh tương thích paper.

QAB/F và mIoU chỉ mở rộng khi implementation, evaluator và thời gian cho phép. Với mIoU phải ghi
model segmentation, label mapping, preprocessing và evaluator; không suy ra từ fusion metrics.

## 4. Kiểm chứng implementation

1. Đối chiếu paper/official code và ghi phiên bản chính xác.
2. Tạo test synthetic có expected behavior, test dtype/range/shape và edge cases.
3. So sánh trên sample nhỏ với implementation độc lập khi có.
4. Review công thức bởi owner evaluation và ít nhất một người khác.
5. Chỉ đổi trạng thái từ `needs_verification` khi evidence được commit.

## 5. Báo cáo

Cho mỗi metric báo `mean`, `std`, `median`, số ảnh hợp lệ và số ảnh bị loại kèm lý do. So sánh
phương pháp mới với baseline theo từng cặp ảnh, báo cả cải thiện lẫn suy giảm và paired delta.
Không chọn vài ảnh đẹp để đại diện toàn bộ test.

Mọi bảng phải phân loại nguồn:

- **Paper-reported:** số công bố, có citation và protocol của paper.
- **Team-reproduced:** run thật của nhóm, trỏ `run_id`, commit, config và manifest.
- **Illustrative:** ví dụ minh họa, không dùng làm claim hiệu năng.
