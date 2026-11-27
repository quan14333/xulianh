# Quyết định scene split Q02B — scene_v1

**Ngày:** 2026-11-16. **Nguồn quyết định:** chỉ ảnh VIS/IR của 1.220 ảnh FMB official **train**, manifest Q02A, contact sheet và feature matching. Không xem model metric, predictions hay ảnh/nhãn official Test. Đây là quy tắc cố định *trước* Q02B GPU. Manifest Q02A không bị sửa.

## Tiêu chí

Một nhóm phải cùng split khi có mốc nền đặc trưng chung (tòa nhà, đường/cột đèn, cây, biển/cầu) và cùng hoặc liền kề góc camera; ảnh liên tiếp có nhiều ORB homography inlier củng cố việc chúng là một đoạn quay. ORB ít inlier **không** phủ nhận cùng cảnh khi thay ánh sáng/góc, nên xem ảnh gốc. Không gộp chỉ vì ID gần, dHash thấp, loại đường giống nhau, hoặc có vài inlier rời rạc. Cắt một chuỗi ở chỗ mốc cảnh thay rõ và matching sụt mạnh. Chọn **chuyển toàn bộ nhóm Train liên quan sang Dev** để giữ nguyên các nhóm Dev Q02A và không dùng nhãn Dev cho Train.

## Nhóm REVISE: chuyển 53 ảnh Train sang Dev

| Nhóm | Dev Q02A liên quan | Train chuyển sang Dev (đều tồn tại trong official train) | Bằng chứng ảnh/feature và ranh giới |
|---|---|---|---|
| Cổng/tường mái vòm | `00052–00055` | `00040–00042` (3) | Mái vòm, các tòa nhà phía sau và tường ngang giống nhau dưới sương/đêm; `00040–00042` có 793–1188 ORB inlier nội nhóm. Ghép `00042–00052` chỉ 0 inlier do ánh sáng thay đổi, nên quyết định dựa trên mốc kiến trúc. `00039–00040` 0 inlier và cảnh đổi từ đường núi sang tường mái vòm. |
| Đại lộ cao ốc | `00172–00175` và các Dev trong `00172–00206` | `00157–00165` (9) | Cùng dãy cao ốc, hàng đèn/cây và đường; từng ảnh Train có tối đa 5–13 inlier với Dev `00172–00175`, liên tiếp trong nhóm có 17–49 inlier. `00156–00157` chỉ 4 inlier và cảnh đổi từ vòng xoay sang đại lộ. |
| Đại lộ cây/tòa tháp ban ngày | `00195–00206` | `00214–00238` (25) | Cùng dải cây bách, tháp đôi và các làn đường; từng ảnh Train có 5–10 inlier với Dev `00195–00206`, các frame Train liền nhau phần lớn có hàng chục inlier. `00238–00239` chỉ 4 inlier và đổi sang đoạn có biển chỉ đường/khu xây dựng. |
| Đại lộ đèn đêm | `00601–00630`, đặc biệt `00629–00630` | `00701–00709` và `00722–00728` (16) | `00630–00701` gần như cùng camera/đèn/cây/mặt đường: 122/95 inlier trong Q02B-Prep; phép ORB mở rộng cho `00701` tối đa 198/154 với `00630`. `00702–00709` 13–60 inlier với Dev, `00722–00728` 6–9 inlier cùng tuyến đèn/cây. `00700–00701` chỉ 4 inlier và đổi từ công viên sang đại lộ. `00728–00729` 0 inlier, còn `00729–00730` 387 inlier trong cảnh đường có cao ốc khác; vì vậy `00729` ở Train. |

Tổng **3 + 9 + 25 + 16 = 53** ID, không có ID lặp. Kết quả dự kiến: **Train 992, Dev 228, tổng 1.220**. ID, path VIS/IR/Label và checksum nội dung giữ từ Q02A; chỉ `split` và tư cách Train/Dev thay đổi. Danh sách/điểm matching đầy đủ ở [q02b_release_cross_features.csv](q02b_release_cross_features.csv), [q02b_release_adjacent_features.csv](q02b_release_adjacent_features.csv) và contact sheets [a](q02b_release_scene_a.png), [b](q02b_release_scene_b.png), [c](q02b_release_scene_c.png), [d](q02b_release_scene_d.png), [e1](q02b_release_scene_e1.png), [e3](q02b_release_scene_e3.png).

## Nhóm KEEP sau review

| Cặp/ranh giới | Bằng chứng | Quyết định |
|---|---|---|
| Dev `00068` – Train `00031` | Ảnh gốc: một bên đường có tháp điện thép, bên kia giao lộ với cột viễn thông và cao ốc; không phải cùng mốc tĩnh. | **KEEP**; dHash VIS 8 là false positive cho scene. |
| Dev `00120` – Train `00136` | Đường mở có nhà phía xa so với bãi xe/cây và chữ đèn; 0 inlier ranh giới. | **KEEP**. |
| Dev `00591` – Train `00586` | `00586` là đường cây/hàng đèn; `00591` nhìn tòa chung cư và bờ dốc; không thấy mốc tĩnh chung, 0 inlier. ID gần không đủ gộp. | **KEEP**, nhưng không đồng nghĩa hai ảnh không thể ở cùng hành trình rộng hơn. |
| Dev `01383` – Train `01364` | Cầu cong, lan can, xe buýt và cụm nhà so với đường thẳng/cây/tòa nhà; cảnh khác. | **KEEP**. |
| Dev `01416` – Train `01466` | Hai cấu trúc đường/cây khác; IR dHash thấp nhưng VIS khác. | **KEEP**. |
| Dev `00600` – Train `00425`; Dev `00598` – Train `00970` | IR ít kết cấu/dHash thấp, VIS khác hẳn; không có inlier ổn định. | **KEEP**, false positive IR. |
| Dev `00060` – Train `00442`; Dev `00072` – Train `00447`; Dev `00206` – Train `01123/01131` | Contact sheet [residual1](q02b_release_pairs_residual1.png): xe/góc/kiến trúc/cây và đường khác; 5–8 inlier rời rạc chưa xác nhận mốc tĩnh. | **KEEP**, false positive feature/dHash screening. |

## Giới hạn của quyết định

Đây là **PASS cho các overlap và ranh giới đã review**, không phải chứng minh tuyệt đối rằng mọi cặp trong 1.220 ảnh độc lập ngữ nghĩa theo địa điểm. Q02A đã kiểm ID/MD5/dHash toàn bộ cặp; Q02B-Prep ORB 147 ứng viên; Q02B-Release mở rộng contact sheet, kiểm 53 ảnh đề xuất với các Dev cùng cảnh và xem các ứng viên còn lại có matching cao. dHash/ORB có thể bỏ sót cùng địa điểm khi khác mùa, ánh sáng hoặc góc; vì vậy kết quả Dev phải được mô tả là **scene-group-corrected development split under audited candidates**, không gọi là chứng minh scene-independent toàn bộ FMB. Nếu sau này phát hiện một mốc tĩnh cross-split chưa review, dừng chọn phương pháp trên Dev và lập phiên bản manifest mới trước khi dùng thêm metric. Pretrained Student SAGE có thể từng học FMB Train theo bằng chứng gián tiếp; quyết định này không giải quyết provenance pretraining đó.
