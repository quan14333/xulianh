# SAGE / FMB scene_v1 — Thẩm định bằng chứng Failure Analysis (19/11/2026)

**Nguồn:** ZIP Kaggle `20261119T043701Z.zip` (SHA256: `005128d8b6472f4d9bbbec0d6095942cec189979422ffb8285d8ea677f906001`). Đây là báo cáo phân tích sau khi đọc nguyên bản `failure_analysis.md`, hai CSV, `analysis_provenance.json`, xem đủ sáu `contact_sheet_*.png` và đối chiếu mã notebook tương ứng. **Không** chạy mô hình, không đánh giá Dev/Test, không tính lại mIoU.

## 1. Phạm vi và mức hoàn thành

- `scene_v1`: **992 Train / 228 Dev**, 992/992 Train được phân tích; 30 ảnh được chọn minh họa; 6 contact sheets, mỗi ảnh đặt VIS / IR / SAGE / GT cạnh nhau.
- Có đủ **10 file** trong ZIP, tổng giải nén khoảng 7,90 MB. Provenance ghi `trained_new_model=false`, `evaluated_dev=false`, `evaluated_test=false`.
- Notebook đã xác minh 21 pin SHA256 của Q02B và MD5 của **10 mẫu Train đầu**, không phải toàn bộ 992 cặp ảnh tại lần này. Không coi nguồn SAGE/metric là parity với B3 khi chưa kiểm tra contract.
- Đây là báo cáo **mô tả dấu hiệu**, không phải bằng chứng nhân quả hay chứng minh cải thiện phân đoạn.

## 2. Kết quả định lượng trên Train

| Chỉ số (định nghĩa notebook) | Số ảnh có giá trị | Trung vị | Số ảnh có tỷ số dưới 1 | Ý nghĩa hạn chế |
|---|---:|---:|---:|---|
| `edge_retention` (Sobel SAGE / Sobel VIS trên 20% cạnh VIS mạnh nhất) | 992 | 0.905 | 944 | Tỷ lệ biên theo độ lớn gradient, không phải Boundary IoU |
| `person_ir_retention` (tương phản mục tiêu/vòng 15x15 trong SAGE chia IR) | 652 | 0.854 | 435 | Chỉ tính khi có mask người đủ lớn và tương phản IR > 8 |
| `vehicle_ir_retention` (xe ô tô/xe tải/buýt/xe máy) | 681 | 0.763 | 491 | Tương phản ROI xe phụ thuộc cách gộp nhiều xe/mask và vòng nền |

*`edge_retention` dưới 1 không có nghĩa chính xác bao nhiêu % cạnh biến mất. `thermal contrast retention` dưới 1 không đồng nghĩa IoU giảm, vì thang sáng của ảnh SAGE và IR khác nhau.*

Trung vị độ sáng VIS: **115.37 / 255**; có `26` ảnh có tỷ lệ cạnh dưới 0,8, `93` ảnh có tỷ lệ tương phản người dưới 0,5, `195` ảnh có tỷ lệ tương phản xe dưới 0,5. Những số này chỉ để ưu tiên chọn ví dụ quan sát.

## 3. 30 ví dụ được chọn

| Nhóm chọn | Số ảnh | `image_id` |
|---|---:|---|
| Ứng viên suy giảm tương phản người | 8 | 00368, 00377, 01032, 01115, 01180, 01181, 01471, 01473 |
| Ứng viên suy giảm tương phản xe | 6 | 00029, 00286, 00433, 01164, 01350, 01468 |
| Ứng viên suy giảm cạnh VIS | 8 | 00141, 00142, 00143, 00239, 00240, 00248, 00249, 00302 |
| Đối chứng ảnh tối | 4 | 00036, 00037, 00389, 00395 |
| Đối chứng ngẫu nhiên | 4 | 00908, 01005, 01341, 01355 |

**Chú ý hiệu lực mẫu:** Các ảnh 00141–00143, 00239–00240, 01180–01181 và nhiều ảnh cảnh phố kế nhau có thể tương quan theo video/camera. 30 ảnh không tương đương 30 quan sát cảnh độc lập. Nên gom theo scene/block khi tổng hợp và chọn đối chứng tương ứng.

## 4. Quan sát trực tiếp trên 6 contact sheets

- **00377 (contact_sheet_01):** Người ở IR sáng rõ; trên SAGE người vẫn nhìn thấy và sáng khác nền, dù `person_ir_retention≈0.075`. Do đó chỉ số ratio thấp **không được gọi là “SAGE làm mất người”**. Cần kiểm thêm biên người và prediction của B3.
- **00368 (contact_sheet_01):** Người ở xa/nhỏ; tỷ lệ tương phản person≈0.088. Cần crop vùng người kích thước gốc để xác nhận trực quan; tỷ số tính trên toàn mask có thể nhạy với vùng nền hoặc mask nhỏ.
- **00141, 00142, 00143 (contact_sheet_03/04):** Cảnh tối, IR và SAGE nâng mạnh độ sáng nhiều vùng cây/công trình. Cần đánh giá có gây lấn át ranh giới/texture hay chỉ là thay đổi độ sáng; edge ratio 0.752–0.766 vẫn chưa chứng minh mất ngữ nghĩa.
- **00239, 00240, 00248 (contact_sheet_04):** Có biển báo, xe, đường; tỷ lệ cạnh thấp khoảng 0.68–0.70; phù hợp làm ứng viên kiểm tra texture/biên nhưng không khẳng định đèn/biển báo nhận diện kém.
- **01115, 01180, 01181 (contact_sheet_01/02):** Vùng ảnh VIS sáng/sương; tương phản nguồn IR khác VIS đáng kể, SAGE đổi thang sáng. So sánh tỷ số contrast phải chuẩn hóa/phân tầng theo cảnh trước khi kết luận “mất IR”.
- **00036/00037, 00389/00395 (contact_sheet_05/06):** Đối chứng tối cho thấy SAGE có thể giữ/làm nổi bật người hoặc xe trong nhiều cảnh. Đó là điều kiện phải kiểm khi thử H1/H2 để tránh đánh đổi lợi ích nhiệt.

## 5. Hai giả thuyết cần đăng ký trước khi thử phương pháp mới

### H1 — Suy giảm cạnh VIS có thể làm mất thông tin hữu ích cho phân đoạn

- **Quan sát hỗ trợ:** Median edge ratio=0.905; ảnh 00239, 00240, 00248 có edge ratio thấp; ở một số cảnh có nhiều biên VIS như biển báo, cột, xe và vạch đường.
- **Chưa chứng minh:** Edge ratio chưa liên hệ với classwise IoU/boundary IoU. Thay đổi độ sáng/độ tương phản có thể khiến Sobel giảm nhưng tín hiệu semantic vẫn tốt.
- **Thử kiểm chứng ưu tiên:** (i) đóng băng B3 VIS và SAGE đã train cùng protocol; xuất prediction **chỉ trên Train**; (ii) tính per-image/class IoU và boundary F1/IoU cho các lớp liên quan; (iii) kiểm tra correlation theo cụm, các ngưỡng, đối chứng ngẫu nhiên và trường hợp ratio thấp nhưng IoU cao.
- **Giải pháp thử nếu qua kiểm chứng:** Một nhánh `HF-VIS-residual` đưa một phần cấu trúc tần số cao VIS vào ảnh SAGE, điều khiển bởi **các cue ở inference không dùng GT** (ảnh VIS/IR/SAGE), thử các ablation `alpha=0`, fixed alpha, có/không clipping. Không mặc định mới về novelty.

### H2 — Tín hiệu nhiệt cục bộ có thể bị suy giảm trong ảnh hợp nhất

- **Quan sát hỗ trợ:** Median person thermal contrast ratio≈0.854 (652 ảnh hợp lệ), vehicle≈0.763 (681 ảnh hợp lệ); một số ảnh person ratio dưới 0.1; IR nhiều khi nổi bật hơn VIS.
- **Chưa chứng minh:** Ratio đo bằng median trong mask GT và ring cục bộ, dùng **GT chỉ để phân tích Train**. Chênh lệch độ sáng do tái tạo ảnh có thể không tương ứng IoU người/xe. Một số ảnh có người rõ trên SAGE dù ratio thấp.
- **Thử kiểm chứng ưu tiên:** Per-instance/local ROI thay vì gộp toàn bộ mask; kiểm signed contrast và contrast-normalization; phân tầng theo sáng/tối; classwise IoU người/ô tô/xe tải/xe buýt/xe máy trên Train; đo cả các trường hợp IR không hữu ích.
- **Giải pháp thử nếu qua kiểm chứng:** `LF-IR-injection / confidence gate` dùng cue thị giác/nhiệt từ ảnh đầu vào, không sử dụng GT hoặc segmentation label tại inference; so với SAGE, VIS, fusion trung bình, fixed IR và global gate; bảo vệ texture VIS.

## 6. Ba điều kiện trước khi tuyên bố có cải tiến

1. **Protocol:** Chỉ thiết kế và hiệu chỉnh trên 992 Train; Dev 228 không dùng cho iterative tuning; Official Test không dùng chọn phương pháp. Giữ nguyên cấu hình và nguồn checkpoint B3 khi so sánh.
2. **Đánh giá:** `mIoU`, classwise IoU và kiểm định theo seed/scene là tiêu chí quyết định; SCD/MS-SSIM/edge/contrast là diagnostic. Không dùng GT làm input của fusion model tại inference.
3. **Nguồn ảnh:** Kiểm tra rằng SAGE PNG trong frozen Q02C thực sự trùng định dạng và nguồn (contract, preprocessing, checkpoint) với ảnh SAGE được dùng huấn luyện B3; khác contract thì không đem so mIoU trực tiếp.

## 7. Gói bàn giao an toàn

- Giữ 10 artifact gốc **nguyên vẹn**. Có thể bổ sung file này vào `SAGE-research/locbe_failure_analysis/` trên branch của bạn; không sửa file của Quân, Mân hoặc Lộc lớn.
- Đề nghị Quân gửi `classwise_comparison.csv` và B3 VIS/SAGE prediction maps theo scene_v1; chỉ cần các file phân tích Train trước, không phải Dev/Test để chốt giả thuyết.
- Đề nghị Mân review novelty/related work cho `HF-VIS-residual` và `LF-IR confidence gate` trước khi nhóm chọn triển khai.
- Nếu chọn prototype, áp dụng một ablation mỗi lần, pin checkpoint + input IDs + source SHA + thông số, xuất PNG cùng export contract và chấm bằng metric đã thống nhất.

**Kết luận:** Nhiệm vụ Failure Analysis đã có deliverable đúng yêu cầu: 992 Train đã phân tích, 30 ví dụ, hai giả thuyết có thể bác bỏ. Chưa có bằng chứng rằng SAGE thực sự làm giảm mIoU ở chính những ảnh này hoặc rằng phương pháp proposed cải thiện hơn baseline; đây là bước cần thực hiện tiếp cùng Quân.
