# Problem–Solution Card — Mân

**Trạng thái:** Draft. Đây là `man_problem.md` trong bảng phân công 08/11.
Chưa chạy thí nghiệm. Tác động của lệch tới SAGE là giả thuyết, chưa đo. Các ngưỡng ở mục 12 là
đề xuất, cần nhóm xem trước khi chạy.

## 1. Tên problem

SAGE Student chịu được IR–VIS lệch tới mức nào, và căn lại đơn giản có hồi phục được không?

## 2. Failure mode cụ thể

Nếu IR và VIS không khớp, cùng một vật xuất hiện ở hai vị trí khác nhau. Ảnh ghép có thể bị bóng đôi
hoặc mờ cạnh. Biểu hiện đo được dự kiến là MS-SSIM, SCD và QAB/F giảm khi độ lệch tăng.

## 3. Evidence

Phân loại nguồn: **team-reproduced** cho số audit. Chưa có dev run.

- Audit FMB đo 300 cặp ngẫu nhiên bằng tương quan pha trên bản đồ cạnh. Nhóm 150 cặp tin cậy cao có
  trung vị |lệch| **3,15 px**, P90 **6,14 px**, 1,33% lệch trên 10 px (`data/audit/6_tom_tat_can_chinh.csv`).
- Audit cũng kết luận **không có lệch hệ thống đáng kể**: trung vị dx 1,06 px, dy 0,96 px.
- Số trên có thể gồm nhiễu đo, và phép đo không thấy lệch xoay hay lệch cục bộ.

Vì vậy evidence chỉ cho thấy khoảng lệch 2–6 px là đáng thử. Nó **chưa chứng minh** lệch làm SAGE
kém đi. Card này không khẳng định FMB bị lệch nặng.

## 4. Giả thuyết nguyên nhân

SAGE dùng ảnh đầu vào đã căn chỉnh. Student không có bước căn chỉnh, nên khi IR bị dịch, chất lượng
ảnh ghép sẽ giảm theo độ dịch và giảm rõ từ 5 px.

Giả thuyết bị bác bỏ nếu chỉ số gần như không đổi ở 5 px (mục 12).

## 5. Related works

1. C. Bian và cộng sự, *FusionRegister: Every Infrared and Visible Image Fusion Deserves
   Registration*, CVPR 2026, arXiv 2603.07667. Căn chỉnh sau khi ghép, chỉ sửa vùng lệch.
2. D. Wang, J. Liu, X. Fan, R. Liu, *Unsupervised Misaligned Infrared and Visible Image Fusion via
   Cross-Modality Image Generation and Registration*, IJCAI 2022, tr. 3508–3515,
   DOI 10.24963/ijcai.2022/487.
3. H. Xu, J. Yuan, J. Ma, *MURF: Mutually Reinforcing Multi-Modal Image Registration and Fusion*,
   IEEE TPAMI 45(10), 2023, tr. 12148–12166, DOI 10.1109/TPAMI.2023.3283682.
4. G. D. Evangelidis, E. Z. Psarakis, *Parametric Image Alignment Using Enhanced Correlation
   Coefficient Maximization*, IEEE TPAMI 30(10), 2008, tr. 1858–1865, DOI 10.1109/TPAMI.2008.113.
   Đây là phương pháp ECC dùng ở mục 6.

## 6. Solution đề xuất

Căn lại IR theo VIS bằng **ECC** trước khi đưa vào SAGE, không sửa model. ECC chạy trên bản đồ cạnh
Sobel của cả hai ảnh, vì độ sáng IR và VIS khác nhau nhiều. Mô hình biến đổi là tịnh tiến.

## 7. Vị trí thay đổi trong SAGE

Bước tiền xử lý đầu vào, trước khi gọi Student. Không đổi kiến trúc, loss hay checkpoint.
Dùng lại notebook inference của Quân (PR #1), có thay danh sách ảnh và thêm bước biến đổi IR.

## 8. Training/inference requirements

- Chỉ inference, dùng cùng checkpoint và tiền xử lý với baseline của Quân (800×600, không resize).
- Dữ liệu: 175 ảnh trong `data/manifests/dev.csv` (PR #2). Cần đưa ảnh train lên Kaggle, vì Kaggle
  của Quân hiện chỉ có test.
- Cần code metric của Lộc lớn. QAB/F chưa có trong `configs/evaluation/fusion_metrics.yaml`.
- Thêm OpenCV cho ECC.

## 9. Chi phí GPU

**Ước lượng:** 12 điều kiện × 175 ảnh = 2.100 lượt, khoảng **11 phút** trên Tesla T4. Con số suy từ
log của Quân (trung bình 320 ms/ảnh trên 280 ảnh test, đỉnh 510 MiB). Chưa tính thời gian chạy ECC.

## 10. Baseline và đối chứng đơn giản

Mỗi ảnh được dịch IR một lượng d = 0, 2, 5, 10 px theo một trong 4 hướng (trái, phải, lên, xuống).
Hướng cố định theo `image_id` với seed 42, và giữ nguyên ở mọi mức d. Dịch số nguyên px, không nội suy.

| Điều kiện | Ý nghĩa |
|---|---|
| Không căn | Đo mức giảm khi lệch. d = 0 là baseline |
| Căn bằng tương quan pha | Đối chứng đơn giản, cùng cách đo trong audit |
| Căn bằng ECC | Solution đề xuất |

3 điều kiện × 4 mức d = 12. Riêng d = 0 có căn lại dùng để xem việc căn có làm hỏng ảnh tốt không.

## 11. Metric chính/phụ

- **Chính:** MS-SSIM, định nghĩa là tổng MS-SSIM với hai ảnh nguồn.
- **Phụ:** SCD. Thêm QAB/F nếu Lộc lớn kịp bổ sung.
- **Báo cáo:** mean, std, median, số ảnh hợp lệ, delta theo từng ảnh so với d = 0.

Quy ước để so công bằng:

- Mọi metric so với **IR gốc chưa dịch** và VIS. Nếu so với IR đã dịch thì ảnh ghép lệch vẫn có thể
  được điểm cao.
- Tính trên cùng vùng giữa ảnh, bỏ 15 px mỗi cạnh để loại phần viền do dịch ảnh.
- Tách kết quả theo độ sáng VIS, vì audit cho thấy ảnh tối và sương đo căn chỉnh kém tin cậy.

## 12. Tiêu chí thành công/bác bỏ

Đề xuất, chốt trước khi chạy. Tất cả tính trên MS-SSIM ở d = 5 px, so với d = 0 theo từng ảnh.

| Kết quả | Điều kiện |
|---|---|
| Problem được xác nhận | Median giảm ≥ 3% và ít nhất 75% số ảnh giảm |
| Problem bị bác bỏ | Median giảm < 1%. Kết luận SAGE chịu được lệch cỡ này |
| Chưa kết luận | Nằm giữa hai mức trên. Báo cáo số, không tự đổi ngưỡng |
| ECC thành công | Hồi phục ≥ 50% mức giảm ở d = 5 và d = 10, và hơn tương quan pha |
| ECC thất bại | Không đạt như trên, hoặc làm giảm > 0,5% ở d = 0 |

Bác bỏ vẫn là kết quả có giá trị, vì cho thấy lệch còn sót trong FMB không đáng lo.

## 13. Novelty risk và fallback

- **Trùng related work:** căn chỉnh IVIF đã có nhiều bài, gồm FusionRegister. Card này không đề xuất
  phương pháp căn chỉnh mới, mà đo độ bền của SAGE và thử một cách căn rẻ, không cần train.
- **Rủi ro kỹ thuật:** ECC có thể hỏng ở ảnh tối hoặc sương vì ít cạnh chung.
- **Rủi ro dữ liệu:** ảnh dev lấy từ train chính thức, nên checkpoint có thể đã thấy khi train (chưa
  xác minh). Kết quả dùng để so mức giảm giữa các độ lệch, không xem là số trên ảnh mới.
- **Rủi ro về độ lệch nền:** d = 0 chưa chắc khớp hoàn toàn, nên độ lệch thật có thể khác d.
- **Fallback:** chuyển sang phân tích SAGE theo điều kiện đêm, sương, khói, chỉ cần ảnh ghép và metric.

Theo [`TEMPLATE.md`](TEMPLATE.md).
