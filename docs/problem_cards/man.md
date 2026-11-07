# Problem–Solution Card — Mân

**Trạng thái:** Draft, bản 2. Đây là `man_problem.md` trong bảng phân công 08/11.

Đã chạy thử trên 175 ảnh dev bằng **metric tạm**, chưa phải code metric chung của nhóm. Lần chạy này phát hiện
cách chấm ở bản 1 có lỗi, nên mục 10–12 được viết lại **sau khi đã thấy kết quả**. Vì vậy số trên dev
chỉ là thăm dò. Tiêu chí mới cần nhóm xem và kiểm tra lại trên ảnh khác (mục 12).

## 1. Tên problem

Khi IR–VIS lệch, SAGE Student có bị hại **riêng** không, hay chỉ dời nội dung IR như mọi cách ghép?
Căn lại đơn giản có đáng dùng không?

## 2. Failure mode cụ thể

Nếu IR và VIS không khớp, cùng một vật xuất hiện ở hai vị trí khác nhau. Ảnh ghép có thể bị bóng đôi
hoặc mờ cạnh. Failure mode này chỉ thuộc về SAGE nếu SAGE giảm điểm nhiều hơn cách ghép trung bình
đơn giản trên cùng ảnh.

## 3. Evidence

Phân loại nguồn: **team-reproduced**. Số audit lấy từ `data/audit/`. Số dev run là lần chạy thử với
metric tạm, script và số liệu đang ở máy Mân, chưa vào repo.

**Từ audit**

- Audit FMB đo 300 cặp ngẫu nhiên bằng tương quan pha trên bản đồ cạnh. Nhóm 150 cặp tin cậy cao có
  trung vị |lệch| **3,15 px**, P90 **6,14 px**, 1,33% lệch trên 10 px (`data/audit/6_tom_tat_can_chinh.csv`).
- Audit kết luận **không có lệch hệ thống đáng kể**: trung vị dx 1,06 px, dy 0,96 px.
- Phép đo không thấy lệch xoay hay lệch cục bộ.

**Từ dev run thăm dò (175 ảnh, không căn, MS-SSIM tổng, median theo từng ảnh)**

| Độ dịch | SAGE, so với IR gốc | Ghép trung bình, so với IR gốc | SAGE, so với IR đưa vào |
|---:|---:|---:|---:|
| 5 px | −2,38% | −2,64% | +0,10% |
| 10 px | −4,48% | −4,59% | +0,24% |

- Mức giảm gần như chỉ đến từ phần IR. Điểm với VIS gần như không đổi (−0,01% ở 5 px).
- **SAGE không giảm nhiều hơn ghép trung bình.** Ở 5 px, median chênh lệch theo từng ảnh là +0,14 điểm
  phần trăm, khoảng tin cậy 95% theo khối là [0; 0,49]. Mức giảm vì vậy chủ yếu do nội dung IR bị dời,
  chưa phải điểm yếu riêng của SAGE.
- Kết quả khác nhau theo khối dev: mức giảm ở 5 px từ −1,49% đến −3,78%.

**Độ tin cậy của phép đo lệch nền**

- Tương quan pha và ECC cùng hướng nhưng chưa khớp: tương quan 0,62 (x), 0,77 (y), chênh nhau median
  **1,72 px** trên 150 ảnh có tương quan pha ≤ 12 px.
- Hai khung liền kề có ước lượng gần nhau hơn hai khung ngẫu nhiên cùng khối (ECC: 1,32 so với 2,42 px).
  Điều này gợi ý có phần thật, nhưng cũng có thể do cùng cảnh nên cùng sai.
- Đo độc lập bằng biên nhãn chỉ tương quan yếu với ECC (khoảng 0,2). Dời IR theo ECC ở d = 0 còn làm
  cạnh IR nằm trên biên nhãn kém đi (median −1,62%), dù vẫn tốt hơn dời ngược chiều (−9,53%).

Kết luận hiện tại: lệch nền khoảng 3 px **chưa chứng minh được là lệch thật**, vì nó chỉ gấp khoảng
hai lần độ chênh giữa hai cách đo. Chưa có bằng chứng lệch nền này làm SAGE kém đi.

## 4. Giả thuyết nguyên nhân

Student không có bước căn chỉnh và học trên ảnh đã căn. Khi IR lệch, Student có thể tạo bóng đôi nặng
hơn phép cộng đơn giản, nên SAGE sẽ giảm điểm nhiều hơn ghép trung bình.

Giả thuyết bị bác bỏ nếu SAGE giảm không hơn ghép trung bình (mục 12). Dev run thăm dò đang nghiêng về
bác bỏ.

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

Nếu độ dịch tìm được vượt 15 px thì xem là căn hỏng và giữ nguyên IR. ECC chỉ nên dùng khi có lý do
nghi lệch thật, chưa nên bật mặc định trên FMB.

## 7. Vị trí thay đổi trong SAGE

Bước tiền xử lý đầu vào, trước khi gọi Student. Không đổi kiến trúc, loss hay checkpoint.
Dùng cùng repo SAGE (commit `377fdbe`) và checkpoint (SHA256 `ecfde477…`) với baseline của Quân.

## 8. Training/inference requirements

- Chỉ inference, dùng cùng checkpoint và tiền xử lý với baseline của Quân (800×600, không resize).
- Dữ liệu: `data/manifests/dev.csv` (PR #2), sau đó thêm ảnh kiểm tra lại (mục 12). Không dùng test.
- Cần code metric chung của nhóm. QAB/F chưa có trong `configs/evaluation/fusion_metrics.yaml`.
- Thêm OpenCV cho ECC.

## 9. Chi phí GPU

**Đo thực tế:** 2.100 lượt (12 điều kiện × 175 ảnh) mất 307 giây trên RTX 3050 Laptop 6 GB, đã gồm
ECC, tương quan pha và chấm điểm. ECC mất khoảng 154 ms/lần trên CPU. Bộ nhớ đỉnh trên T4 theo log
của Quân là 510 MiB.

## 10. Baseline và đối chứng đơn giản

Mỗi ảnh được dịch IR một lượng d = 0, 2, 5, 10 px theo một trong 4 hướng (trái, phải, lên, xuống).
Hướng cố định theo `image_id` với seed 42, và giữ nguyên ở mọi mức d. Dịch số nguyên px, không nội suy.

| Điều kiện | Ý nghĩa |
|---|---|
| Không căn | Đo mức giảm khi lệch. d = 0 là baseline |
| Ghép trung bình `(Y + IR) / 2` | Đối chứng cho problem, không cần model |
| Căn bằng tương quan pha | Đối chứng đơn giản cho solution |
| Căn bằng ECC | Solution đề xuất |

## 11. Metric chính/phụ

Điểm nền **M** là MS-SSIM tổng với hai ảnh nguồn. Phụ là SCD, thêm QAB/F nếu code metric chung kịp bổ sung.
Tính trên kênh Y, cùng vùng giữa ảnh, bỏ 15 px mỗi cạnh. Mọi chỉ số dưới đây tính theo từng ảnh.

| Ký hiệu | Công thức | Trả lời câu hỏi |
|---|---|---|
| L(d) | % thay đổi của M ở mức d so với d = 0, so với IR gốc | Nội dung IR bị dời ảnh hưởng bao nhiêu |
| **E(d)** | L của SAGE − L của ghép trung bình | **SAGE có bị hại riêng không** |
| A(d) | Như L nhưng so với IR thật sự đưa vào | SAGE có giữ được cái nó nhận không |
| G(d) | % thay đổi của M khi căn, so với chính cách căn đó ở d = 0 | Căn có triệt tiêu được lệch nhân tạo không |
| **S(d)** | Độ dài của [ước lượng(d) − ước lượng(0) − lượng dịch thật] | **Phép căn sai bao nhiêu px** |
| C | % ảnh có ước lượng lệch nền trên 15 px | Tỉ lệ căn hỏng nặng |

Quy ước bắt buộc:

- **Không dùng điểm ở d = 0 để kết luận căn đúng hay sai.** IR gốc có thể đã lệch sẵn, nên mọi phép căn
  đều rời khỏi IR gốc và bị trừ điểm, kể cả khi căn đúng. Với ECC, điểm giảm tăng theo độ dời: −0,11%
  khi dời ≤ 1 px, −2,01% khi dời 3–6 px. Không có đáp án thật cho vị trí đúng ở d = 0.
- **S chỉ đo được phần lệch nhân tạo**, vì chỉ phần đó biết đáp án. Tương quan pha có thể có S rất nhỏ
  mà vẫn sai, nên phải xem cùng C.
- **Báo cáo:** mean, std, median, số ảnh hợp lệ, kết quả theo từng khối dev và theo độ sáng VIS.
  Khoảng tin cậy phải lấy mẫu theo khối, vì ảnh cùng khối không độc lập.

## 12. Tiêu chí thành công/bác bỏ

Các ngưỡng này được đặt **sau** dev run thăm dò, nên không dùng chính dev run đó để kết luận. Cần chốt
trước khi kiểm tra lại, bằng metric chung của nhóm trên 5 khối lớn nhất trong `train.csv`. Không dùng test.

| Kết quả | Điều kiện | Dev run thăm dò (chưa tính) |
|---|---|---|
| Problem được xác nhận | Median E(5) ≤ −1% và cận trên khoảng tin cậy 95% < 0 | +0,14; [0; 0,49] |
| Problem bị bác bỏ | Cận dưới khoảng tin cậy 95% của E(5) > −0,5% | Nghiêng về mức này |
| Chưa kết luận | Không thuộc hai mức trên. Báo cáo số, không tự đổi ngưỡng | |
| Solution thành công | Ở d = 5 và 10: median G trong ±0,5%, S median ≤ 1 px, S P90 ≤ 2 px, và C ≤ 2% | Chưa đạt |

Dev run thăm dò cho solution:

| Cách | G(5) | G(10) | S median (5 / 10) | S P90 (5 / 10) | C ở d = 5 |
|---|---:|---:|---:|---:|---:|
| ECC | −0,19% | −0,20% | 1,09 / 1,20 px | 3,62 / 5,45 px | 1,1% |
| Tương quan pha | 0,00% | 0,00% | 0,01 / 0,02 px | 0,05 / 0,12 px | 14,9% |

ECC ít hỏng nặng nhưng chưa đủ chính xác. Tương quan pha ổn định nhưng hỏng nặng ở gần 15% số ảnh.

**Kết quả theo tiêu chí bản 1 (giữ lại để minh bạch):** ở d = 5, MS-SSIM giảm 2,38%, thuộc mức "chưa
kết luận". ECC thất bại: hồi phục 37,7% ở d = 5, 58,5% ở d = 10, và giảm 0,77% ở d = 0. Bản 1 bỏ qua
việc IR gốc có thể lệch sẵn và chưa định nghĩa cách tính hồi phục. Tùy cách tính, con số của ECC ở
d = 5 chạy từ khoảng 15% đến 39%.

Bác bỏ vẫn là kết quả có giá trị, vì cho thấy lệch còn sót trong FMB không đáng lo.

## 13. Novelty risk và fallback

- **Trùng related work:** căn chỉnh IVIF đã có nhiều bài, gồm FusionRegister. Card này không đề xuất
  phương pháp căn chỉnh mới, mà đo độ bền của SAGE và thử một cách căn rẻ, không cần train.
- **Rủi ro kỹ thuật:** ECC có thể hỏng ở ảnh tối hoặc sương vì ít cạnh chung.
- **Rủi ro dữ liệu:** ảnh dev lấy từ train chính thức, nên checkpoint có thể đã thấy khi train (chưa
  xác minh). Kết quả dùng để so mức giảm giữa các độ lệch, không xem là số trên ảnh mới.
- **Rủi ro về độ lệch nền:** d = 0 chưa chắc khớp hoàn toàn, và chưa có đáp án thật. Biên nhãn không
  dùng làm trọng tài được, vì chính nhãn cũng lệch khoảng 2 px so với cả IR và VIS.
- **Rủi ro từ mẫu nhỏ:** dev chỉ có 5 khối, nên khoảng tin cậy theo khối còn thô.
- **Rủi ro đổi tiêu chí:** bản 2 viết sau khi thấy số, nên bắt buộc kiểm tra lại trên ảnh khác.
- **Fallback:** chuyển sang phân tích SAGE theo điều kiện đêm, sương, khói, chỉ cần ảnh ghép và metric.

Theo [`TEMPLATE.md`](TEMPLATE.md).
