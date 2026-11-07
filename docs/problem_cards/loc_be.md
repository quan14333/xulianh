# Problem–Solution Card — Lộc bé

**Trạng thái: Draft / evidence-backed hypothesis.** Chưa phải approved method;
chưa công nhận novelty, chưa có adapter/prototype được kiểm chứng.

Theo [template của nhóm](TEMPLATE.md). Card tích hợp bản thảo của Lộc bé với
[báo cáo và giới hạn thực nghiệm](../../reports/weekly/loc_be_2026-11-08.md).

## 1. Problem

**Tên problem:** Metric disagreement and localized IR-contrast fidelity in
lightweight semantic-guided IVIF.

**Failure mode cần kiểm chứng:** Phép đo biên toàn ảnh và fidelity tương phản IR
cục bộ có thể đánh giá khác nhau. Dấu hiệu từ proxy chưa chứng minh fused làm mất
người/xe; cần ROI có annotation và downstream evaluator để xác nhận ảnh hưởng.

## 2. Evidence

- **Team-reproduced exploratory evidence:** run
  `sage_pcdfusion_08nov_20261106_170517`; cùng 10 ID FMB/test repackaged, 640×512.
  Không phải official full-test 280 ảnh, không phải dev độc lập.
- SAGE có SCD/MS_SSIM_sum cao hơn PCDFusion ở 10/10 ảnh; PCDFusion có edge_corr_IR
  cao hơn ở 10/10 ảnh. Mean local_IR_abs_error SAGE 15.2133, PCDFusion 19.0539;
  PCDFusion tốt hơn trên 4/10 ảnh theo proxy này. Xem
  [per-image](../../reports/tables/loc_be/pcdfusion_per_image.csv) và
  [paired comparison](../../reports/tables/loc_be/pcdfusion_paired_comparison.csv).
- 9 ROI thủ công trên 5 cảnh: mean/ring=24 có contrast SAGE cao hơn và gần IR hơn
  ở 9/9 ROI; contrast cao hơn theo median/ring=12/24/36 là 6/9, 7/9, 8/9. Không
  thấy SAGE dưới VIS qua cả 6 cấu hình theo tiêu chí hẹp này. Các rectangle hậu
  nghiệm, không phải segmentation GT. Replay CPU khớp CSV/checksum gốc.
- Chưa có dev run, team-reproduced mIoU hoặc ablation adapter. SCD/MS-SSIM chưa
  xác nhận parity với paper. Không dùng số paper thay cho kết quả nhóm.

## 3. Giả thuyết nguyên nhân

Metric global có thể không phản ánh ổn định tương phản vùng thermal-salient.
Độ sáng IR cần phân biệt tín hiệu mục tiêu với nền/đèn/nhiễu; giữ tương phản cao
không luôn là giữ thông tin đúng. Giả thuyết bị bác bỏ nếu không tìm thấy bất
đồng có ý nghĩa trên ROI có nhãn và dev từ train.

## 4. Related works

1. Wu et al., **Every SAM Drop Counts: Embracing Semantic Priors for Multi-Modality
   Image Fusion and Beyond**, CVPR 2025, pp. 17882–17891.
   [Paper chính thức](https://openaccess.thecvf.com/content/CVPR2025/html/Wu_Every_SAM_Drop_Counts_Embracing_Semantic_Priors_for_Multi-Modality_Image_CVPR_2025_paper.html),
   [SAGE source](https://github.com/RollingPlain/SAGE_IVIF).
   Semantic priors và distillation là nền tảng của baseline frozen Student.
2. Liu et al., **PCDFusion: Proposal-Context-Detail Bayesian Rendering for
   Infrared-Visible Image Fusion**, arXiv:2610.23663, 2026.
   [Paper](https://arxiv.org/abs/2610.23663),
   [source](https://github.com/BIT-RuiLiu/PCDFusion).
   Local reliability/rendering là prior art gần; không tuyên bố novelty cho
   reliability map đơn thuần. Không chép số benchmark paper vào bảng nhóm.

Hai nguồn chính được đối chiếu khi tích hợp. Survey sâu về region-adaptive fusion
còn thiếu; các công trình khác trong bản nháp chưa được dùng để xác nhận novelty.

## 5. Solution đề xuất

**Candidate chưa triển khai:** sau frozen SAGE Student, dùng reliability map từ
tương phản IR–VIS và độ nhất quán biên để điều khiển residual luminance refinement
nhỏ, có giới hạn biên độ. Thử đối chứng no-training blend/sharpening trước; chỉ
fine-tune adapter nếu dev có cải thiện thật, sau khi nhóm duyệt card/protocol.

## 6. Vị trí thay đổi trong SAGE

Sau đầu ra Y_fused của Student (`model_sub/model.py`, pin
`377fdbef6b8b6ccbd1139358ad66bb41ea084dc1`), trước ghép chroma và phục hồi RGB.
Đây là đề xuất vị trí integration; chưa sửa upstream/baseline/evaluator hoặc thêm
module vào `src/methods/`.

## 7. Yêu cầu training/inference

Frozen pretrained SAGE và PCDFusion làm đối chứng. Cần train/dev manifest tách từ
train, checkpoint pin/hash, preprocessing thống nhất và ROI có nhãn. Adapter nếu
được duyệt chỉ dùng train/dev; chưa đề xuất train lại SAM/Teacher. Tuyệt đối không
tune trên 10 test ID đã xem rồi gọi chúng là test độc lập.

## 8. Chi phí GPU

Adapter: **chưa đo**, không có ước lượng được duyệt. Log SAGE v3 ghi Tesla T4,
135.617 tham số Student, forward mean 21.141 ms/ảnh; PCDFusion ghi subprocess
25.560385 s gồm khởi tạo/load. Hai thời gian không cùng điều kiện, không so trực
tiếp hay dùng để suy ra chi phí adapter. GPU inference chưa chạy lại khi tích hợp.

## 9. Baseline và đối chứng đơn giản

- Frozen SAGE Student với đúng checkpoint và preprocessing.
- Y_AVG, Y_MAX, global sharpening, IR blend hệ số cố định; mọi hệ số phải đăng ký
  trên dev trước đánh giá. PCDFusion frozen là đối chứng học sâu.
- Chưa có kết quả sharpening/blend/adapter. Y_AVG/Y_MAX có số liệu trong run gốc,
  không phải phương pháp mới.

## 10. Metrics

Primary **chưa khóa**; đề nghị local contrast abs error trên ROI có annotation và
SCD đã được owner xác minh. Secondary: EN/SD/MS-SSIM/edge correlation; mIoU chỉ khi
có segmentation model, label mapping và evaluator chung. Contrast rectangle và
heuristic-mask là hai protocol khác nhau, phải báo riêng.

## 11. Tiêu chí quyết định

Chưa đăng ký ngưỡng số và chưa được nhóm phê duyệt. Trước dev run phải khóa split,
annotation, primary metric, ngưỡng cải thiện và mức suy giảm chấp nhận được.

- Hướng thành công dự kiến: paired error giảm ở phần lớn ảnh dev có nhãn, giữ
  chất lượng global/semantic và có ích hơn đối chứng đơn giản.
- Bác bỏ: không tái hiện trên dev; chỉ cải thiện 10 test demo; hoặc blend/sharpening
  đạt tương đương với chi phí thấp hơn. Không điều chỉnh tiêu chí sau khi xem test.

## 12. Novelty risk và fallback

Region-adaptive fusion có prior art gần, rủi ro incremental cao. Cần survey trước
khi claim đóng góp. Fallback: failure analysis/robustness benchmark có nhãn, phép
đo và replay rõ ràng. **Đã có nghiên cứu thực nghiệm và trade-off; chưa có phương
pháp cải tiến mới được phê duyệt.**

## Review

- Owner: Lộc bé.
- Reviewer cần: Quân (integration), Lộc lớn (metric/tests), Mân khi cần review data/protocol.
- Ngày quyết định: chưa có; chờ quyết định nhóm, không đánh dấu Approved.
- Run: `sage_pcdfusion_08nov_20261106_170517`.
- Hướng dẫn/artifact: [README](../../notebooks/loc_be/README.md); Kaggle URL **link pending**.
