# Audit source SAGE Student và baseline B3

Ngày audit: 14/11/2026. Deadline: 19/11/2026. **Audit đã thực hiện; chưa train, chưa triển khai trainer, chưa có kết quả fine-tuning.**

Phạm vi kiểm chứng: đọc source local, load checkpoint trên CPU, kiểm tra manifest Train/Dev, query metadata GPU và probe một mask Train. Không forward model, backward, optimizer step hoặc evaluation FMB Test.

## 1. Phiên bản và bằng chứng

| Thành phần | Phiên bản / trạng thái |
|---|---|
| Workspace | D:\SAGE_Research\xulianh-finetune |
| Branch / HEAD | feature/locbe-sage-finetune-1911 / c4a6de4f77c5cbaf9790ef566a1565f8bbc4b16a |
| SAGE upstream | D:\SAGE_Research\SAGE_IVIF / 377fdbef6b8b6ccbd1139358ad66bb41ea084dc1; tracked tree sạch khi bắt đầu |
| B3 reference | D:\SAGE_Research\xulianh-b3-review / detached 58a24847ac68aca6f232194cf1bc84d9fcf4be52; không sửa code Mân |
| Python kiểm chứng | D:\SAGE_Research\xulianh\.venv-sage\Scripts\python.exe; Python 3.11.9 |
| Checkpoint | SAGE_IVIF/checkpoints/checkpoints.pt; 579020 bytes |
| SHA256 | ecfde477ca4cd65401f207d5dc91ce6c475a49891d768dbdb7a66366255ee836 |
| MD5 | 241396a51a3c5e85e360a409d9e8fe4d; khớp manifest B3 |
| Load thực tế | torch.load(map_location=cpu, weights_only=True), Network().load_state_dict(strict=True): PASS |
| Tensors | 121 state entries; missing/unexpected=0; mọi floating tensor hữu hạn; từng tensor load bằng chính xác checkpoint |
| Kích thước model | 135617 parameters; 18 Conv2d; 17 BatchNorm2d |

Checkpoint không có optimizer, scheduler hay training-ID list. Fine-tuning từ weights này không phải resume toàn bộ trạng thái training gốc. Chưa thể xác minh Dev độc lập với pretraining fusion; không dùng suy luận từ BatchNorm thay cho provenance.

| Manifest | Số ID duy nhất | SHA256 |
|---|---:|---|
| train.csv | 1045 | 86158cdcd02ad08c1143135f35b06c1803489e842c4eda0a79c90198590b3071 |
| dev.csv | 175 | 4720243878c093a8717f5c09a907b164618c97e8fb68ffb8dff2db4e93ba6f15 |

Hai tập không giao nhau. Hợp 1220 IDs khớp toàn bộ tên PNG trong FMB/train/train/Visible và Infrared. Đã xác nhận tồn tại 2440 đường dẫn VIS/IR của Train+Dev; chưa decode/QA toàn bộ ảnh. Hash hai manifest khớp worktree B3. Không sửa manifest hoặc mở ảnh/nhãn FMB Test.

## 2. Kiến trúc Student thực sự

[model_sub/model.py](https://github.com/RollingPlain/SAGE_IVIF/blob/377fdbef6b8b6ccbd1139358ad66bb41ea084dc1/model_sub/model.py#L6) định nghĩa Network(dim=32). Input (VIS_Y,IR) gồm hai tensor N×1×H×W được nối theo channel. Năm khối module1..5, mỗi khối ba Conv3×3 + BatchNorm + LeakyReLU(0.01), giữ 32 channels. Hai MaxPool2× tạo scales 1,1/2,1/4. Decoder bilinear align_corners=False, skip-concatenation và hai Conv1×1 giảm 64→32. Conv3×3 cuối tạo một channel, sau đó Tanh.

Forward trả (F,(x1,x2,x3,x4,x5)), output cùng H×W nhờ interpolation theo skip. Raw F trong [-1,1]; không sigmoid, không min-max và không (F+1)/2. Các lớp Dense trong common không được Network Student gọi. Mô tả “dense blocks” trong README không thay cho source kiến trúc đã load.

Student forward không cần Teacher, SAM, semantic masks, X-Decoder hoặc SegFormer. Điều này không có nghĩa loss Student gốc cũng độc lập với các thành phần đó.

## 3. Teacher và training loop gốc

[model_main/model.py](https://github.com/RollingPlain/SAGE_IVIF/blob/377fdbef6b8b6ccbd1139358ad66bb41ea084dc1/model_main/model.py#L14) nhận (V,I,P_V,P_I): encoder ảnh nguồn, encoder các semantic-prior patches VIS/IR, SpatialTransformer trong [attention.py](https://github.com/RollingPlain/SAGE_IVIF/blob/377fdbef6b8b6ccbd1139358ad66bb41ea084dc1/model_main/attention.py), decoder với skip. Năm feature maps được nén về 32 channels để align Student. Đầu ra Teacher kết thúc bằng LeakyReLU, không phải Tanh.

[train.py](https://github.com/RollingPlain/SAGE_IVIF/blob/377fdbef6b8b6ccbd1139358ad66bb41ea084dc1/train.py#L32) mặc định batch4, 100epochs, crop192×256 (H×W), seed2. Adam: Teacher LR5e-4, Student LR2e-3; cosine eta_min1e-5, T_max=epochs×len(loader). student_epoch=1, ini_epoch default=1. Cả hai model được train() mỗi epoch.

Từ student_epoch, batch index chẵn update Student, batch lẻ update Teacher. Khi Student học, Teacher forward trong no_grad để tạo image/feature reference; khi Teacher học, Student forward no_grad làm feedback. Không phải mỗi batch update cả hai model; no_grad cũng không tự chuyển model sang eval. Scheduler riêng chỉ step khi mạng tương ứng update, nhưng T_max tính theo toàn loader. Đây là hành vi source, không tự “sửa cho hợp lý” rồi gọi tái lập nguyên bản.

Teacher thêm segmentation objective qua X-Decoder frozen weights, nhưng gradient đi về fused image. Không có Teacher checkpoint trong source local. checkpoint_teacher rỗng khởi tạo Teacher mới, không phải tiếp tục pretrained Teacher. Student pretrained cũng chỉ load khi truyền checkpoint_student.

## 4. Loss thực thi và phụ thuộc

E(A,B)=mean((A−B)^2). G(X)=|Kx*X|+|Ky*X| với Sobel3×3, zero-padding1; g(F,A,B)=mean|G(F)−max(G(A),G(B))|. R là output Teacher.

Theo [losses.py](https://github.com/RollingPlain/SAGE_IVIF/blob/377fdbef6b8b6ccbd1139358ad66bb41ea084dc1/model_sub/losses.py#L15) và [process_student_batch](https://github.com/RollingPlain/SAGE_IVIF/blob/377fdbef6b8b6ccbd1139358ad66bb41ea084dc1/train.py#L294):

$$
L_s=3[3E(F,V)+2E(F,I)+4E(F,R)]
+3[9g(F,V,I)+3g(F,R,R)]
+C(F,V,I,R,P_V,P_I)/3000
+0.3\sum_{m=1}^{5}[1-\operatorname{mean}_{batch}\cos(\mathrm{vec}(f_s^m),\mathrm{vec}(f_t^m))].
$$

| Thành phần | Teacher | SAM / patches | GT class mask |
|---|---|---|---|
| E(F,V), E(F,I), g(F,V,I) riêng lẻ | Không | Không | Không |
| E(F,R), g(F,R,R), cosine features | Có | Gián tiếp qua Teacher | Không trực tiếp |
| Contrast C | Reference R | Có encoder SAM và patches | Không trực tiếp |
| Teacher base: E(R,V)+E(R,I)+2g(R,V,I) | Teacher output | Teacher input cần patches | Không trực tiếp |
| Teacher segmentation | Teacher→RGB→X-Decoder | X-Decoder/tokenizer | Có label_mask |
| Teacher feedback Student | Cả hai networks | Gọi cùng Losses.cal | Qua nhánh segmentation riêng |

Teacher thêm 0.3*CE và, từ student_epoch, thêm 0.1*Losses.cal(R,V,I,F,patches)+0.01*sum(1−cos(feature_teacher,feature_student)). Không nhầm với hệ số cosine0.3 ở bước Student.

[common.py](https://github.com/RollingPlain/SAGE_IVIF/blob/377fdbef6b8b6ccbd1139358ad66bb41ea084dc1/model_sub/common.py#L265) tính C bằng tổng các tỷ số khoảng cách feature: tử fused–reference, mẫu fused–nguồn tương ứng cộng1e-7, theo từng patch VIS/IR. Dù tên biến vgg_loss, code dùng SAM encoder + InstanceNorm; VGG đã comment. load_sam tạo encoder depth2 và lọc checkpoint keys, không phải full ViT-B depth12.

SAM perceptual module được eval nhưng source đặt requires_grad=True. Optimizer Student chỉ nhận model_student.parameters(): SAM có gradient/bộ nhớ nhưng không được optimizer đó update. Constructor Losses vẫn load SAM/hard-coded CUDA dù người dùng định đặt contrast coefficient=0. Không dùng constructor này cho baseline B.

## 5. Dataset, masks và các vấn đề đã phát hiện

[dataset_teacher_FMB.py](https://github.com/RollingPlain/SAGE_IVIF/blob/377fdbef6b8b6ccbd1139358ad66bb41ea084dc1/dataset/dataset_teacher_FMB.py#L16) cần Vis/Ir/Label/Mask_cache, không đọc manifest để giới hạn IDs. SAM ViT-B tạo binary proposals riêng VIS/IR, sort area, cache default20 và dùng4. Dataset nhân proposals vào ảnh rồi trả masked luminance/grayscale intensities ở ys/irs. Vì vậy vi_mask/ir_mask trong train không còn đơn thuần binary masks hoặc FMB class IDs; contrast code còn nhân fused với các tensor này.

Có label-aware random crop và horizontal flip; thiếu vùng thì lấp bằng ảnh nguồn. Không trỏ loader này thẳng vào FMB và mặc định có Development split. GT semantic label và SAM region proposals là hai loại dữ liệu khác nhau.

Các phát hiện cần Quân xác nhận; không sửa source trong audit:

1. test.py:16 import model_sub.model_sub, file đó không có; model_sub/model.py có. Audit load đúng model_sub.model.Network.
2. train.py:64 hard-code CUDA_VISIBLE_DEVICES='1', không hợp máy chỉ có GPU0.
3. Checkpoint SAM dùng đường Linux /mnt/disk3/...; local chỉ có checkpoints/checkpoints.pt, thiếu SAM generator/encoder weights, xdecoder_focall_last.pt và local_clip_model. Config được gọi ở ./configs/xdecoder/... nhưng file local nằm dưới xdecoder/configs/xdecoder/...
4. ini_epoch khai báo type=str dù default1 là số; truyền CLI có thể gây so sánh int/str. Không import train.py để “audit” vì parse_args và imports nặng ở top level.
5. **CPU probe Train00001:** raw IDs [0,1,2,3,6,7,8,9,13]; ToTensor chia255 rồi long() cho [0]. Với mask FMB ID0..14, target CE bị mất class IDs. Đây là kiểm chứng preprocessing, không phải training/mIoU. Không dùng nhãn đó để xây loss candidate.
6. ROI crop dùng mask>0.5, không hợp class IDs đã chia255; có nhánh dùng label đã crop làm ROI cho ảnh còn ở tọa độ gốc. Cần test alignment riêng, chưa execute toàn bộ loader.
7. Có nhánh OOM bỏ qua segmentation/Student feedback, khiến objective đổi theo batch. Baseline mới phải fail rõ, không sao chép cách bỏ loss để chạy tiếp.

Ngoài ra, train.py:440–444 xếp prompts X-Decoder theo road trước, background cuối và đặt num_classes=14; FMB lại có ID0=unlabeled, ID1=road, ID14=pole. Logit-channel/GT remap còn phải kiểm chứng trước A/C. Chỉ sửa phép chia255 chưa đủ chứng minh segmentation loss đã đúng. Không suy rằng background prompt tự tương ứng ID0.

## 6. Preprocessing và hai RGB contracts

[dataset_test.py](https://github.com/RollingPlain/SAGE_IVIF/blob/377fdbef6b8b6ccbd1139358ad66bb41ea084dc1/dataset/dataset_test.py) dùng OpenCV BGR→YCrCb cho VIS, BGR→GRAY cho IR, ToTensor(uint8) chia255 thành float[0,1]. Thứ tự YCrCb là Y,Cr,Cb; biến cb của source thực ra giữ Cr, biến cr giữ Cb. Giữ theo vị trí kênh, không đổi theo tên. Model nhận (Y,IR). Test transform rỗng; không resize/ImageNet normalization mặc định.

[utils.YCrCb2RGB](https://github.com/RollingPlain/SAGE_IVIF/blob/377fdbef6b8b6ccbd1139358ad66bb41ea084dc1/utils.py#L174) tính:

$$R=F+1.403(Cr-0.5),\quad G=F-0.714(Cr-0.5)-0.344(Cb-0.5),\quad B=F+1.773(Cb-0.5).$$

[test.py](https://github.com/RollingPlain/SAGE_IVIF/blob/377fdbef6b8b6ccbd1139358ad66bb41ea084dc1/test.py#L67) không clamp Y trước RGB, không min-max. torchvision save_image mặc định không normalize: (255*RGB+0.5).clamp(0,255) rồi uint8. Đã đọc implementation torchvision0.26.0+cu130 local. Clamp RGB xảy ra khi lưu; utils hard-code CUDA cho matrix/bias.

[B3 xuat_anh.py](https://github.com/quan14333/xulianh/blob/58a24847ac68aca6f232194cf1bc84d9fcf4be52/src/segmentation/xuat_anh.py#L61) lại clamp F về[0,1], nhân255, torch.round, byte **trước** ghép Cr/Cb uint8 và cv2YCrCb→BGR/imwrite. TB dùng trung bình float64, numpy.round, uint8. Round của torch/numpy là tie-to-even, khác cộng0.5 ở save_image.

**Synthetic contract probe, không phải model output:** F=−0.1,Cr=1,Cb=128/255: source PNG RGB=[153,0,0], B3=[178,0,0]. Tại0.5 mức xám, cộng0.5→1 còn round→0. Hai exporter không tương đương tuyệt đối. Chưa đo sai khác trên FMB. Proposal chọn B3 contract cho cả pretrained control và candidate khi so B3; giữ source RGB thành contract đối chiếu riêng.

## 7. B3 của Mân

Đã đọc README, chung.py, xuat_anh.py, train_seg.py, eval_seg.py, báo cáo khóa và artifact manifest tại commit B3 ở trên:

- PNG cùng image_id, full H×W,3channels. Loader BGR→RGB; (RGB/255−mean)/std với mean(.485,.456,.406),std(.229,.224,.225).
- Segmentation augmentation scale.5–2,crop512,hflip.5; masks nearest-neighbor,pad255. Khác augmentation fusion192×256.
- Locked final:1220 Train, --gop_dev --lop0, nvidia/mit-b3,15outputs ID0..14, CE ignore255; ID0 là lớp được train.
- 100epochs,batch8,accum1,seeds0/1/2,AdamW encoderLR6e-5/head×10,wd.01,warmup1500,poly power1,AMPfloat16+GradScaler,drop_lastTrue.152batches/epoch,15200updates; chọn last.pt.
- Defaults không --lop0 là14class/remap; không --gop_dev là1045Train/175Dev/chọn best. Không dùng defaults để gọi final-compatible.
- Eval interpolate logits bilinear align_corners=False về inputH×W rồi argmax;15class predictions lưu ID0..14.
- Primary mIoU_co_mat loại GT row0, prediction0 trên foreground vẫn tính sai, average trên **GT-present classes**, không union-present. Report273ảnh có13classes hiện diện; không hard-code Dev có13classes.
- PersonID8, index7 trong bảng foreground14. mIoU_15_nan0,mIoU_15_mrfs và primary khác nhau; không trộn với B0 Dev mIoU15 cũ.
- Manifest có12runs VIS/IR/TB/SAGE×seeds0/1/2,last.pt,MD5. Weights/confusions lớn ở private artifact storage, không có trong worktree; chưa tải/verify nội dung B3 checkpoints. Notebook hiện chọn seed2 không có nghĩa toàn bộ report chỉ một seed.

Xem [B3_INTEGRATION_PLAN.md](B3_INTEGRATION_PLAN.md) cho công thức metric, layout và cách thêm candidate mà không sửa Mân.

## 8. Môi trường và giới hạn

PyTorch2.11.0+cu130; RTX5060 Laptop,compute capability12.0,VRAM8546484224bytes. Chỉ query metadata; **training GPU/VRAM/time NOT_MEASURED**. Load state trên CPU không chứng minh backward trên GPU.

Đủ torch,numpy2.4.6,cv2 5.0.0.93 cho loss B tối thiểu. Có torchvision0.26.0+cu130,pandas3.1.1,PyYAML6.0.3,transformers4.57.1,Pillow13.0.0. Thiếu timm,einops,detectron2,fvcore,tensorboard,pytest,kornia,deepspeed,diffdist. Vendored SAM source không chứng minh dependency chain chạy được.

Upstream pin numpy1.23.5,Pillow9.4.0,transformers4.19.2 khác local; repo nhóm yêu cầu Pillow<12 nhưng local13. Không cài/downgrade môi trường hoặc tải weights trong audit. Không báo A/C ready-to-run khi contract/weights còn thiếu.

## 9. Nguồn và giới hạn diễn giải

[Paper SAGE arXiv2503.01210v2](https://arxiv.org/html/2503.01210v2) mô tả SAM-guided main network, Student distillation và alternating optimization. [README chính thức](https://github.com/RollingPlain/SAGE_IVIF) yêu cầu SAM/X-Decoder để train. CVF HTML/PDF trả403 khi tra cứu, dùng bản arXiv tác giả để đối chiếu ý tưởng. Công thức thực thi ở audit dựa trên commit source local, không lấy prose paper thay cho code.

Không khẳng định sửa những lỗi trên sẽ tái lập số paper. B3 report là tài liệu nhóm đã có, không phải run mới trong lần này. [Candidate B](BASELINE_PROPOSAL.md) không phải official SAGE loss và chưa có căn cứ novelty. [Quyết định cần review](CONFIG_DECISIONS.md) chưa phải approval.

## 10. SHA256 source

Bảng hash được bổ sung từ probe local; dùng kiểm tra source drift, không thay training provenance.

| File | SHA256 |
|---|---|
| SAGE/train.py | 32c6b12c0504d0c5d8b01a1211afecce490e7bdb4f943ab995032456dd52ffd1 |
| SAGE/test.py | 853226301a60b946bce35942b9b866a1c4315922b7c692c84483955f404a13b3 |
| SAGE/utils.py | bb0f7cc7c6f9d3c8781727e1ee10c4f2571086806ea6606ed94519295185f52b |
| SAGE/model_sub/model.py | e9120042916db89a2056a13d1b15552d57390bc1052508af1cbed096b4dcab4d |
| SAGE/model_sub/losses.py | 6ecd29b296a8e467369018c942558f424a35d4f858fd70ff7a92b5e1a75847b1 |
| SAGE/model_sub/common.py | 905f10ea510442abccd6cb22c08b92c1c5af27a15e26b14673f067f3e21f24a9 |
| SAGE/model_main/model.py | 954f8758481287a4c37783ec1b9fa013930d8718885058724b477ac561542370 |
| SAGE/model_main/common.py | 1486d1306bf8a3a9ae10e034b736e026fd7063c51d2d97bf4106de335fe5ae1a |
| SAGE/model_main/attention.py | 0bc8db739e82ec1ab80311f69f1e7911660b2d3262fe64eb9c87b854309b3db0 |
| SAGE/model_main/segment_loss.py | 038304cb261ab5c92a507104a5b9bfe49fd69a69f383c682dab1fd9aff9ec01e |
| SAGE/dataset/dataset_teacher_FMB.py | b0b1822fff56dd5dfc55289c4adea6260be9e54f8d29a82771a7dac616c01a66 |
| SAGE/dataset/dataset_test.py | cc299bd49bc44b6574c26943bc164e7279b05f8113effaad24bcefe5ae518398 |
| SAGE/xdecoder/infer_semseg.py | 9cab3f151e61deb85b001852b23954405e179522f74e7341579918adc5e60d08 |
| B3/src/segmentation/README.md | e105affab57961aff721a71a891528c5e65291d842c3ea645cb844bab68e8efc |
| B3/src/segmentation/chung.py | 945d6412f0370cab79ed52764b39cb56d4d40ca12ebfcc90a8ea04753b6c8413 |
| B3/src/segmentation/xuat_anh.py | da20ea2a75c316d898064196cf221e1fc436e9982532e8114704e66ec0e822f3 |
| B3/src/segmentation/train_seg.py | f060ba30b6920423d5a7c91cd467912c5b433452d32e0b98ccda53cf25764b63 |
| B3/src/segmentation/eval_seg.py | f37605e99de8b86b2d6ff85d52480743fdc184609cae958f07a9a45bd77abec0 |
| B3/src/segmentation/tong_hop.py | f482f49ccd25017b2663fbc3fb2c7abbb3a1a6f90e010ec4ef91ddcd1a9ceac4 |
| B3/src/segmentation/so_sanh_b3.py | 2825c06062a700aea84cb2088fffe4c3f9db8be03116edc85dc7cfe310f7ec2d |
| B3/reports/experiments/b3_fmb_seg/README.md | be76abc084b76814446a40da137616d3145474e547edcb70ba893ca726ad5f04 |
| B3/reports/experiments/b3_fmb_seg/artifact_manifest.json | b7bbe923ec4a53eae66b84928d53145d4e29d4156619d0ee34699736854228e7 |

## 11. Kiểm tra bàn giao tài liệu

- PASS: đủ bốn file UTF-8, liên kết nội bộ tồn tại, YAML candidate parse được, loss weights/LR/epochs/seed và số updates nhất quán.
- PASS: repository validator,31directories/26required files/3YAML. Đây là structural check, không phải training hoặc scientific validation.
- PASS: hash toàn bộ279tracked files SAGE,201tracked files B3 và131tracked files branch cá nhân trước/sau audit giữ nguyên; checkpoint và manifest nằm trong tập đã đối chiếu.
- Git chỉ có bốn file mới docs/locbe_finetune; không stage/commit/push. SAGE/B3 references sạch.
- NOT_RUN: model forward,backward,dry-run training,full training và evaluation; FMB Test không được mở. Pytest không có sẵn và không cần được cài cho lần thay đổi chỉ tài liệu này.

Lệnh tái kiểm chứng checkpoint sau đây chỉ load CPU, không train; chạy từ thư mục SAGE_IVIF bằng Python hiện có:

~~~python
import hashlib, torch
from model_sub.model import Network
p = "checkpoints/checkpoints.pt"
s = torch.load(p, map_location="cpu", weights_only=True)
m = Network()
r = m.load_state_dict(s, strict=True)
assert not r.missing_keys and not r.unexpected_keys
assert all(torch.equal(v, s[k]) for k, v in m.state_dict().items())
assert all(torch.isfinite(v).all() for v in s.values() if v.is_floating_point())
print(sum(v.numel() for v in m.parameters()), len(s))
print(hashlib.sha256(open(p, "rb").read()).hexdigest())
~~~
