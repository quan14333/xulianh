# Bàn giao Lộc bé — 19/11/2026

Cập nhật thực tế **15/11/2026, Windows local, RTX 5060 Laptop 8GB**. Phạm vi lần này: đồng nhất API fusion, bổ sung exporter sage_finetuned và kiểm chứng đầu vào B3. **Tích hợp kỹ thuật PASS; chưa đủ điều kiện đánh giá chất lượng fine-tuning hoặc mIoU B3.** Không full training trong lần này.

## 1. Git và bảo toàn công việc

- Fetch origin thành công. [PR #8](https://github.com/quan14333/xulianh/pull/8) đã merge ngày 14/11/2026 14:25:32 UTC, merge commit 8cc907b5338efe21fd26355258cc2d05411ca875.
- Branch feature/locbe-sage-finetune-1911 đã fast-forward từ c4a6de4f77c5cbaf9790ef566a1565f8bbc4b16a đến origin/main **5352eedecaa2a34e703f12e917e31f03627b32b1**. Kiểm tra ancestry, tracked/index sạch và không có collision với file untracked trước khi cập nhật.
- Snapshot 161 file sẵn có: 160 file giữ nguyên byte; .gitignore là thay đổi chủ đích từ upstream, khớp HEAD sau chuẩn hóa CRLF. Bao gồm giữ nguyên pipeline Student, config PROVISIONAL, báo cáo và artifact cũ. Không reset, clean, stash, xóa file, cherry-pick nguyên PR hoặc ghi đè công việc riêng.
- Workspace làm việc: D:\SAGE_Research\xulianh-finetune. Code mới vẫn untracked, chưa commit/push; HEAD không đại diện các file mới, cần xem inventory SHA256 và review diff trước bàn giao Git.
- Review worktree giữ detached HEAD 58a24847ac68aca6f232194cf1bc84d9fcf4be52 tại D:\SAGE_Research\xulianh-b3-review; không sửa source. Loader smoke dùng source đã merge trong worktree riêng, pin hash.
- So với review snapshot: train_seg.py trên main thêm metadata môi trường/resume checks; FMBSeg và thuật toán xuất ảnh không đổi. Không sửa bất kỳ file src/segmentation của Mân.

Bằng chứng: before_integration.json, main_integration.json, github_audit.json, HANDOFF_AUDIT.json ở artifact root mục 7.

## 2. Ba vai trò khác nhau

| Variant | Tình trạng | Ý nghĩa |
|---|---|---|
| sage | FROZEN_PRETRAINED | Student pretrained chính thức, strict=True, checkpoint gốc giữ nguyên |
| sage_finetuned | PILOT_ONLY_NOT_FULL_BASELINE | Student-only candidate của nhóm; checkpoint hiện có chỉ 2 microbatch, 4 ảnh Train, 1 optimizer update, **0 epoch hoàn tất** |
| quan_improved | WAITING_FOR_METHOD_SPEC | Interface có sẵn, chưa có thuật toán; gọi CLI trả status này, exit 2, không tạo PNG/fallback |

Quân: thiết kế phương pháp cải tiến khi có đặc tả được cung cấp. Lộc bé: engineering pipeline, adapter, tests và provenance hiện tại. Student-only candidate trong [BASELINE_PROPOSAL.md](BASELINE_PROPOSAL.md) **không phải official SAGE loss, không phải phương pháp cải tiến đã được Quân xác nhận**, chưa có claim novelty.

Đã kiểm tra problem card Quân trên origin/main và origin/feature/quan-baseline, notebook baseline cũ, PR và issue trong GitHub nhóm. Card vẫn placeholder; notebook là SAGE pretrained baseline. Chưa tìm thấy đủ hypothesis, công thức/thuật toán, config, prototype, ablation protocol trong các nguồn đó. Không suy diễn về trao đổi bên ngoài repository.

## 3. Code được bàn giao

| File | Chức năng |
|---|---|
| [fusion_interface.py](../../src/methods/fusion_interface.py) | FusionPair BGR uint8 aligned + image_id string; FusionPrediction RGB uint8 cùng H×W; protocol chung |
| [b3_adapter.py](../../src/methods/b3_adapter.py) | Factory sage/sage_finetuned/quan_improved; strict load, provenance, export PNG/CSV/JSON, giữ baseline |
| [b3_loader_adapter.py](../../src/methods/b3_loader_adapter.py) | Chạy **FMBSeg thật** từ source Mân, chỉ điều chỉnh IO trong module globals tạm thời, pin hash, resolver chỉ 5 Dev labels |
| [quan_method/](../../src/methods/quan_method/README.md) | Spec gồm 5 trường bắt buộc; trạng thái chờ; không tự tạo kiến trúc |
| [export_b3_variant.py](../../scripts/export_b3_variant.py) | CLI thống nhất cho ba variant |
| [check_b3_inputs.py](../../scripts/check_b3_inputs.py) | CLI kiểm PNG + loader B3 + normalization + raw label mapping |
| [b3_variants.yaml](../../configs/methods/b3_variants.yaml) | Registry variant, checkpoint/hash, source B3, recipe reference, quyền chỉ smoke |
| [test_b3_adapter.py](../../tests/test_b3_adapter.py) | 11 tests interface, independent pixel oracle, hash/role/limit/Test guards, actual FMBSeg, tampering |

Pipeline cũ được tái sử dụng nguyên trạng tại [sage_finetune/README.md](../../src/methods/sage_finetune/README.md): dataset, model, losses, train, infer, checkpoint, config; hai CLI train/export; 34 tests hiện có. [IMPLEMENTATION_REPORT.md](IMPLEMENTATION_REPORT.md) là báo cáo lịch sử trước main integration, giữ nguyên commit/timestamp cũ để không làm sai provenance.

Input chung: image_id 5 chữ số dạng string; VIS và IR BGR uint8 cùng H×W×3, H/W>=4. VIS BGR→YCrCb, IR BGR→GRAY bằng OpenCV; float32 /255; Network(VIS_Y,IR), eval và no_grad. Output raw Tanh kiểm finite/range/shape.

**Contract B3:** Y8=byte(round(255*clamp(F,0,1))); ghép Cr/Cb gốc VIS trước chuyển OpenCV YCrCb→BGR và ghi PNG. File giải mã bằng Pillow là RGB. Không dùng (F+1)/2 hoặc min-max normalization. Contract B3 này khác đường lưu float-RGB của upstream test.py; không trộn cache giữa hai contract. Chroma được giữ ở bước ghép, không yêu cầu RGB→YCrCb ngược lại giữ từng byte khi có gamut clipping.

Layout:
~~~text
<export-root>/sage/train/00052.png
<export-root>/sage_finetuned/train/00052.png
~~~

Mỗi lần CLI tạo một export-root mới. Ảnh Dev thuộc official Train nên folder nguồn vẫn train; CSV ghi manifest_split=dev riêng. Xuất output_manifest.csv, export_result.json, variant_registration.json, environment.json. Không sửa choices hoặc nhánh xử lý variant cũ trong xuat_anh.py; train_seg.py nhận tên variant tự do, loader tìm ANH/variant/train/id.png.

## 4. Config, checkpoint và phạm vi được phép

Registry hiện tại chỉ **TECHNICAL_SMOKE_ONLY**. PROVISIONAL giới hạn tối đa 5 ảnh/export; pilot checkpoint cần --allow-pilot-checkpoint. Không tự tạo approval record. Full Student training/export vẫn qua guard config-bound có sẵn; B3 training không được adapter tự chạy.

### Student candidate Development

[Config](../../configs/methods/sage_student_finetune.yaml): 1045 Train cập nhật trọng số, 175 Dev validation, không đọc nhãn để tạo fusion loss. Adam LR1e-5, betas(.9,.999), weight_decay0; cosine theo optimizer updates đến1e-6; target20 epochs, selection last; seed2, FP32/noTF32/deterministic; crop192×256, paired flip.5; micro2/accum2; clip1; BN running stats frozen, affine học được.

V,I là nguồn [0,1]; F=S_theta(V,I)[0], F0=stopgrad(S0(V,I)[0]) là raw Tanh:
~~~text
E(A,B) = mean((A-B)^2)
G(X) = abs(Sobel_x*X) + abs(Sobel_y*X), 3x3 unnormalized, zero-padding1
L_int = .5 E(F,V) + .5 E(F,I)
L_grad = mean(abs(G(F)-max(G(V),G(I))))
L_anchor = E(F,F0)
L_total = 1.0 L_int + 0.2 L_grad + 1.0 L_anchor
~~~

Đây là heuristic candidate của nhóm, không Teacher-distillation objective gốc. Objective/config chưa approved: **BLOCKED_FOR_APPROVAL** đối với full run. Checkpoint được registry trỏ đến pilot thật của lần trước, không có full fine-tuned baseline hoặc Dev175 quality comparison hoàn chỉnh trong bàn giao này.

### Hash khóa

~~~text
SAGE original: D:\SAGE_Research\SAGE_IVIF\checkpoints\checkpoints.pt
SHA256 ecfde477ca4cd65401f207d5dc91ce6c475a49891d768dbdb7a66366255ee836
SAGE Student source SHA256
e9120042916db89a2056a13d1b15552d57390bc1052508af1cbed096b4dcab4d
Upstream commit 377fdbef6b8b6ccbd1139358ad66bb41ea084dc1

Pilot: D:\SAGE_Research\locbe-finetune-artifacts\pilot_20261115_c681ea42\gpu_pilot\last.pt
SHA256 a8b668a08cb6c20f821e9eb876399138c2bb5b70eddd37138064e2ab6bba4475
best.pt có cùng SHA; dry_run_only=true; dùng last, không chọn Person/mIoU.

Student canonical config fingerprint
f116f29ff441373d13c4c9d18551b3f92818f02c5f91f3d9c11113dc19ffad7c
b3_variants.yaml raw SHA256
d7cf30f41c71b57c86d2b1ff303d557900c2576c286398e4140b26400c77dd30
Locked B3 config raw SHA256
519ad1bdb94ba47ad552015364276e760ddde3dff6fdb6a792a8b8d3066515fa

Train manifest SHA256
86158cdcd02ad08c1143135f35b06c1803489e842c4eda0a79c90198590b3071
Dev manifest SHA256
4720243878c093a8717f5c09a907b164618c97e8fb68ffb8dff2db4e93ba6f15
~~~

Cả 1220 IDs unique, Train/Dev disjoint, giữ số0 đầu. SAGE pretrained chưa có training-ID list để chứng minh Dev độc lập với fusion pretraining.

## 5. Fair B3 evaluation: đăng ký reference, chưa cấp full run

Source of truth: [B3 locked config](../../configs/experiments/b3_fmb_seg.yaml), [README Mân](../../src/segmentation/README.md), [báo cáo B3](../../reports/experiments/b3_fmb_seg/README.md). Không sửa config gốc.

| Thuộc tính B3 final | Reference đã khóa của Mân |
|---|---|
| Dữ liệu | 1220 official Train = Train1045 + Dev175; không dùng Test chọn checkpoint |
| Kiến trúc/init | SegFormer-B3; nvidia/mit-b3 encoder ImageNet, head15 lớp khởi tạo theo seed; cần kiểm từng tensor giữa methods |
| Recipe |100epochs,batch8,accum1,drop_last;152updates/epoch,total15200 |
| Optimizer |AdamW encoderLR6e-5,decode-headLR×10,wd.01 |
| Scheduler |linear warmup1500updates,poly power1 |
| Input/augmentation |paired image/mask randomscale.5–2,crop512×512,hflip;ImageNet mean/std |
| Precision/seeds |FP16 autocast+GradScaler;seeds0/1/2, DataLoader seeded, reference không bật deterministic |
| Checkpoint |last.pt sau100epochs; không best theo Dev/Person/Test |
| Labels |15IDs0..14,ID0 được train;ignore255 cho padding;personID8 |
| Eval |full600×800,single-scale,no-flip;logits bilinear về H×W;pooled15×15 confusion |

Primary **mIoU_co_mat**: loại GT row0, dự đoán0 trên pixel GT1..14 vẫn sai; mean IoU trên các lớp có GT trong tập đang đánh giá. Không hardcode13 lớp cho Dev; số lớp present phụ thuộc tập. Báo thêm mIoU_14, mIoU_15_nan0, mIoU_15_mrfs theo code Mân khi thực sự có predictions. Không thay bằng B0 mIoU15 cũ.

Development Student1045/175 và final Student1220 là **hai protocol khác nhau**. Muốn đối chiếu final B3 phải khóa recipe Student sau review, đăng ký mới final1220 và selection không dùng Test, tạo cache đủ1220, giữ provenance từng checkpoint/epoch. Không đổi config fingerprint của pilot rồi giả định cùng run.

Khi đủ phê duyệt và tài nguyên: xác minh B3 initialization policy/từng tensor, full input inventory; chạy pilot VRAM/gradient trước; đăng ký riêng mọi đổi batch/accumulation/precision/scheduler do RTX5060 8GB. Cùng effective batch không tự bảo đảm tương đương optimizer trajectory/scheduler. Không lấy thời gian T4 trong report làm ETA local.

Không chạy eval_seg.py trong lần này vì script nhắm Test. Không có checkpoint SegFormer-B3 FMB phù hợp tại các checkpoint roots đã audit; artifact Mân công bố nằm kho Kaggle riêng tư, chưa được cấp truy cập/tải. B3 inference: **NOT_AVAILABLE / NOT_AUTHORIZED**. Không dùng ADE20K/ImageNet head hoặc checkpoint B0 để thay thế.

## 6. Kiểm thử thực tế và bảng kết quả

| Hạng mục | Trạng thái / bằng chứng |
|---|---|
| Existing34 + adapter11 unit tests |**PASS45**, unittest; runtime suite7.807s, subprocess wall12.213s |
| Official/pilot checkpoint |**PASS**, SHA256 và strict load toàn bộ state; file không thay đổi |
| SAGE pretrained CUDA |**PASS**,5Dev IDs00052–00056; tổng inference window0.779779s |
| SAGE fine-tuned pilot CUDA |**PASS**,cùng5Dev; tổng0.201469s, PILOT_ONLY/0epoch |
| PNG/shape/hash |**PASS10/10**,600×800×3,uint8,đúngID/split |
| Independent color/pixel oracle |**PASS10/10**,max_abs_pixel_error=0; official Network nạp trực tiếp, công thức B3 độc lập, PillowRGB/OpenCVBGR |
| Actual B3 FMBSeg |**PASS**,5ảnh/variant,tensor3×600×800FP32,mask600×800int64 IDs0..14,ImageNet normalization khớp tuyệt đối |
| Quân placeholder guard |**PASS**,WAITING_FOR_METHOD_SPEC,exit2,không output/fallback |
| B3 model inference / mIoU |**NOT_AVAILABLE**,không checkpoint FMB được cấp truy cập |
| Full Student/B3 training |**NOT_RUN**,không được trình bày pilot thành full baseline |

Timing là smoke ngắn có first-call overhead; adapter window gồm chuyển input lên GPU + forward/validation, có synchronize, không gồm load/checkpoint/PNG IO. Không dùng chênh0.78/0.20s để kết luận mô hình fine-tuned nhanh hơn.

Unit tests sử dụng synthetic fixtures để kiểm logic; bảng CUDA/PNG/loader sử dụng FMB Dev thật. Có đọc **đúng5Dev GT masks chỉ trong loader QA**, không đưa vào fusion loss hoặc cập nhật trọng số. Audit hook Python open và wrapper cv2.imread chặn Test, ghi FMB, và đọc label ngoài stage loader; data_access_audit.json lưu events. Verification độc lập chỉ đọc VIS/IR Train. Không chạy full45tests thành FMB Test evaluation.

## 7. Artifacts, môi trường và reproduce Windows

Artifact root mới, không chèn checkpoint/PNG lớn vào Git:
~~~text
D:\SAGE_Research\locbe-finetune-artifacts\handoff_20261115_3af49d82
~~~

- exports/sage và exports/sage_finetuned: mỗi root có PNG subtree, output_manifest.csv với input/output/checkpoint/config SHA256 và timing, variant_registration.json, environment.json, export_result.json.
- sage_loader.json, sage_finetuned_loader.json; *_loader_manifests/dev.csv chỉ là bản sao5rows bên ngoài dataset/manifests gốc.
- unit_tests.log, unit_tests_result.json, integration_smoke.log, pixel_verification.json, integration_summary.csv.
- github_audit.json, availability_audit.json, before_integration.json, main_integration.json, data_access_audit.json, metric_contract_checks.json, validation.log, HANDOFF_AUDIT.json.
- integration_smoke.py, verify_pixels.py và runner kiểm tra được lưu để audit/reproduce. Không dùng chúng để thay đổi source gốc.

Environment: Python3.11.9,torch2.11.0+cu130,torchvision0.26.0+cu130,numpy2.4.6,OpenCV5.0.0.93,PyYAML6.0.3,Pillow13.0.0; CUDA RTX5060 capability12.0,8546484224bytesVRAM. Không cài/downgrade package; pytest chưa có, dùng unittest.

~~~powershell
Set-Location 'D:\SAGE_Research\xulianh-finetune'
$py = 'D:\SAGE_Research\xulianh\.venv-sage\Scripts\python.exe'
$reg = 'configs/methods/b3_variants.yaml'

& $py -X utf8 -B -m unittest tests.test_sage_finetune_dataset tests.test_sage_finetune_losses tests.test_sage_finetune_checkpoint tests.test_sage_finetune_inference tests.test_b3_adapter -v

# Thay NEW_* bằng thư mục/file CHƯA tồn tại; không ghi đè output cũ.
$pre = 'D:\SAGE_Research\locbe-finetune-artifacts\NEW_PRETRAINED'
$ft = 'D:\SAGE_Research\locbe-finetune-artifacts\NEW_FINETUNED'
& $py -X utf8 -B scripts/export_b3_variant.py --registry $reg --variant sage --split dev --limit 5 --device cuda --output $pre
& $py -X utf8 -B scripts/export_b3_variant.py --registry $reg --variant sage_finetuned --allow-pilot-checkpoint --split dev --limit 5 --device cuda --output $ft
& $py -X utf8 -B scripts/check_b3_inputs.py --registry $reg --variant sage --export-root $pre --output 'D:\SAGE_Research\locbe-finetune-artifacts\NEW_pre_loader.json'
& $py -X utf8 -B scripts/check_b3_inputs.py --registry $reg --variant sage_finetuned --export-root $ft --output 'D:\SAGE_Research\locbe-finetune-artifacts\NEW_ft_loader.json'

# Dự kiến exit2, WAITING_FOR_METHOD_SPEC và không tạo output.
& $py -X utf8 -B scripts/export_b3_variant.py --registry $reg --variant quan_improved --output 'D:\SAGE_Research\locbe-finetune-artifacts\NEW_QUAN'
& $py -X utf8 -B scripts/validate_repo.py
~~~

Muốn chọn checkpoint khác: --checkpoint <path> --checkpoint-sha256 <sha>; phải là checkpoint pipeline có provenance/config đúng, strict load; cập nhật registry qua review trước khi dùng check_b3_inputs.py vì loader pin checkpoint registry. Không gắn tên sage cho fine-tuned weights. Không dùng --allow-pilot-checkpoint để tạo đánh giá full: pilot/partial vẫn bị cap5.

B3 loader bridge chỉ kiểm input, không training. Root $pre/$ft tương ứng SAGE_ANH cho variant đó; không đặt SAGE_ANH vào folder train bên trong. Resolver được thay tạm trong module để tránh chung._tim_fmb() quét cả Test; source Mân không sửa. Chưa cung cấp lệnh B3 full run cho cache5ảnh; cần đủ cache, protocol và approval trước.

## 8. Nghiệm thu trước 19/11

- [x] Fetch, PR8 audit, FF main an toàn, giữ worktree review và source/checkpoint/manifest gốc.
- [x] API thống nhất và adapter sage_finetuned; registry với baseline/pilot/method chờ tách biệt.
- [x] 45unit tests,5Dev mỗi SAGE variant,10PNG,exactpixel parity,actual B3 loader và provenance.
- [x] Lệnh Windows,logs,config,hash và giới hạn được bàn giao; chưa commit/push.
- [ ] Quân cung cấp/duyệt hypothesis,algorithm,config,prototype,ablation. Sau đó mới implement và test quan_improved.
- [ ] Quân phê duyệt exact objective/recipe bằng approval record khớp fingerprint; chạy Student full và báo số epoch thật.
- [ ] Mân/nhóm cấp checkpoint B3 FMB + hash/config/access; chạy segmentation inference pilot trên Dev sau audit.
- [ ] Khóa final1220 Student/B3 protocol, nguồn pretraining và hardware deltas; chỉ chạy các full runs được duyệt.
- [ ] Review diff + artifact inventory và quyết định commit/push riêng. Lần này không push.

Chưa có số fusion-quality hoặc mIoU mới để so pretrained/fine-tuned/Quân; không kết luận cải thiện semantic, không khẳng định novelty. Không dùng Test, không thay checkpoint SAGE, không chỉnh manifest và không sửa các biến thể cũ của Mân.
