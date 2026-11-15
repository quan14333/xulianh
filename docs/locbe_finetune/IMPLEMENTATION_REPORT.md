# Implementation report — SAGE Student fine-tuning

Ngày bàn giao: **15/11/2026**, Asia/Bangkok. Branch **feature/locbe-sage-finetune-1911**, HEAD **c4a6de4f77c5cbaf9790ef566a1565f8bbc4b16a**. Workspace D:\SAGE_Research\xulianh-finetune.

**Implementation + smoke PASS. Objective vẫn PROVISIONAL; KHÔNG full training.** Đây là baseline adaptation của nhóm, không phải training protocol/loss chính thức SAGE. Checkpoint smoke chỉ chứng minh pipeline chạy, không chứng minh chất lượng fusion hay mIoU tăng.

## 1. Kết quả đã thực thi

| Kiểm tra | Trạng thái | Bằng chứng |
|---|---|---|
| Unit tests | PASS,34 tests | Bốn test modules mới,unittest standard library; CPU synthetic fixtures |
| CPU pipeline / resume | PASS | Uneven batch2+1; state optimizer/scheduler/RNG và partial-epoch resume khớp next updates liên tục |
| Official pretrained load | PASS | strict=True,135617parameters,121state entries,từngtensor và zero-step reference equality |
| Input inventory | PASS |1045Train+175Dev,2440VIS/IRpaths,100%decode/MD5/SHA256/shape;không Label |
| CUDA deterministic FP32 | PASS |RTX5060Laptop,2microbatches/4Train images,1optimizerupdate,noAMP/noTF32 |
| Forward/backward/gradients | PASS |Tất cả trainable tensors có finite gradients,globalnorm>0,reference không gradient |
| Frozen state | PASS |Pretrained checkpoint,reference,BN running buffers giữ nguyên |
| Save/reload | PASS |best.pt và last.pt;last strict-reload bằng state sau update |
| Full-training guard | PASS |PROVISIONAL bị chặn trước data/model/update;không tạo approval giả |
| 5-image inference | PASS |Dev00052–00056,full600×800,RGBuint8PNG |
| B3/color parity | PASS |Independent official Network + B3 formula,5/5 maxpixel delta0;PillowRGB khớp BGR đảo channel |
| Static/structural checks | PASS |31Python files AST;repo validator31dirs/26files/3YAML;links/config consistency |

Log unit tests: unit_tests.log (34tests,không phải full legacy pytest suite). Pytest không cài; không cài/downgrade thư viện trong lần này. Fixtures có tên Test-layout chỉ là dữ liệu tổng hợp trong TEMP, không phải FMB Test. GPU/PNG evidence là checkpoint và ảnh FMB thật.

## 2. Artifact thật và thời gian

Artifact root ngoài Git/worktrees/dataset:

~~~text
D:\SAGE_Research\locbe-finetune-artifacts\implementation_20261115_8e42b9c1
~~~

| Phép đo | Giá trị |
|---|---:|
| Smoke compute window (2forward/backward microbatch,reference,gradient checks,1update) | 0.817396 s |
| Smoke wall gồm preflight1220cặp và validation2Dev | 50.057382 s |
| Peak CUDA allocated | 542424064 bytes (0.505172 GiB) |
| Peak CUDA reserved | 1136656384 bytes (1.058594 GiB) |
| CUDA free lúc kiểm guard cuối | 6212812800 bytes |
| Tổng5CNN forward inference,synchronize trước/sau | 0.407451 s |
| Trung bình forward/ảnh ở5ảnh này | 0.081490 s |

Đây là đo smoke ngắn,không phải benchmark throughput dài hoặc ETA20epochs. CUDA allocated/reserved không đồng nghĩa toàn VRAM nvidia-smi. Export forward timing không gồm model load/PNG IO.

Checkpoint:

~~~text
gpu_smoke/last.pt
SHA256: a8b668a08cb6c20f821e9eb876399138c2bb5b70eddd37138064e2ab6bba4475
gpu_smoke/best.pt
SHA256: a8b668a08cb6c20f821e9eb876399138c2bb5b70eddd37138064e2ab6bba4475
original SAGE checkpoints.pt (không thay đổi):
ecfde477ca4cd65401f207d5dc91ce6c475a49891d768dbdb7a66366255ee836
config canonical SHA256: f116f29ff441373d13c4c9d18551b3f92818f02c5f91f3d9c11113dc19ffad7c
~~~

last.pt/best.pt đều có dry_run_only=true. best là auxiliary Dev reconstruction loss, KHÔNG chọn theo Person/mIoU; checkpoint chính vẫn last theo proposal. Smoke validation chỉ2Dev đầu,không được trình bày như toàn175Dev. Hai microbatch có Train IDs 00443|00146; 00871|00548,norm trước/sau clipping 0.6033852310415084;clip1.0 không bị kích hoạt giảm norm ở run này.

5output ở export5/sage_finetuned/train/<image_id>.png;source_split=train vì Dev lấy từ officialTrain. output_manifest.csv giữ manifest_split=dev riêng:

| ID | H×W×C | Raw Y min/max | PNG SHA256 |
|---|---|---|---|
| 00052 | 600×800×3 | 0.100701/0.986173 | 1474793d002f9229dadccf0043fb61654696dd1f4db7db4e1bda16c9d5348544 |
| 00053 | 600×800×3 | 0.050385/0.987219 | d9761753dab6f7e5564a9f6f135723f54de65dd056c0926fb49bfe0f48923f64 |
| 00054 | 600×800×3 | 0.075267/0.987567 | d8a5620c832a57836ae51e668dd424059accdd5f77db1410e1e302df49da6cf7 |
| 00055 | 600×800×3 | 0.093983/0.986073 | 45d69b8914e197787bd9de723d115c95d870c5e94a4edd1d684dacf6efb5fe06 |
| 00056 | 600×800×3 | 0.056558/0.985769 | eef0f7dfbe04d8eb95f17902dcbb11393c7a29cb20af859fa7298b8fc145507b |

Chroma VIS giữ nguyên khi ghép YCrCb trước chuyển màu; không yêu cầu RGB→YCrCb round-trip khôi phục chroma chính xác khi có gamut clamp. B3 contract khác official float-RGB save_image:không trộn cache. Pixel verification được tái tính từ official Network,không import/chạy B3 exporter vốn gộp Test.

Evidence đi kèm: gpu_smoke/{config.json,environment.json,input_audit.csv,initialization.json,batch_losses.csv,history.csv,run_result.json,best.pt,last.pt};export5/{output_manifest.csv,export_result.json,environment.json,PNGs};b3_pixel_verification.json;full_training_guard.json;unit_tests.log;unit_tests_result.json;smoke_file_access.json;export_file_access.json;handoff_checks.json. Các runner kiểm chứng cũng được lưu ở artifact root.

## 3. Objective và cấu hình triển khai

V,I float32/255 theo OpenCV BGR→YCrCb/GRAY,model(VIS_Y,IR). F và frozen-reference F0 là rawTanh,không clamp trước loss.

~~~text
E(A,B)=mean((A-B)^2)
G(X)=abs(Sobel_x*X)+abs(Sobel_y*X),unnormalized3x3,zero-padding1
L_int=.5E(F,V)+.5E(F,I)
L_grad=mean(abs(G(F)-max(G(V),G(I))))
L_anchor=E(F,stopgrad(F0))
L_total=1.0L_int+0.2L_grad+1.0L_anchor
~~~

Công thức đúng [proposal](BASELINE_PROPOSAL.md),không import official Losses/common có SAM. HaiStudent instances đều load cùng original state;reference eval/frozen/no_grad. BN running stats frozen,affine vẫn học. Freeze/unfreeze theo parameterpatterns trong YAML,unknown patterns fail.

Recipe được implement: AdamLR1e-5,betas(.9,.999),eps1e-8,wd0;cosine eta_min1e-6;20epochs đề xuất,seed2,crop192×256,pairedflip.5,micro2×accum2,FP32,noTF32,workers0. Accumulation normalize theo actual samples/window (không chia cố định2 ở window thiếu). Scheduler theo optimizer updates:5240updates nếu full20epochs1045Train. Run thực tế chỉ1update.

Gradient clip norm1.0 là safeguard mới theo yêu cầu hiện tại;giá trị vẫn PROVISIONAL. CUDA cap4GiB allocated/free>=1GiB đúng YAML. Không silent CPU fallback,không skip loss/OOM hoặc tự đổi recipe. Deterministic CUDA backward đã chạy thật cho hai microbatch này.

## 4. CLI để tái kiểm chứng

Từ worktree finetune;output phải là thư mục MỚI. Những lệnh này sẽ chỉ smoke/export,cần thay placeholder nếu chạy lại:

~~~powershell
$py = 'D:\SAGE_Research\xulianh\.venv-sage\Scripts\python.exe'
& $py -B -m unittest tests.test_sage_finetune_dataset tests.test_sage_finetune_losses tests.test_sage_finetune_checkpoint tests.test_sage_finetune_inference -v
& $py -B scripts/run_sage_finetune.py --config configs/methods/sage_student_finetune.yaml --mode smoke --smoke-batches 2 --device cuda --output '<NEW>\gpu_smoke'
& $py -B scripts/export_sage_finetuned.py --config configs/methods/sage_student_finetune.yaml --checkpoint '<NEW>\gpu_smoke\last.pt' --split dev --limit 5 --device cuda --output '<NEW>\export5'
~~~

Lần kiểm chứng thật gọi chính các main/CLI arguments trên dưới Python file-open audit hook:chặn FMB Test/Label và mọi write vào FMB. Smoke ghi2454open events,2440unique VIS/IRpaths;export5ghi10unique inputpaths;không Test/Label/datasetwrites. Independent pixel verifier chỉ đọc5VIS/IRTrain paths đã xác minh.

CPU smoke dùng --device cpu;unit pipeline CPU dùng fixtures tổng hợp,không lấy số đó làm performance SAGE/FMB.

Resume: --resume <OLD>/last.pt vào output mới,cùngconfig/source/input/device/mode,giữ companion best.pt. Model/optimizer/scheduler/Python/numpy/torch/CUDARNG,epoch/cursor/history đều được lưu. Checkpoint tại optimizer boundary;không resume gradient accumulation đang dở. CPU tests chứng minh next-update và partial-epoch2updates khớp uninterrupted. Chưa kiểm chứng full20epochs hoặc long CUDA interrupted replay.

Full mode=train vẫn bị chặn. Approval record phải có allow_full_training=true,approved_by,evidence,config_sha256 khớp recipe;không có record như vậy được tạo ở lần này. Smoke checkpoint không được promote thành full run. PROVISIONAL microbatch<=2,smoke<=2batches,export<=5images.

## 5. Files tạo mới

| File | Vai trò |
|---|---|
| src/methods/sage_finetune/__init__.py | Package PROVISIONAL |
| src/methods/sage_finetune/dataset.py | Manifest, paired crop/flip, hash/shape guards; không Label/Test |
| src/methods/sage_finetune/model.py | Import official Student, strict pretrained load, freeze/BN/output checks |
| src/methods/sage_finetune/losses.py | Objective candidate thuần torch, từng loss component |
| src/methods/sage_finetune/train.py | Smoke/full guard, accumulation, optimizer/scheduler, validation, resume |
| src/methods/sage_finetune/infer.py | B3-compatible PNG, QA, output manifest/hash |
| src/methods/sage_finetune/checkpoint.py | Safe strict state loading, optimizer/scheduler/RNG, provenance |
| src/methods/sage_finetune/config.py | Schema validation, canonical hash, approval/scope guards |
| src/methods/sage_finetune/README.md | Lệnh chạy, formulas, B3 contract và limitations |
| configs/methods/sage_student_finetune.yaml | PROVISIONAL, đúng recipe và original identities |
| scripts/run_sage_finetune.py | CLI train/smoke |
| scripts/export_sage_finetuned.py | CLI export |
| tests/sage_finetune_helpers.py | Fixtures tổng hợp, không experimental results |
| tests/test_sage_finetune_dataset.py | IDs/layout/preprocessing/pairing/hash/leakage guards |
| tests/test_sage_finetune_losses.py | Independent Sobel arithmetic, gradients, weights/borders |
| tests/test_sage_finetune_checkpoint.py | Strict load, freezing, guards, RNG and partial resume parity |
| tests/test_sage_finetune_inference.py | PNG/color/clamp/rounding + tiny CPU pipeline |
| docs/locbe_finetune/IMPLEMENTATION_REPORT.md | Biên bản bàn giao này |

README: [hướng dẫn module](../../src/methods/sage_finetune/README.md).
YAML: [candidate config](../../configs/methods/sage_student_finetune.yaml).
Không sửa bốn tài liệu audit/proposal trước đó;report mới này ghi tiến độ ngày15/11.

## 6. Git và tính bất biến

131trackedfiles worktree cá nhân,279trackedfiles SAGE,201trackedfiles B3 giữ nguyên SHA256;bốn tài liệu cũ giữ nguyên. SAGE/B3 sạch. Manifest gốc/checkpoint không thay đổi. Code mới/config/tests/report vẫn untracked,chưa stage/commit/push. Artifact lớn nằm ngoài Git.

Không đọc FMB Test hoặc GTLabel trong bước này;không train SegFormer,không sửa B3,không tải dataset/weights hoặc giả lập experimental results.

## 7. Cần review và giới hạn còn lại

- Quân chưa phê duyệt objective/hệ số1/.2/1,anchor,BN policy,LR20epochs,lastselection,clip1.0 và resource caps. Giữ PROVISIONAL,không tự tạo approval.
- Final1220/Test chưa được bật trong implementation;config parser chỉ nhận Development1045/175. [B3 integration plan](B3_INTEGRATION_PLAN.md) là kế hoạch sau freeze/review,không phải những run đã thực hiện.
- Alias output sage_finetuned ứng với candidate sage_ft_selfsup_v1;không đổi loss. B3 đọc PNGvariant qua --bien_the sage_finetuned và SAGE_ANH root sau khi có đủ inputs;5DevPNG không đủ để train B3.
- Pretraining Student training-ID list chưa có,không gọi Dev độc lập với fusionpretraining.
- Chưa đo mIoU hoặc SCD ở lần implementation; loss smoke/5ảnh không chứng minh semantic improvement hoặc novelty.
- Python3.11+ được dùng (hashlib.file_digest);torch2.11.0+cu130,torchvision0.26.0+cu130,cv2 5.0.0.93,numpy2.4.6. Pillow13 nằm ngoài repo constraint<12;không sửa môi trường. Environment JSON ghi đầy đủ versions/GPU/Gitdirty status/source hashes.
- Không chạy full legacy pytest suite vì thiếu pytest;34tests mới chạy bằng unittest. GPU dài/nhiềuepochs/multi-seed,finalB3/matchedmIoU đều NOT_RUN/NOT_MEASURED.
