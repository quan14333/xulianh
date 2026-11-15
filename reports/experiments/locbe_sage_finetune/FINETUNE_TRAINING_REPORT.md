# SAGE Student fine-tuning — pilot và trạng thái phê duyệt

Ngày15/11/2026 (Asia/Bangkok). **BLOCKED_FOR_APPROVAL** cho full training và so sánh175Dev. Pre-flight/pilot đã PASS. Không có full fine-tuned baseline được tạo trong lần này.

## 1. Quyết định thực thi

Config hiện tại vẫn status=PROVISIONAL,approval_record=null. CONFIG_DECISIONS và BASELINE_PROPOSAL ghi các giá trị mới chưa được Quân phê duyệt. Guard authorize(config,train) thực sự từ chối; không tự đổi trạng thái hoặc tạo approval giả.

Canonical recipe SHA256:

~~~text
f116f29ff441373d13c4c9d18551b3f92818f02c5f91f3d9c11113dc19ffad7c
~~~

Theo điều kiện người dùng, chỉ chạy unit tests/pre-flight/pilot. Không triển khai full20epochs hoặc xuất175Dev dưới tên “fine-tuned baseline” từ checkpoint pilot.

| Bước | Trạng thái |
|---|---|
| Git/GPU/packages/manifest/checkpoint | PASS runtime; package constraint có ghi chú bên dưới |
| Unit tests | PASS,34 tests mới bằng unittest |
|1045Train/175Dev | PASS,unique/disjoint,manifestSHA và100%decode/hash/shape |
| GPU2microbatch forward/backward | PASS,4Train images,1optimizer update |
| Strict pretrained/checkpoint reload | PASS |
| CUDA optimizer/scheduler/RNG reload | PASS,210state tensors,140moment tensors trênCUDA |
| Full training | BLOCKED_FOR_APPROVAL |
|175Dev comparison/figures | NOT_RUN — chưa có approved trained baseline |
|Q_ABF | NOT_AVAILABLE — chưa xác minh implementation cho run này |
|mIoU/B3evaluation | NOT_MEASURED |

## 2. Epochs và checkpoint thật

**0 full epochs hoàn tất trong20epochs đề xuất.** Pilot đi qua4/1045Train images ở epoch index1,2microbatch×2ảnh,accumulation2 →1optimizerupdate. Progress lưu epoch=0,sample_cursor=4,global_step=1. History có epoch_complete=False; không được đọc row epoch1 như đã học xong một epoch.

Pretrained SHA256:

~~~text
ecfde477ca4cd65401f207d5dc91ce6c475a49891d768dbdb7a66366255ee836
~~~

Pilot artifacts (ngoài Git/worktrees/FMB):

~~~text
D:\SAGE_Research\locbe-finetune-artifacts\pilot_20261115_c681ea42\gpu_pilot\last.pt
D:\SAGE_Research\locbe-finetune-artifacts\pilot_20261115_c681ea42\gpu_pilot\best.pt
~~~

last.pt SHA256: a8b668a08cb6c20f821e9eb876399138c2bb5b70eddd37138064e2ab6bba4475

best.pt SHA256: a8b668a08cb6c20f821e9eb876399138c2bb5b70eddd37138064e2ab6bba4475

Cả hai dry_run_only=true,không phải checkpoint20epoch. Last là lựa chọn chính;best chỉ diagnostic minimum Dev reconstruction loss,không mIoU/Person. Checkpoint pretrained gốc giữ nguyên hash.70parameter tensors Student đã thay đổi;reference và BN running buffers giữ nguyên. Không gọi đây là evidence tăng chất lượng.

Reload trên CUDA so từng tensor model/Adam states,LR/cosine scheduler và Python/numpy/torch/CUDA RNG:đều bằng checkpoint. Không thêm optimizerstep khi kiểm reload. Chi tiết:optimizer_scheduler_reload_gpu.json ở artifact root.

## 3. Loss và recipe

Đây là **candidate Student-only của nhóm**,không phải official SAGE loss/protocol. VIS BGR→YCrCb lấyY,IR BGR→GRAY,FP32/255. F và frozen-reference F0 là rawTanh output.

~~~text
E(A,B)=mean((A-B)^2)
G(X)=abs(Sobel_x*X)+abs(Sobel_y*X),unnormalized3x3,zero-padding1
L_int=.5E(F,V)+.5E(F,I)
L_grad=mean(abs(G(F)-max(G(V),G(I))))
L_anchor=E(F,stopgrad(F0))
L_total=L_int+.2L_grad+L_anchor
~~~

Intensity/edges nhằm bảo toàn ảnh nguồn;anchor nhằm hạn chế drift khỏi pretrained. Hệ số là heuristic chưa approved;không bảo đảm semantic preservation. Không TeacherSAGE/SAM/XDecoder/GTLabel trong objective này.

Recipe không đổi:AdamLR1e-5,betas(.9,.999),eps1e-8,wd0,cosine eta_min1e-6,seed2,crop192×256 H×W,pairedhflip.5,microbatch2,accumulation2,FP32/noTF32,deterministic=True,workers0,clipnorm1.0.20epochs là target đề xuất,chưa chạy. BN running stats frozen,affine vẫn trainable.

| Pilot đo thật | Giá trị |
|---|---:|
| Train mean total loss | 0.03432092256844044 |
| Mean intensity | 0.02886886615306139 |
| Mean gradient | 0.027260283939540386 |
| Mean anchor | 0.0 |
| Global gradient norm trước/sau clip | 0.6033852310415084 |
| Pilot compute window | 0.725299 s |
| Pilot wall,gồm QA1220cặp và validation2Dev | 38.703068 s |
| Peak allocated | 542424064 bytes (0.505172 GiB) |
| Peak reserved | 1136656384 bytes (1.058594 GiB) |

Không OOM,không thay batch/crop/LR/loss. Anchor=0 trong hai microbatch đầu vì Student chưa được optimizerstep giữa hai microbatch;điều này không có nghĩa anchor bị tắt. Validation chỉ2Dev đầu để chẩn đoán finite objective,không phải toàn175Dev. Thời gian pilot không phải ETA/fullrun benchmark.

## 4. Pretrained khác fine-tuned như thế nào?

Chỉ xác nhận được weights sau1pilotupdate khác pretrained,70parameter tensors thay đổi và saved/reloaded state bằng nhau. Chưa có trained baseline đủ20epochs,không có paired175Dev metrics. Vì vậy chưa xác định trường hợp cải thiện/suy giảm;không lấy train/dev pilotloss làm delta chất lượng.

SCD/MS-SSIM và runtime comparison chưa chạy do thiếu approval. Q_ABF ghi NOT_AVAILABLE,không tự gọi một implementation chưa kiểm chứng là metric paper. Không suy mIoU từ fusion metrics.

## 5. Artifact của báo cáo

- train_history.csv:1row pilot,stage=GPU_PILOT_ONLY,completed_full_epochs=0,epoch_complete=False.
- dev_fusion_comparison.csv:chỉheader,0data rows;không có số liệu giả hoặc placeholder175ảnh.
- checkpoint_metadata.json:original/pilot paths,hashes,progress,approval và reload proof.
- environment.json:Python3.11.9,torch2.11.0+cu130,RTX5060Laptop8GB,library/Git metadata thật.
- run_config.yaml:copy byte-exact YAML PROVISIONAL,không sửa recipe.
- figures/README.md:NOT_RUN;không tạo figure từ checkpoint pilot rồi gọi full comparison.
- ARTIFACT_AUDIT.json:inventory/status/hash,scope,integrity và phần chưa chạy.

Các logs/inputSHA/pilot checkpoints nằm tại artifact root ở mục2;unit fixtures tổng hợp dùngTEMP. Audit file-open chặn đọcFMBTest/Label và mọi writeFMB;2454open events chỉ2440uniqueVIS/IRpaths của officialTrain. Verification reload không đọc ảnh.

## 6. Giới hạn và điều kiện để tiếp tục

Cần phê duyệt đúng objective,weights1/.2/1,BN policy,LR/scheduler,20epochs,lastselection,clip1.0,crop/batch/seed vàresource caps,gắn approvalrecord với canonical hash mục1. Chỉ đổi chữPROVISIONAL thànhAPPROVED không thay thế review evidence.

Pretraining Student training-ID list vẫn UNKNOWN;không gọi Dev độc lập với fusionpretraining. Chưa full epoch/chưa B3/chưa multiseed/chưa175Dev comparison. Fullrun checkpoint retention và per-epoch time/VRAM cần được đối chiếu trong protocol được review;pilot chỉ có run-level measurements.

Runtime packages đã chạy PASS;Pillow13 ngoài giới hạn repo<12 là environment warning đã có,không tự cài/downgrade. Không push,không sửa manifest/FMB/SAGE/B3 hoặc sourcepipeline. Report folder hiện bịGitignore bỏ qua;không force-add/đổiignore.

## 7. Lệnh tái lập

Từ D:\SAGE_Research\xulianh-finetune,output phải mới:

~~~powershell
$py = 'D:\SAGE_Research\xulianh\.venv-sage\Scripts\python.exe'
& $py -B -m unittest tests.test_sage_finetune_dataset tests.test_sage_finetune_losses tests.test_sage_finetune_checkpoint tests.test_sage_finetune_inference -v
& $py -B scripts/run_sage_finetune.py --config configs/methods/sage_student_finetune.yaml --mode smoke --smoke-batches 2 --device cuda --output '<NEW_PILOT>'
~~~

Pilot thật được gọi cùng CLI arguments dưới Python file-open audit hook;runner vàgpu_pilot.log có ở artifact root. Fullmode sau đây CHƯA chạy và hiện sẽ bịguardchặn:

~~~powershell
& $py -B scripts/run_sage_finetune.py --config configs/methods/sage_student_finetune.yaml --mode train --device cuda --output '<NEW_APPROVED_FULL_RUN>'
~~~

Chỉ dùng fullmode sau khi có reviewrecord hợp lệ và đúng configđãkhóa. Không tự phát hành approval hoặc tiếp tục từpilot để giả làm fullrun. Nếu fullrun tương lai bị ngắt,resume vào thư mục mới với cùngconfig/source/input/mode và companionbest.pt,khôi phục optimizer/scheduler/RNG.
