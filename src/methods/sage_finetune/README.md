# SAGE Student fine-tuning — PROVISIONAL

Baseline adaptation của nhóm CS406, không phải official SAGE training protocol.
Implementation dùng đúng Student source được pin SHA256, không sửa SAGE hoặc B3.
Hiện chưa có approval của Quân: full training bị chặn, smoke tối đa2microbatch,
export PROVISIONAL tối đa5ảnh. Không có CLI FMB Test.

## Objective và cấu hình

V=VIS_Y,I=IR_GRAY chia255, F=rawTanh Student, F0=frozen pretrained Student.
E(A,B)=mean((A−B)^2),G(X)=|Sobel_x*X|+|Sobel_y*X|,zero-padding1,unnormalized kernels.

- intensity=.5E(F,V)+.5E(F,I)
- gradient=mean|G(F)−max(G(V),G(I))|
- anchor=E(F,stopgrad(F0))
- total=1.0intensity+.2gradient+1.0anchor

Chi tiết công thức và phân biệt source/candidate:
[BASELINE_PROPOSAL](../../../docs/locbe_finetune/BASELINE_PROPOSAL.md).
Config: configs/methods/sage_student_finetune.yaml.
Adam LR1e-5,cosine eta_min1e-6,20epochs đề xuất,crop192×256,pairedflip.5,
micro2×accum2,FP32/noTF32,seed2,workers0.
Gradient clip norm1.0 là safeguard mới được yêu cầu;giá trị1.0 còn PROVISIONAL.
Method folder sage_finetuned là alias tích hợp của candidate sage_ft_selfsup_v1,
không đổi công thức/loss.

Student và reference load strict=True từ pretrained, so từng tensor và zero-step output.
BN running stats frozen, BN affine học; train_patterns/freeze_patterns hỗ trợ unfreeze/freeze
theo tên parameter, pattern không match sẽ fail. Đổi BN/loss/recipe làm config hash mới.

## Lệnh thực thi

Chạy từ xulianh-finetune bằng Python hiện có. Mỗi output phải là thư mục mới;
các đường dẫn sau chỉ là ví dụ, không ghi đè artifact có sẵn.

~~~powershell
$py = 'D:\SAGE_Research\xulianh\.venv-sage\Scripts\python.exe'
& $py -X utf8 -B scripts/run_sage_finetune.py --config configs/methods/sage_student_finetune.yaml --mode smoke --smoke-batches 2 --device cuda --output '<new-artifact-root>\smoke'
& $py -X utf8 -B scripts/export_sage_finetuned.py --config configs/methods/sage_student_finetune.yaml --checkpoint '<new-artifact-root>\smoke\last.pt' --split dev --limit 5 --device cuda --output '<new-artifact-root>\export'
& $py -X utf8 -B -m unittest tests.test_sage_finetune_dataset tests.test_sage_finetune_losses tests.test_sage_finetune_checkpoint tests.test_sage_finetune_inference -v
~~~

CPU smoke: đổi --device cpu. Smoke thực hiện2microbatch và1optimizerupdate khi accumulation2.
Smoke validation dùng2Dev IDs đầu, KHÔNG phải loss trung bình toàn175Dev.
Checkpoint được ghi dry_run_only=true; không có mIoU hoặc claim improvement.

## Dữ liệu và output contract

Đọc đúng manifesthash,1045Train/175Dev,unique/disjoint IDs dạng string5chữsố.
Loader chỉ mở VIS/IR; không mở Label. Hỗ trợ flat train/Visible và nested train/train/Visible,
tương tự IR. Pure resolver có tùy chọn explicit cho nested Test để adapter tương lai dùng,
nhưng Development dataset/CLI luôn cấm Test trước khi mở ảnh. Unit Test-layout dùng
fixtures tổng hợp trong TEMP, không dùng FMB Test.

Preflight decode/hash100%Train+Dev,MD5 từ manifest và SHA256 provenance.
Không resize ảnh hoặc đổi màu trước fusion:OpenCV BGR→YCrCb/GRAY,float32/255.
Crop/flip cùng tọa độ trên hai nguồn,deterministic theo seed/epoch/ID.
Không ImageNet normalization trong SAGE.

Export B3: clampY[0,1]→×255→torch.round→uint8,giữ CrCb VIS,
cv2YCrCb→BGR→PNG. File PNG decoder đọc RGB; OpenCV array BGR.
Không (F+1)/2,minmax hay clampRGB như official source test.py.
Source RGB contract khác B3:không trộn cache giữa chúng.
Reopen mỗiPNG,pixel equality,shape,dtype,SHA256 được kiểm tra.

Layout artifact export: sage_finetuned/train/<image_id>.png cho cảTrain vàDev,
do cả hai manifest trỏ officialTrain; manifest_split vẫn phân biệt train/dev.
Gắn SAGE_ANH với root export và --bien_the sage_finetuned trong B3 trainer sau review.
Không chạy xuat_anh.py Mân vì exporter đó gộp cả Test.
Không copy hoặc sửa B3 code. Xem [B3 integration](../../../docs/locbe_finetune/B3_INTEGRATION_PLAN.md).

## Checkpoints, best/last và resume

last.pt là checkpoint chính theo proposal. best.pt chỉ tối thiểu Dev reconstruction objective
để diagnostic,không Person/mIoU/semantic selection. Save model,optimizer,scheduler,
Python/numpy/torch/CUDA RNG,epoch/sample cursor,history,config/source/inputhashes.
Checkpoint load dùng weights_only=True,không strip keys hoặc strict=False.

Resume vào output MỚI bằng --resume <old-run>/last.pt,cùngmode/config/source/input/device.
Giữ companion best.pt cạnh last.pt để carry diagnostic best khi resume.
Chỉ checkpoint tại optimizer boundary; không checkpoint gradient đang accumulation.
Cursor và epoch-seeded permutation khôi phục thứ tự dữ liệu.
Smoke checkpoint không được promote thành full training. Raw pretrained state chỉ dùng
khởi tạo/inference,không giả làm optimizer resume. Unit tests kiểm replay next update.

## Guards và approval

Full mode=train đòi statusAPPROVED và approval_record JSON chứa
allow_full_training=true,approved_by,evidence,config_sha256 khớp Config.fingerprint().
Không tạo approval giả; không bật full training trong giai đoạn này.
Hash excludes status/approval_record nhưng bao toàn recipe,input/source/checkpoint identities.
Fingerprint chỉ cung cấp công cụ bind config,không tự xác nhận review của con người.

NaN/Inf,missing/nonfinite/zero-global gradient,wronghash,shape,Test/Dev vào optimizer,
reference/BN frozen drift,OOM hoặc deterministic-kernel lỗi đều fail và lưu failure.json.
Không skip loss,đổi batch/crop/precision hoặc fallback CPU âm thầm.
CUDA guards peakallocated<=4GiB/free>=1GiB theo YAML; log peak reserved riêng.
Các giá trị cap/clip/recipe vẫn cần Quân review.

## Artifact

Training:config.json,environment.json,input_audit.csv,initialization.json,batch_losses.csv,
history.csv,last.pt,best.pt,run_result.json hoặc failure.json.
Export:config/environment,output_manifest.csv,export_result.json,PNGs.
Config/source/input/original/model/output hashes được ghi để audit.
Python3.11+ (hashlib.file_digest). Dependencies tối thiểu torch,numpy,opencv-python,PyYAML;unit tests dùng unittest standard library,
không cần cài pytest/SAM/XDecoder. Không tự cài/downgrade môi trường đang có.
Pretraining Student training-ID list vẫn UNKNOWN;Dev không được gọi độc lập với pretraining.


## B3 adapter và kiểm tra trước commit

Registry configs/methods/b3_variants.yaml giữ riêng sage pretrained,
sage_finetuned pilot và quan_improved chưa có spec. Xem
[HANDOFF_19NOV](../../../docs/locbe_finetune/HANDOFF_19NOV.md) và
[báo cáo pilot](../../../reports/experiments/locbe_sage_finetune/README.md).

~~~powershell
# Chạy tại root repository, sau khi cài dev dependencies phù hợp.
$pytestTemp = Join-Path ([IO.Path]::GetTempPath()) ('sage-pytest-' + [guid]::NewGuid().ToString('N'))
if (Test-Path -LiteralPath $pytestTemp) { throw 'Use a new pytest temp path' }
& $py -X utf8 -B -m pytest -q -p no:cacheprovider --basetemp $pytestTemp
& $py -X utf8 -B scripts/validate_repo.py
& $py -X utf8 -B scripts/export_b3_variant.py --help
& $py -X utf8 -B scripts/check_b3_inputs.py --help
~~~

Pytest chạy các fixtures tổng hợp/metadata checks, không train mô hình hoặc
đánh giá ảnh FMB Test. Local pre-commit đã bổ sung pytest theo requirements-dev;
không đổi PyTorch/CUDA. Pillow13 trong môi trường local nằm ngoài constraint
repo<12 đã có; đây là giới hạn tái lập, không phải dependency lock mới.
