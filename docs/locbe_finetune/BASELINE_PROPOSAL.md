# Đề xuất baseline fine-tuning SAGE Student

**DRAFT — candidate của nhóm, chưa phê duyệt chạy training.** Ngày14/11/2026; deadline19/11. Khuyến nghị **B** cho implementation/bàn giao nhẹ. A/C là các phương án phân tích, không phải những run được bật trong lần này. Chưa có loss,VRAM,time,mIoU sau fine-tuning.

## 1. So sánh A/B/C

| Tiêu chí | A: Teacher+Student loss gốc | B: Student-only nguồn+anchor | C: Student có segmentation signal |
|---|---|---|---|
| Assets | Student; Teacher checkpoint nếu tiếp tục train; SAM generator,encoder depth2 đúng source; X-Decoder/tokenizer | Pretrained Student và một bản Student frozen làm reference; không Teacher SAGE/SAM/X-Decoder | Student và segmenter đã train head FMB,mapping/provenance verified |
| Thư viện | Full SAGE/X-Decoder stack; nhiều package/extension đang thiếu trên Windows | torch,numpy,cv2;CSV/JSON/hash standard library;không import SAM common | B+transformers/stack segmenter đã audit |
| Dữ liệu | VIS/IR,GT class masks Train,SAM region cache | Chỉ VIS/IR Train;fusion loader/loss không đọc labels | VIS/IR+GT Train đúng integerIDs;không GT Dev/Test trong loss |
| Compute | Hai fusion nets,nhiều SAM forwards,mask generation,X-Decoder và autograd lớn | Một Student forward học,một reference forward no_grad,một Student backward/crop | Student forward/backward+segmenter forward/input-backward;anchor nếu giữ B |
| Trung thành source | Gần cơ chế nhất,nhưng cần xác nhận labels/path/OOM/loop | Giữ architecture/weights/preprocess;đổi objective và recipe công khai | Đổi thành task-aware objective,khác distillation gốc |
| Trước19/11 | Rủi ro cao:thiếu assets,môi trường,8GB chưa fit | Khả thi nhất cho implementation+smoke+một Dev run nếu gates đạt | Rủi ro vừa/cao:valid FMB evaluator,gradient contract,task/domain bias |
| Fairness | Sửa source hoặc skip OOM đổi objective;không có pretrained Teacher để resume | Có thêm budget so frozen Student;dễ tiến về averaging;phải có TB control | Thêm supervision labels;dùng cùng network cho loss và score có thể quá chuyên biệt |
| Quân xác nhận | Teacher/weights/provenance,intended masks/labels,loop/loss và runtime fixes | Loss/hệ số,BN,recipe,budget,RGB contract,tên method | Segmenter checkpoint/IDs,class0/255,loss và evaluator độc lập |

C không được dùng một ADE20K/ImageNet head chưa train FMB để làm target có ý nghĩa. Checkpoint B0 local cũ nếu cân nhắc phải audit riêng trước, không tự coi tương thích B3. Với C, freeze weights và eval segmenter nhưng vẫn cần gradient về input; no_grad quanh segmentation forward sẽ chặn signal. VIS-trained loss network có domain bias; cần matched downstream assessment độc lập.

## 2. Lựa chọn B và ranh giới

Tên đề xuất: sage_ft_selfsup_v1. Câu hỏi là adaptation từ pretrained bằng source-preservation có giúp hơn frozen Student và TB không. Không thiết kế architecture/gating mới,không chọn lại alpha/threshold. Loss fusion không tự chứng minh cải thiện semantics.

Load Network(dim32) với checkpoint SHA256 ecfde477ca4cd65401f207d5dc91ce6c475a49891d768dbdb7a66366255ee836. S_theta và S_0 cùng strict-load toàn bộ state. S_0 eval,frozen,no_grad,không update; đó là reference Student,không phải Teacher SAGE hoặc semantic supervision mới. Có **hai instance Student**,chi phí reference phải tính thật.

S_theta học mọi Conv và BN affine gamma/beta. Sau mỗi train(),đặt BatchNorm2d về eval để giữ running_mean/running_var/num_batches_tracked pretrained;affine vẫn có gradient. Đây là policy nhóm mới,không phải mode source. Test phải chứng minh buffers không drift.

## 3. Loss chính xác

V,I là VIS_Y/IR_GRAY float32[0,1],cùng crop/flip. F=S_theta(V,I)[0],F0=stopgrad(S_0(V,I)[0]) là **raw Tanh**. Không clamp,quantize hoặc (F+1)/2 trong loss.

N=BHW; E(A,B)=N^{-1}sum(A−B)^2.

$$L_{int}=\tfrac12E(F,V)+\tfrac12E(F,I).$$

$$K_x=\begin{bmatrix}-1&0&1\\-2&0&2\\-1&0&1\end{bmatrix},\quad
K_y=\begin{bmatrix}1&2&1\\0&0&0\\-1&-2&-1\end{bmatrix}.$$

Sobel unnormalized,stride1,zero-padding1,kernels fixed cùng device/dtype. G(X)=|Kx*X|+|Ky*X|;max elementwise.

$$L_{grad}=N^{-1}\sum|G(F)-\max(G(V),G(I))|,\qquad L_{anchor}=E(F,F_0).$$

$$\boxed{L=1.0L_{int}+0.2L_{grad}+1.0L_{anchor}}.$$

Hệ số1/.2/1 là **heuristic duy nhất đề xuất**,chưa tối ưu/chưa approved. Intensity và anchor cùng thang MSE;Sobel khác thang nên đề xuất trọng số thấp hơn. Không có evidence .2 tối ưu. Thay hệ số phải version lại trên Development,không dùng Test lựa chọn.

Sobel/max theo operator GradientLoss source;balanced intensity,output-anchor S0 và tổ hợp/hệ số là baseline nhóm,không official SAGE loss. Không import model_sub.common/Losses vì constructor kéo SAM/hard-coded CUDA;implementation sau viết loss tối thiểu thuần torch.

### Hạn chế và đối chứng

Bỏ gradient/anchor thì intensity có cực tiểu (V+I)/2. Bỏ gradient nhưng giữ weights1/1 thì nghiệm pixel không ràng buộc là .25V+.25I+.5F0. Vì vậy cần TB và frozen SAGE;không diễn giải adaptation như learned semantic reasoning. Có thể đăng ký phép trộn đóng trên làm ablation riêng sau review,không chạy thêm tự phát.

Max-Sobel có thể giữ nhiễu/halo;balanced MSE có thể mất IR vùng nhỏ;anchor mạnh có thể khóa cải thiện. Zero-padding tạo gradient border trên ảnh constant;unit test phải phân biệt interior/border. Không có fused GT để bảo đảm target đúng. Chưa review đủ prior art để claim novelty.

## 4. Recipe Development đề xuất

| Mục | Candidate v1 | Nguồn / thay đổi |
|---|---|---|
| Train/Dev |1045 Train/175 Dev;chỉ VIS/IR vào fusion loss | Split nhóm;khác source loader liệt kê thư mục |
| Epoch/selection |20epochs,last_epoch20;không early stop/best SCD/Person | Nhóm mới;không phải100epochs source |
| Optimizer |Adam,betas(.9,.999),eps1e-8,weight_decay0 | Cùng họ source,defaults explicit |
| LR/schedule |1e-5,cosine theo optimizer updates,eta_min1e-6,không warmup | Fine-tune proposal;source StudentLR2e-3 |
| Crop/augmentation |192×256 H×W,một crop/ảnh/epoch,paired hflip.5;không jitter/scale/label-aware crop | Kích thước/flip source;crop không nhãn do nhóm |
| Batch |micro2,accum2,effective4,drop_lastFalse,workers0 | Đề xuất cho Windows8GB,chưa memory-tested |
| Precision |FP32,no AMP,no TF32 | Nhóm mới;khác B3 downstream AMP |
| Seed |2 cho Python/numpy/torch/CUDA,generator shuffle | Source default2;v1 một seed feasibility |
| Mode |Student train,BN eval/affine trainable;reference eval/frozen | Nhóm mới |
| Output |Cùng B3 PNG contract cho S0/candidate,full H×W | Khóa theo evaluator,không claim source RGB parity |

523 microbatches/epoch,262updates/epoch,**5240updates**/20epochs. Cửa sổ cuối có1ảnh:normalize gradient theo số ảnh thực tế trong accumulation window,không cố chia2 như cửa sổ đầy. Scheduler step sau optimizer,một lần/update. Ảnh nhỏ hơn crop phải fail,không resize âm thầm.

cudnn.benchmark=False;determinism guard cần được kiểm tra với bilinear-backward CUDA của torch local trước run. Nếu unsupported thì dừng để review,không tự tắt guard/đổi interpolation. Không hứa bitwise replay trước dry-run. FP32 gồm tắt TF32 matmul/cuDNN,không chỉ tắt autocast.

## 5. Hai protocol không trộn

| Protocol | Fusion học | Selection | Downstream / giới hạn |
|---|---|---|---|
| Development |1045 Train;175 Dev ngoài optimizer |v1 cố định last_epoch20;Dev chỉ chẩn đoán | Fusion metrics trên cùng175 IDs;segmentation nếu làm phải khóa evaluator/recipe và báo bias |
| Final B3-compatible |Reset pretrained gốc,train lại1220 officialTrain |Last_epoch20,không Test/Person selection | B3 riêng mỗi method,100epochs,seeds0/1/2,last.pt;eval sau freeze |

Final có305updates/epoch=6100updates/20epochs;cosine vẫn cùng công thức theo updates/total. Không tiếp tục từ Dev-fitted checkpoint làm final init để thêm budget âm thầm. Model final đã học1220 không có held-out Dev nữa.

Đối chứng tối thiểu frozen SAGE,TB/Y_AVG,candidate. Cùng source IDs,RGB contract,range,segmentation init/seed/evaluator. Cache cũ phải hash/config-parity trước reuse. Dev có thể từng xuất hiện trong pretraining Student;chỉ xác minh được split mới ở bước fine-tune. Không lấy Test scores B3 đã đọc để chọn coefficients. Không so B0 Dev mIoU15 với B3 primary mIoU_co_mat.

SCD/edge/MS-SSIM/loss là evidence fusion,chưa phải semantic evidence. Semantic claim cần matched mIoU/per-class/person/dark thực đo. Frozen VIS-trained evaluator chỉ diagnostic có domain bias. Một seed đủ feasibility,chưa chứng minh generalization.

## 6. Dự toán minh bạch

**Chưa đo training time/VRAM.** Những số dưới là workload/giả định thiết kế:

- B:20×1045=20900 lượt crop train,41800 forward instance lượt ảnh gồm reference,20900 backward lượt ảnh,5240optimizer updates.
- Hai Student weights FP32 khoảng1.09MB;weights+gradient+hai Adam moments của bản học cộng reference khoảng2.71MB,chưa buffers/activations/CUDA context/workspaces.
- Một tensor B2×32×192×256 FP32 khoảng12.6MB. Nhiều saved activations mới quyết định peak. Dự toán thô0.5–3GiB allocated **chưa đo**;không bảo đảm fit. Gate đề xuất peak allocated<=4GiB và free>=1GiB,bao gồm reference.
- Throughput đo được q ảnh/s phải gồm reference/backward. Train time≈20900/q giây:giả định q=1→5.81h,q=5→1.16h,chưa IO/Dev export. Đây không phải benchmark RTX5060. Budget đề xuất8h/Dev run;dry-run phải dự báo đủ buffer,dùng AC và log power/throttling.
- A:100epochs,luân phiên2fusion nets,SAM vùng/encoder và X-Decoder. Nếu đủ20 bool masks/modality cho1220ảnh600×800,chỉ masks khoảng23.4GB,chưa metadata;thực tế có thể ít vùng hơn. Thiếu assets/throughput/peak nên không có ETA đáng tin trước19/11.
- C: nếu dùng cùng20epochs/1045Train như B, cần thêm20900segmenter forwards và20900input-backwards ngoài phần Student; freeze weights không loại activations. Đây là đếm workload có điều kiện, không phải recipe C đã khóa hoặc benchmark. Cần checkpoint FMB hợp lệ và dry-run riêng trước khi ước tính giờ/VRAM.
- B3 downstream không tính vào giờ Student. Report Mân ghi~2.6h/run T4 (~7.8GPUh/candidate×3seeds),**không phải ETA RTX5060**. B3batch8crop512 trên8GB chưa fit-tested;không giảm batch/đổi recipe rồi gọi matched với bảng cũ.

## 7. Kế hoạch bàn giao

| Ngày | Deliverable / điều kiện |
|---|---|
|14/11|Bốn tài liệu audit/design;phạm vi lần này,chưa train |
|15/11|Review Quân/Mân;sau khi được giao mới implement trainer/exporter/tests,parity,2batch dry-run |
|16/11|Một Dev run B nếu approval/hash/VRAM/ETA đạt |
|17/11|Dev report thật,baseline parity/failure cases;không Test/semantic claim chưa đo |
|18/11|Khóa final config/source/manifest;1220run và B3 handoff tùy resources đã xác nhận |
|19/11|Bàn giao code/config/log/checkpoint nếu có run thật;nếu thiếu báo NOT_MEASURED,không điền số |

Đây là lịch đề xuất,không ủy quyền tự chạy. Mục tiêu khả thi nhất là implementation có test/provenance và một feasibility run sau review;không cam kết đủ full matched B3 trước deadline.

## 8. Gates trước optimizer step đầu tiên

1. Hash/IDs1045/175 unique/disjoint đúng;fusion loader chỉ mở VIS/IR Train,không Label/Dev/Test. Resolve train/Visible/x thành FMB/train/train/Visible/x.
2. Decode100%,shape/alignment/uint8 đúng,paired crop/flip,float32 N1HW[0,1];không ghi FMB gốc.
3. Strict-load S_theta/S0,so từng tensor và BN buffers;zero-step eval outputs bằng nhau;reference/checkpoint immutable.
4. Unit test intensity/Sobel/anchor với tensors biết trước,border/interior,finite/nonzero gradients,input order;không gradient vào S0.
5. Hai batch dry-run CUDA,optimizer chỉ update allowed tensors,BN stats không đổi,input/hash/labels guards,determinism,peak/free VRAM đạt. **Chưa chạy trong audit.**
6. Serialize/strict-reload checkpoint mới,log optimizer/scheduler/RNG/sampler để resume;OOM/NaN dừng,không skip loss hoặc tự đổi batch/precision.
7. Export full H×W,reopen/hash100%PNG;S0 parity với B3 exporter trên cùng ảnh/environment. Contract definition không thay real-image parity test.

Các hyperparameter,BN/loss/output/budget/final policies đều **PROPOSED**. Quân xác nhận source/assets/mục tiêu;Mân xác nhận exporter/evaluator/resources;owner metric review definitions. [CONFIG_DECISIONS.md](CONFIG_DECISIONS.md) ghi rõ chưaapproved. Không gửi tin thay người dùng,không coi việc tạo tài liệu là approval.
