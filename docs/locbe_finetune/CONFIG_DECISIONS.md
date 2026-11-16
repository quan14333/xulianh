# Decision register fine-tuning

Ngày14/11/2026;deadline19/11. **DRAFT: audit đã thực hiện,training chưa được bật.** Không tạo/sửa fair-training lock,không tái sử dụng authorization/checkpoint B0 cũ. Tài liệu này không phải approval chạy A/B/C.

## 1. Ràng buộc người dùng đã xác lập

| ID | Quyết định | Trạng thái |
|---|---|---|
|U01|Workspace finetune,branch feature/locbe-sage-finetune-1911|CONFIRMED |
|U02|Lần này audit source và tạo4tài liệu,chưa train|CONFIRMED |
|U03|Không thay SAGE checkpoint,B3 code,manifest,FMB gốc hoặc push|CONFIRMED |
|U04|Development1045Train/175Dev,không Test|CONFIRMED,hashes/IDs đã audit |
|U05|Final1220Train,freeze config trước evaluation;không trộn Development|CONFIRMED,chưa chạy |
|U06|Không gọi candidate official loss/novel method hoặc claim semantic chưa đo|CONFIRMED |

## 2. Facts đã xác minh

| ID | Fact | Giới hạn |
|---|---|---|
|F01|Student135617params,Tanh,5features,strict121stateentries|CPU load,chưa forward |
|F02|SHA256 ecfde477ca4cd65401f207d5dc91ce6c475a49891d768dbdb7a66366255ee836|Immutable pretrained input |
|F03|Loss Student gốc cần Teacher/SAM/patches;Teacher segmentation cần GT|Static audit |
|F04|Sai import test.py,GPUindex1,Linux paths,thiếuweights,Train label chia255→long mấtIDs|Source + CPU label probe;không train A |
|F05|SourceRGB khác B3 clamp/rounding|Synthetic probe;chưa đo FMB parity |
|F06|B3 train15class nhưng primary loạiGT0,GT-present mean|Không gọi primary là mIoU15 |
|F07|B3 manifest12runs/MD5,last.pt không local|Chưa verify B3 weight contents |
|F08|RTX5060~8GB nhận diện,torch/cv2/numpy có|Training VRAM/time NOT_MEASURED |

## 3. Candidate v1 cần review

| ID | Giá trị PROPOSED | Căn cứ / rủi ro | Người xác nhận |
|---|---|---|---|
|P01|B;Network32,Conv+BN affine trainable,BN stats frozen|Nhẹ nhưng đổi adaptation policy|Quân |
|P02|L_int=.5MSE(F,V)+.5MSE(F,I)|Có bias averaging|Quân |
|P03|L_grad=max-source Sobelmag L1,padding1|Theo operator source,có noise/border risk|Quân/owner metric |
|P04|L_anchor=MSE(F,F0),referenceStudent frozen raw output|Giữ gần pretrained,2forward|Quân |
|P05|Weights1/.2/1,loss raw Tanh|Heuristic,chưa sweep/chưaapproved|Quân |
|P06|Adam LR1e-5,betas(.9,.999),eps1e-8,wd0|Fine-tune setting mới|Quân |
|P07|20epochs,last;cosine/update,eta_min1e-6,no warmup|Budget proposal,không bestTest|Quân |
|P08|Crop192×256,paired hflip.5,không label-aware crop,workers0|Size theo source,crop policy mới|Quân |
|P09|Micro2,accum2,effective4,drop_lastFalse|1045ảnh→window cuối1ảnh;normalize actual samples|Quân |
|P10|FP32,no AMP/no TF32,seed2,determinism guard|Chưa dry-run CUDA backward|Quân |
|P11|B3 PNG contract cho cảS0/candidate,full H×W|Tránh exporter confound|Quân/Mân |
|P12|Budget8h,peakallocated<=4GiB,free>=1GiB|Operational gates chưa đo|Quân/người chạy |
|P13|Final reset pretrained,1220×20epochs,last;6100updates|Không dùng Dev-fitted init|Quân/Mân |
|P14|sage_ft_selfsup_v1,B3seeds0/1/2|Cần resources/initialization artifacts|Mân |

Không giả lập approval. Quân có thể bác B/đổi version. Không cần thêm permission để hoàn thành bốn tài liệu đang được yêu cầu;việc review ở đây là điều kiện cho giai đoạn training tương lai.

## 4. Config có thể chuyển thành YAML sau review

~~~yaml
status: proposed_not_authorized_to_train
method: sage_ft_selfsup_v1
source_commit: 377fdbef6b8b6ccbd1139358ad66bb41ea084dc1
checkpoint_sha256: ecfde477ca4cd65401f207d5dc91ce6c475a49891d768dbdb7a66366255ee836
architecture: model_sub.model.Network
architecture_dim: 32
protocol: development_1045_175
seed: 2
epochs: 20
checkpoint_selection: last_epoch_20
optimizer: {name: Adam, lr: 0.00001, betas: [0.9, 0.999], eps: 0.00000001, weight_decay: 0.0}
scheduler: {name: cosine_per_optimizer_update, eta_min: 0.000001, warmup_updates: 0}
training: {microbatch: 2, accumulation: 2, drop_last: false, num_workers: 0, precision: fp32, tf32: false}
augmentation: {crop_hw: [192, 256], paired_hflip_p: 0.5, label_aware_crop: false}
normalization: uint8_div_255
input_order: [VIS_Y, IR_GRAY]
student_bn: {running_stats: frozen_pretrained, affine: trainable}
reference: {type: frozen_original_student, mode: eval, gradients: false}
loss: {intensity: 1.0, max_sobel_gradient: 0.2, output_anchor: 1.0, raw_tanh_outputs: true}
export_contract: b3_y_clamp_round_uint8_then_opencv_ycrcb_to_bgr
use_train_labels_in_fusion_loss: false
use_dev_labels_in_fusion_loss: false
use_test: false
~~~

Block YAML chưa được trainer đọc/chạy. Sau review serialize config canonical,ghi SHA256 và revision trong registry **trước** run. Thay values phải tạo hash/version mới;hash tài liệu không phải bằng chứng approval. Final đổi protocol/manifest union/tổngupdates doNkhác,mọi hyperparameter còn lại freeze.

## 5. Câu hỏi cụ thể để bàn giao

1. Quân có training-ID list/dataset/seed/provenance của pretrained Student không? Có Teachercheckpoint và SAMencoderweights đúng depth/source không? Thiếu thì ghi UNKNOWN,không gọi Dev độc lập.
2. Có đồng ý B là baseline source-preserving adaptation,không tái lập đầy đủ semantic distillation?
3. Intended label encoding source là integerclassID hay binarymask? Vì sao ToTensor/long,threshold.5,ROI tọa độ và XDecoder class mapping không nhất quán? Có bản sửa đã review không?
4. Có duyệt output-anchor/BNfreeze,loss/hệ số,20epochs,lastselection,AdamLR và compute cap không? Không chạy ablation tự phát trên Test.
5. Mân xác nhận PNG control thực tế cùng exporter commit nào? Cần samplePNG/hash để parity. Có config,HF revision,initialization tensors/MD5/SHA weights B3 nào để kiểm không?
6. Batch8,crop512,B3AMP trênRTX8GB có fit chưa? Nếu không,ai nhận handoff trên nguồn lực đã chốt? Không tự chuyển Kaggle/đổi recipe.
7. Ai thực hiện final Test evaluation sau freeze? Metricprimary/exclusions/version đã khóa chưa? Báo cáo TestB3 cũ đã được đọc,không mô tả final như blindstudy hoàn toàn.

## 6. Go/no-go và artifact tương lai

Chỉ Go sau review và input hash/strictload/parity/gradient/determinism/VRAM đạt. OOM,NaN,initkhác,Label vào fusion B hoặc Test access là stop,không skip loss/tự giảm batch. Tạo runID/path mới,không overwrite artifact cũ.

Cần lưu sau implementation:canonical config/hash,source/environment,manifest/checkpoint hashes,initial state verification,2batchdryrunloss/grad/VRAM/ETA,history,last.pt,optimizer/scheduler/RNGresume,PNG QA/per-image hash,Dev report. Những artifact này chưa có vì chưatrain;không thay NOT_MEASURED bằng PASS giả.

Xem [audit](FINETUNE_AUDIT.md),[proposal](BASELINE_PROPOSAL.md),[B3 integration](B3_INTEGRATION_PLAN.md).
