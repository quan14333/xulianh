# SAGE Student fine-tuning — evidence bàn giao

**PROVISIONAL / BLOCKED_FOR_APPROVAL** cho full Student training. Đây là
Student-only candidate của nhóm, không phải loss/training protocol chính thức
SAGE hoặc phương pháp cải tiến Quân. **0/20 epoch hoàn tất**; pilot thật chỉ
2 microbatch,4Train images,1optimizer update. Không có claim cải thiện semantics.

Báo cáo thực nghiệm gốc:
[FINETUNE_TRAINING_REPORT.md](FINETUNE_TRAINING_REPORT.md).
Giữ nguyên văn bản và thời điểm lịch sử; câu “folder đang Gitignore” trong báo cáo
là trạng thái trước đợt chuẩn bị commit này. Chỉ bốn file nhỏ trong thư mục này
được allowlist; logs,environment đầy đủ,checkpoints,PNGs vẫn local.

## Cấu hình và kết quả có thật

| Hạng mục | Giá trị |
|---|---|
| Development |1045Train/175Dev,disjoint;fusion loss không dùng GT labels |
| Objective nhóm |L_int+.2L_grad+L_anchor;chi tiết trong proposal |
| Recipe đề xuất |Adam1e-5,cosine eta_min1e-6,target20epochs,last,seed2 |
| Input/batch |crop192×256,pairedflip.5,micro2×accum2,FP32,noTF32 |
| Pilot |finite losses/gradients,70parameter tensors thay đổi;BN/reference giữ nguyên |
| Resume |optimizer/scheduler/Python,numpy,torch,CUDA RNG reload PASS |
| B3 integration |5Dev×2variants,10PNG600×800RGB,actual FMBSeg loader PASS |
| Independent B3 pixels |10/10exact,max_abs_error0 |
| Quân method |WAITING_FOR_METHOD_SPEC;interface và fail-fast stub |
| Full fine-tuned baseline / B3 mIoU |NOT_AVAILABLE / NOT_MEASURED |

[train_history.csv](train_history.csv) có đúng một row GPU_PILOT_ONLY,
epoch_complete=False. Không đọc epoch index1 thành một epoch hoàn tất.
[public_evidence.json](public_evidence.json) trích số đo/hash thật từ artifact
local, bỏ raw paths và git-status logs. Hai timing export là smoke có first-call
overhead, không dùng để kết luận fine-tuned nhanh hơn.

## Reproduce và review

[Handoff](../../../docs/locbe_finetune/HANDOFF_19NOV.md),
[README CLI](../../../src/methods/sage_finetune/README.md),
[proposal](../../../docs/locbe_finetune/BASELINE_PROPOSAL.md).
Dùng Python có dependency phù hợp, từ root repo:

~~~powershell
python -X utf8 -B -m pytest -q
python -X utf8 -B scripts/validate_repo.py
python -X utf8 -B scripts/run_sage_finetune.py --help
python -X utf8 -B scripts/export_b3_variant.py --help
python -X utf8 -B scripts/check_b3_inputs.py --help
~~~

Không ship dataset/checkpoint hoặc tải lại chúng trong unit tests. Source Student
và pretrained phải được cung cấp riêng,hash-pinned theo YAML khi chạy real GPU
smoke. Dataset1045/175 và config local paths được cấu hình rõ,không giữ token.

Quân review objective/hệ số/BN/recipe và cung cấp method spec. Mân review RGB/PNG
contract,mapping15IDs,mIoU_co_mat và B3 recipe. Full runs cần approval record khớp
fingerprint và valid B3 FMB checkpoint; không dùng ADE20K/B0 thay B3.
Dev chưa được chứng minh độc lập với SAGE pretraining. Không dùng Test lựa chọn
tham số/checkpoint; không có mIoU mới trong evidence này.
