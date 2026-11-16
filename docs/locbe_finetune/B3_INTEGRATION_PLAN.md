# Kế hoạch tích hợp fine-tuned Student với B3

Ngày14/11/2026. **DESIGN ONLY: không export/train/eval lần này.** B3 reference=58a24847ac68aca6f232194cf1bc84d9fcf4be52,worktree chỉ đọc. Branch cá nhân ở c4a6de4f... chưa chứa B3;không tự merge nhánh chưa review.

## 1. Input/output contract

| Thành phần | Contract |
|---|---|
| ID |image_id chuỗi5chữsố theo manifest,không zip theo thứ tự thư mục |
| Fusion input |VIS BGR→YCrCb(Y,Cr,Cb),IR BGR→GRAY,divide255,float32,N1HW,model(VIS_Y,IR) |
| Raw output |Tuple phần tử0,Tanh hữu hạn;không minmax/(F+1)/2 |
| B3 exporter |Y8=round(clamp(F,0,1)*255).byte();Cr/Cbuint8 VIS giữ nguyên;cv2YCrCb→BGR;imwritePNG |
| PNG |uint8,3channels,H×W gốc;reopen/QA/SHA256;không resize khi export |
| Layout |SAGE_ANH/method/source_split/image_id.png;source_split từ vis_path |
| Train+Dev |Cả1045/175 đều source_split=train;1220IDs disjoint có thể ở cùng method/train |
| Seg input |BGR→RGB,(RGB/255−mean)/std;mean(.485,.456,.406),std(.229,.224,.225),CHW |
| Seg labels |--lop0:integerIDs0..14,không chia255;nearestresize,pad255,CEignore255 |
| Pred |Bilinear logits align_cornersFalse vềH×W rồiargmax;15class giữID0..14 |

Chỉ điều chỉnh Y,không thay VIS chroma. Officialsource RGB khác contract B3 đã audit;control pretrained và candidate phải cùng B3 exporter. Cache B0/source-save_image cũ không tự tương thích.

Cache manifest từng ảnh cần method,protocol,split,image_id,VIS/IRhash,pretrained/candidateSHA256,source commit,exportconfigSHA256,shape,dtype,PNGhash,QAstatus. Không coi “file tồn tại” là bằng chứng cache hợp lệ. Checkpoint mới path riêng,không đổi upstream checkpoints.pt.

## 2. Development1045/175

Fusion học đúng1045Train;loss B chỉ VIS/IR,không labels. Dev175 ngoài optimizer,không GTDev vào fusion loss. Last_epoch20 đề xuất cố định trước run;Dev fusionmetrics là diagnostic.

Không chạy eval_seg.py Mân để chấm Dev:script hard-code FMBSeg(['test']). Nếu cần Dev segmentation,runner riêng trong branch cá nhân dùng cùng convention/evaluator trên Dev,không sửa reference. FrozenVIStrainedsegmenter có domain bias;matchedretraining1045/175 mạnh hơn,nhưng phải đăng ký recipe/init/metric/seed riêng.

Không suy semantic từ SCD/loss. Dev có thể đã nằm trong Student pretraining vì chưa có training-ID list. Audit chỉ xác minh không đưa Dev vào bước fine-tune mới.

## 3. Final B3-compatible1220

1. Freeze formula,weights,epochs,BN,RGBcontract,source/config/manifest hash và checkpoint selection trước Test evaluation mới. Báo cáo TestB3 cũ đã đọc nên không gọi final hoàn toàn blind.
2. Tạo manifest dẫn xuất mới hợp train.csv+dev.csv=1220uniqueIDs,không sửa originals/không Test.
3. Reset pretrained gốc,cùngseed2,train1220×20epochs theo config approved,last_epoch20;6100updates. Không tiếp tục từ Dev-fitted checkpoint để thêm budget.
4. Export1220Train bằng final Student eval/no_grad,full resolution;freezecheckpoint/hash trước Test export.
5. MatchedB3 train riêng cho candidate/cáccontrol theo cùng initializer/recipe Mân;100epochs,seeds0/1/2,last.pt,không chọn Test best.
6. Chỉ sau freeze và khi được giao finalevaluation mới exportTest/chấm bằng B3,không tune theo Test. Chưa thực hiện bất kỳ bước này ở audit.

Model final học1220 không có held-out Dev nữa. Không gộp Dev/B0 mIoU15 với final B3 primary. Đổi batch/crop/AMP đểfit8GB tạo protocolkhác cần matchedcontrols mới,không reuse bảngB3cũ như cùngrecipe.

## 4. Thêm method mà không sửa Mân

README gợi ý thêm nhánh xuat_anh.py,nhưng file đó chỉ có choices vis/ir/tb/sage/clahe và mặc địnhgộpTrain/Dev/Test. Vì yêu cầu reference chỉ đọc vàDevelopmentkhôngTest,không chạy exporter này cho candidate.

Implementation sau review ở namespace Lộc bé:

- Trainer/loss/dataloader riêng,import Student pinnedversion,không import train.pyupstream.
- Exporter riêng với split train|dev tường minh,Testmặc địnhkhóa;RGB function theo B3 và parity tests trên ảnh thực. Method=sage_ft_selfsup_v1.
- train_seg.py --bien_the không có choiceshạn chế nên đọc được candidate PNG đã tạo sẵn. SAGE_ANH/SAGE_RUNS trỏ root mới,không viết vào outputs/runs cũ.
- SAGE_FMB=D:\SAGE_Research\FMB;SAGE_MAN trỏ manifest đúnghash. Common hỗ trợ train/train,test/test nhưng fusionDevelopmentloader riêng phải chặn Test,không gọi dòTest chỉ để nạpTrain.
- Eval và analysis runner mới phải phân biệt Development/Final,không sửa Mân.

Ví dụ **chỉ để bàn giao,CHƯA THỰC THI** cho final downstream,sau khi config/resources đã được duyệt:

~~~powershell
$env:SAGE_FMB = 'D:\SAGE_Research\FMB'
$env:SAGE_MAN = 'D:\SAGE_Research\xulianh-finetune\data\manifests'
$env:SAGE_ANH = '<new-artifact-root>\b3_inputs'
$env:SAGE_RUNS = '<new-artifact-root>\b3_runs'
& 'D:\SAGE_Research\xulianh\.venv-sage\Scripts\python.exe' 'D:\SAGE_Research\xulianh-b3-review\src\segmentation\train_seg.py' --bien_the sage_ft_selfsup_v1 --model b3 --seed 0 --gop_dev --lop0 --epochs 100 --batch 8 --tich_luy 1
~~~

Thay placeholder bằng path artifact mới,không FMB hoặc checkpoint cũ; seeds1/2 chỉ khi được giao. Một command đúng không bảo đảm hardwarefit hoặc initial tensor parity.

so_sanh_b3.py có BT/CAP hard-coded và top-level execution:không import/run để “thêm method”,không sửa reference. Sau này viết analysis runner riêng nhậnmethods/paths,dùng cùng chung.cac_miou và blockmetadata/bootstrap;replay control tables từ confusion trước. tong_hop.py/eval có ghiCSV/predictions nên không chạy vào artifact cũ.

## 5. Recipe B3 và checkpoints

MiT-B3 encoder pretrained ImageNet,head15class khởi tạo theo seed,train FMB. Cùng seed giữa methods phải đi kèm từng tensor initial state giống nhau;không fine-tune từ VIS/SAGE B3 đã học. Lưu HF revision/weight SHA256,torch/transformers và init state hash. MD5last weights trong artifact manifest không chứng minh initialization tensorparity;Mân cần cung cấp config/init artifact để audit.

Final recipe:1220images,100epochs,scale.5–2,crop512,hflip.5,batch8,accum1,workers2,seeded shuffle/worker init,drop_lastTrue. AdamWwd.01,encoderLR6e-5/headLR6e-4,warmup1500,polypower1,AMPfloat16/GradScaler.152batches/epoch,15200updates,last.pt. Khác Studentfusion FP32/crop192×256.

Manifest B3 có12last.pt (VIS/IR/TB/SAGE×seed0/1/2),full MD5 trong [artifact_manifest.json](https://github.com/quan14333/xulianh/blob/58a24847ac68aca6f232194cf1bc84d9fcf4be52/reports/experiments/b3_fmb_seg/artifact_manifest.json). Weights189MB/file và confusion/predictions ở private artifacts;không local/không tải lần này. Báo cáo ghi~2.6h/run T4,không dùng như RTX5060ETA.

B3batch8crop512 trên8GB chưa dry-run. Accumulation không tự tương đương mọi chi tiết(batchnorm/dropout,RNG/drop_last,scheduler). Nếu đổi recipe phải matchedcontrols mới hoặc bàn giao cho Mân trên resources đã xác nhận,không tự dùng Kaggle.

## 6. Định nghĩa primary mIoU

C15×15 pooled confusion,hàngGT/cộtprediction. Với j=1..14:

$$t_j=C_{jj},\quad g_j=\sum_{k=0}^{14}C_{jk},\quad p_j=\sum_{i=1}^{14}C_{ij},\quad u_j=g_j+p_j-t_j.$$

$$IoU_j=100t_j/u_j\ (g_j>0),\qquad mIoU_{co\_mat}=\operatorname{mean}_{j:g_j>0}IoU_j.$$

LoạiGT row0 nhưng giữ prediction0 trongg_j để vẫn tính lỗi. Class khôngGT nhưng bịpredict không vào meanprimary;khác union-present. Pooled confusion trước rồi IoU,không average per-image mIoU thànhoverall.

| ID | Tên | ID | Tên | ID | Tên |
|---:|---|---:|---|---:|---|
|0|unlabeled|5|traffic_sign|10|truck|
|1|road|6|vegetation|11|bus|
|2|sidewalk|7|sky|12|motorcycle|
|3|building|8|person|13|bicycle|
|4|traffic_light|9|car|14|pole|

PersonID8, index7 khi14foreground. Final report giữ primary 273exclusion đã khóa,kèm280full,mIoU14fixed,mIoU_15_nan0,mIoU_15_mrfs và support. Không tự loại ảnh/classes theo điểm candidate. Đây là thiết kế metric,không chạy Test.

Bootstrap ghép đúngIDs/cùngblock resamples qua methods/seeds. Pooled seed image bootstrap không bao training seed uncertainty;SD giữa seeds riêng. Brightnessbins từ VIS và thresholds cùng protocol;không trộn tercilesDev/B0 với273Test/B3.

## 7. Acceptance và handoff

Artifact sau implementation:source/lossformula,canonical config/hash,protocolID,manifestsource/unionhash,pretrained/candidateSHA256,environment,history/selection,init verification,2batchdryrunloss/grad/VRAM/ETA,100%PNG QA/per-image hash,commands và limitations. Artifacts lớn ngoài Git.

Technical PASS:strictload,IDs/shape/range/PNGđúng,noNaN,reference bất biến,noTest access,provenance và export parity. Effectiveness riêng:deltafusionmetrics/matchedmIoU/person/dark thựcđo;chưa đo=NOT_MEASURED,không tự điền0.

Frozen SAGE và TB là controls bắt buộc. Candidate không vượtTB thì chưa chứng minh learned fusion có giá trị hơn averaging;mộtseed tốt chỉ evidence banđầu,không novelty/generalization. Công bố hai protocols và việc đã biết priorTestresults.

Xem [audit](FINETUNE_AUDIT.md),[proposal](BASELINE_PROPOSAL.md),[decision register](CONFIG_DECISIONS.md).

## 8. Danh mục checkpoint B3 theo manifest đã đọc

Chỉ xác minh metadata Git; chưa xác minh file .pt local.

| Run ID | Seed | Checkpoint | MD5 theo manifest |
|---|---:|---|---|
| ir_b3_s0_1220_15lop | 0 | last.pt | 68512c087fd201300b63dfce5b584c5e |
| ir_b3_s1_1220_15lop | 1 | last.pt | f1fd66f264b84bb0211f40201d2737c1 |
| ir_b3_s2_1220_15lop | 2 | last.pt | a8f19c4616fdd81527a9757a89e3c510 |
| sage_b3_s0_1220_15lop | 0 | last.pt | c736af3a0f55968c213986fb69bacb48 |
| sage_b3_s1_1220_15lop | 1 | last.pt | e62e43a87c36fd28cc0ad9ebffd865fb |
| sage_b3_s2_1220_15lop | 2 | last.pt | 63b1b453ddf85297887697c50fb746f9 |
| tb_b3_s0_1220_15lop | 0 | last.pt | 0e8578e93b3be0122e8ca33501f87cf6 |
| tb_b3_s1_1220_15lop | 1 | last.pt | c58ed79cc0779c1307227bd9b83d61cc |
| tb_b3_s2_1220_15lop | 2 | last.pt | cec18cf7270b8f68c0894ebdff236135 |
| vis_b3_s0_1220_15lop | 0 | last.pt | 1785b91752cdf11363862981af92529b |
| vis_b3_s1_1220_15lop | 1 | last.pt | 4a46786b22342f00d4979ed93a8ba8e0 |
| vis_b3_s2_1220_15lop | 2 | last.pt | c482d5e6c0e6c92446929775609329d9 |
