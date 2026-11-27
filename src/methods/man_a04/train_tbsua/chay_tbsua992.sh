#!/bin/bash
# 24/11: ứng viên tbsua trên giao thức Dev228. Student TB + phần sửa, CÙNG tham số tbsua_w150 (s 0,5; w_tb 150; w_g 27; 100 epoch; seed 0),
# train 992 ảnh (không đụng Dev228). Rào (đặt trước): MS-SSIM >= 99%, Q_ABF >= 98% SAGE; SCD chỉ báo kèm.
# ĐẶT TRƯỚC (mIoU13 Dev228 B3, seed 0,1 cùng seed; TB 60,19): TB 2 seed >= 60,9 VÀ 8 lớp phổ biến không thua TB → seed 2 → Test.
set -e
export SAGE_MAN=D:/CS406/DoAn/Sage/tai_lap/_kaggle_vnc_code/manifests SO_TRAIN=992 PYTHONUTF8=1
export SAGE_ANH=D:/CS406/DoAn/Sage/luu_tru/tim_gap_b0_0811/seg/anh
PY=D:/Miniconda3/envs/DoAn/python.exe; TL=D:/CS406/DoAn/Sage/tai_lap; LOG=$TL/scene_v1/tbsua_992.log; ten=tbsua_992
cd $TL/cai_tien
[ -f $TL/b3/sage_student/$ten.pt ] || $PY tb_sua.py --chi_train --ten $ten --w_tb 150 > $LOG 2>&1
cd $TL/b3; $PY xuat_anh.py --bien_the $ten >> $LOG 2>&1
cd $TL/cai_tien; $PY chi_so_ghep.py --bien_the sage tb nosam_992 $ten --tap dev --hau d228_tbsua --workers 4 >> $LOG 2>&1
tail -n 10 $LOG; echo XONG
