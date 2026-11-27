"""tbsua_a04 = TB + 0,4*(tbsua_992 - TB) (alpha chọn trên Train theo luật đặt trước). 992 train + Dev228, CrCb của VIS."""
import sys; sys.path.insert(0, "D:/CS406/DoAn/Sage/tai_lap/b3")
import cv2, numpy as np, pandas as pd
from chung import ANH, MAN, duong, thu_muc
a, ten = .4, "tbsua_a04"
df = pd.concat([pd.read_csv(MAN / f"{t}.csv", dtype={"image_id": str}) for t in ("train", "dev")]); assert len(df) == 1220
d = []
for r in df.itertuples():
    ycc = cv2.cvtColor(cv2.imread(str(duong(r.vis_path))), cv2.COLOR_BGR2YCrCb); ir = cv2.imread(str(duong(r.ir_path)), 0).astype(np.float32)
    u = cv2.cvtColor(cv2.imread(str(ANH / "tbsua_992" / thu_muc(r) / f"{r.image_id}.png")), cv2.COLOR_BGR2YCrCb)[..., 0].astype(np.float32)
    t = (ycc[..., 0].astype(np.float32) + ir) / 2; y = (t + a * (u - t)).round().clip(0, 255).astype(np.uint8)
    o = ANH / ten / thu_muc(r) / f"{r.image_id}.png"; o.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(o), cv2.cvtColor(np.dstack([y, ycc[..., 1:]]), cv2.COLOR_YCrCb2BGR)); d.append((np.abs(y - t).mean(), (y >= 250).mean() * 100))
print(ten, len(d), "|Y-TB|", np.mean(d, 0).round(3))
