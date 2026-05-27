from pathlib import Path
from rknnlite.api import RKNNLite

MODEL_PATH = Path("/home/marvsmart/animal_patrol/models/best_fp.rknn")

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"RKNN 模型不存在: {MODEL_PATH}")

rknn = RKNNLite()

print("--> load rknn model")
ret = rknn.load_rknn(str(MODEL_PATH))
print("load_rknn ret =", ret)
if ret != 0:
    raise RuntimeError(f"load_rknn failed, ret={ret}")

print("--> init runtime")
ret = rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
print("init_runtime ret =", ret)
if ret != 0:
    raise RuntimeError(f"init_runtime failed, ret={ret}")

print("RKNN runtime init success")

rknn.release()
print("done")
