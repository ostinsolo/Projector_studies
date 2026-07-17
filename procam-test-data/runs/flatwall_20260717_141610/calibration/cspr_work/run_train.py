
import random, numpy as np, torch, sys
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
sys.argv = ["train_adapted.py", "train"]
# Execute adapted script as __main__
g = {"__name__": "__main__", "__file__": r"/Users/ostino/Documents/Code/Projector_studies/procam-test-data/runs/flatwall_20260717_141610/calibration/cspr_work/train_adapted.py"}
with open(r"/Users/ostino/Documents/Code/Projector_studies/procam-test-data/runs/flatwall_20260717_141610/calibration/cspr_work/train_adapted.py") as f:
    code = f.read()
# Re-apply preprocess_and_crop override after class defs by exec in two stages is hard;
# instead monkeypatch after import-like exec without running main, then run.
ns = {"__name__": "cspr_adapt_mod", "__file__": r"/Users/ostino/Documents/Code/Projector_studies/procam-test-data/runs/flatwall_20260717_141610/calibration/cspr_work/train_adapted.py"}
exec(compile(code.replace('if __name__ == "__main__":', 'if False and __name__ == "__main__":'), r"/Users/ostino/Documents/Code/Projector_studies/procam-test-data/runs/flatwall_20260717_141610/calibration/cspr_work/train_adapted.py", "exec"), ns)
# Override preprocess in module ns
def preprocess_and_crop(img_path):
    import cv2 as _cv
    img = _cv.imread(img_path)
    out = img_path.replace("_origin.jpeg", ".jpeg")
    if out == img_path:
        out = img_path + ".nocrop.jpeg"
    _cv.imwrite(out, img)
    print(f"[adapt] uncropped {out} {img.shape[1]}x{img.shape[0]}")
    return out
ns["preprocess_and_crop"] = preprocess_and_crop
Config = ns["Config"]
Config.TRAIN_PAIRS = [
    ("./real_data/1_image_distorted_origin.jpeg", "./real_data/1_pro.png"),
    ("./real_data/2_image_distorted_origin.jpeg", "./real_data/2_pro.png"),
    ("./real_data/3_image_distorted_origin.jpeg", "./real_data/3_pro.png"),
]
Config.MASK_EXTRACT_PATH = "./real_data/red_distorted_origin.jpeg"
Config.OUTPUT_DIR = "./neural_results_exp"
Config.H_CAM, Config.W_CAM = 1440, 1920
Config.H_PROJ, Config.W_PROJ = 1080, 1920
Config.TRAIN_SCALE = 0.25
Config.H_TRAIN = int(Config.H_PROJ * Config.TRAIN_SCALE)
Config.W_TRAIN = int(Config.W_PROJ * Config.TRAIN_SCALE)
Config.ITERS = 2500
import os
os.chdir(r"/Users/ostino/Documents/Code/Projector_studies/procam-test-data/runs/flatwall_20260717_141610/calibration/cspr_work")
warper = ns["NeuralWarper"](MODE="train")
warper.train()
