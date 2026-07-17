
import os, sys
os.chdir(r"/Users/ostino/Documents/Code/Projector_studies/procam-test-data/runs/flatwall_20260717_141610/calibration/cspr_work")
sys.argv = ["train_adapted.py", "inference_custom", r"/Users/ostino/Documents/Code/Projector_studies/procam-test-data/runs/flatwall_20260717_141610/projected_validation/content_validation.png"]
# Use original adapted file if present
ns = {}
path = r"/Users/ostino/Documents/Code/Projector_studies/procam-test-data/runs/flatwall_20260717_141610/calibration/cspr_work/train_adapted.py"
code = open(path).read().replace('if __name__ == "__main__":', 'if False:')
exec(compile(code, path, "exec"), ns)
def preprocess_and_crop(img_path):
    import cv2
    img = cv2.imread(img_path)
    out = img_path.replace("_origin.jpeg", ".jpeg")
    if out == img_path:
        out = img_path + ".nocrop.jpeg"
    cv2.imwrite(out, img)
    return out
ns["preprocess_and_crop"] = preprocess_and_crop
warper = ns["NeuralWarper"](MODE="inference_custom")
warper.run_inference(img_path=r"/Users/ostino/Documents/Code/Projector_studies/procam-test-data/runs/flatwall_20260717_141610/projected_validation/content_validation.png")
