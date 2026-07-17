# Reproduction Plan

Exact install and run commands derived from each repository. Do not invent flags or paths beyond what README or scripts encode. Notes call out README vs code mismatches.

Working root for commands below: `Projector_studies/` (or absolute paths to each clone).

---

## A. CSPR-Net

### A.1 Install

From `CSPR-Net/README.md`:

```bash
cd CSPR-Net
pip install numpy opencv-python matplotlib pyvista torch scikit-image Pillow
```

Notes:

- No Python version pin; use a recent 3.x with a matching `torch` wheel.
- PyVista is only needed for the interactive visualization in `qurdric_transfer.py`; omit if skipping that step (README note).
- CUDA optional — code selects CPU when CUDA is unavailable.

### A.2 Run (bundled data + weights already in clone)

**Pattern generation** (described in README; no explicit command — script has no CLI):

```bash
python generate_ununiform_mesh.py
```

**Ray-tracing simulation** (README):

```bash
python qurdric_transfer.py
```

Note: ends with PyVista `plotter.show()` for pattern index 3 — requires a display / may block.

**Neural simulation** (README):

```bash
python deep_learning_simulation_for_gradient.py train
python deep_learning_simulation_for_gradient.py inference
```

Code default if no arg: `inference` (`deep_learning_simulation_for_gradient.py:508`).

**Neural experimental** (README):

```bash
python deep_learning_exp_for_gradient.py train
python deep_learning_exp_for_gradient.py inference
```

**Code-only mode** (not in README; encoded in `__main__`):

```bash
python deep_learning_exp_for_gradient.py inference_custom
python deep_learning_exp_for_gradient.py inference_custom ./real_data/test.jpg
```

Code default if no arg: `inference_custom` (`deep_learning_exp_for_gradient.py:828`).

### A.3 Expected artifacts

| Command | Writes |
|---------|--------|
| `generate_ununiform_mesh.py` | `sim_data/{1,2,3}_pro.png`, `real_data/{1,2,3}_pro.png`, `real_data/4_proj.png`, `data/generated_meshes.png` |
| `qurdric_transfer.py` | `data/{1,2,3}_image_*.png`, `sim_data/*_image_distorted.png` |
| sim inference | `neural_results_simulation/pre_warped.png` etc. |
| exp inference / custom | `neural_results_exp/pre_warped.png`, `pre_warped_test.png` |

### A.4 Immediate-run status (this machine)

**Can run now:** clone includes `sim_data/`, `real_data/`, `data/`, and pretrained `net_*.pth` under both `neural_results_*` directories. Inference commands above are the fastest verification path.

---

## B. GS-ProCams

### B.1 Install

From `GS-ProCams/README.md` (exact):

```bash
# 1. Clone the repo
git clone https://github.com/RealQingyue/GS-ProCams.git

# 2. Create a python environment
conda env create -f environment.yml
conda activate gs-procams

# 3. Install packages
pip install -r requirements.txt
pip install -e submodules/diff-gaussian-rasterization -e submodules/simple-knn
```

Requirements implied by README:

- CUDA-capable GPU
- CUDA Toolkit 11.8
- C++ compiler for PyTorch extensions (MSVC VS2019 mentioned for Windows)
- COLMAP installed locally for custom / real-world registration

### B.2 Download datasets (required before pipelines)

From README L32: download real-world / compensation data from the SharePoint link on the project page, and optionally Nepmap synthetic data from https://github.com/yoterel/nepmap — place under `data/`.

**This clone has no `data/` or `output/`.** Pipelines cannot run until datasets are present.

### B.3 Ready-made pipelines (README L35–42)

```bash
cd GS-ProCams
bash ./scripts/synthetic.sh
bash ./scripts/real-world.sh
bash ./scripts/compensate.sh
```

Windows: use the corresponding `.ps1` scripts (README L44).

**Caveat:** these shell scripts pass `train.py -s <path>` without `-r` in several places (`synthetic.sh` L71, `real-world.sh` L81, `compensate.sh` L65), which may not match `train.py`’s expected `root` + `setup` join. Prefer the documented Python form in B.4 if scripts fail.

### B.4 Custom-setup commands (README L80–105)

```bash
python scripts/registrate.py -r 'data/compensation' -s example
python train.py -r 'data/compensation' -s example -m 'output/example'
python scripts/registrate.py -r 'data/compensation' -s example --view_id 26
python render.py -r 'data/compensation' -s example -m 'output/example' -o 'output/example/render' --views 26
python compensate.py -r 'data/compensation' -s example -m 'output/example' --view_id 26
```

When compensating, if the desired folder is missing, check README path `cam/desired/test` vs code default `cam/desire/test` (`compensate.py:36`) and pass `--desired` explicitly.

### B.5 Immediate-run status (this machine)

**Cannot run end-to-end now:**

1. No NVIDIA CUDA stack assumed for macOS host
2. No datasets under `data/`
3. No pretrained checkpoints under `output/`
4. CUDA extensions must compile successfully

---

## C. Verification checklist

| Step | Repo | Pass criterion |
|------|------|----------------|
| Pip install | CSPR-Net | Imports succeed |
| Sim inference | CSPR-Net | Writes `neural_results_simulation/pre_warped.png` |
| Exp custom inference | CSPR-Net | Writes `neural_results_exp/pre_warped_test.png` |
| Conda + CUDA build | GS-ProCams | `import diff_surfel_rasterization` and `simple_knn` succeed |
| Dataset layout | GS-ProCams | `data/.../setups/...` exists |
| Train one setup | GS-ProCams | `point_cloud/iteration_20000/point_cloud.ply` exists |
| Compensate | GS-ProCams | `prj/cmp/.../patterns/*.png` exist |
