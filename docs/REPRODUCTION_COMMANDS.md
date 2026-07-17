# Reproduction Commands

Commands derived only from repository READMEs and encoded CLI.  
**M0 status:** commands documented; **M1/M2 execution pending** (do not claim success until artifacts exist).

## Environment record (host)

| Field | Value |
|-------|-------|
| OS | macOS 15.7.2 (24G325), Darwin 24.6.0 |
| Arch | arm64 (Apple Silicon) |
| Hostname | Agostinos-MacBook-Pro.local |
| `CSPR_NET_ROOT` | `/Users/ostino/Documents/Code/Projector_studies/CSPR-Net` |
| `GS_PROCAMS_ROOT` | `/Users/ostino/Documents/Code/Projector_studies/GS-ProCams` |
| CSPR commit | `a2a0a28ac2022c876f1ac1e52620b44bfe96672a` |
| GS commit | `dfb7f449710ce41a63369a447ca728e0cb220d97` |

Artifact root for future runs:  
`TEST_DATA_ROOT/repro/<repo>/<run_id>/` with `commands.txt`, `environment.txt`, stdout/stderr, outputs.

---

## M1 — CSPR-Net (unchanged)

### Install (README exact)

```bash
cd /Users/ostino/Documents/Code/Projector_studies/CSPR-Net
python3 -m venv .venv
source .venv/bin/activate
pip install numpy opencv-python matplotlib pyvista torch scikit-image Pillow
```

Optional: omit `pyvista` if skipping `qurdric_transfer.py` visualization.

### Smallest meaningful examples

**Preferred (pretrained geometric pre-warp):**

```bash
python deep_learning_simulation_for_gradient.py inference
python deep_learning_exp_for_gradient.py inference
python deep_learning_exp_for_gradient.py inference_custom ./real_data/test.jpg
```

**Simulation calibration / maps (side effects: PyVista GUI):**

```bash
python qurdric_transfer.py
```

**Pattern regen:**

```bash
python generate_ununiform_mesh.py
```

### Success criterion

- `neural_results_simulation/pre_warped.png` and/or `neural_results_exp/pre_warped_test.png` written or refreshed  
- stdout/stderr saved  
- Confirm output is geometric pre-warp (projector-space PNG), not only app startup  

### Known issues (document as patches only if applied later)

| Issue | Evidence |
|-------|----------|
| Exp default mode `inference_custom` not in README | `deep_learning_exp_for_gradient.py:828` |
| PyVista `show()` blocks | `qurdric_transfer.py` |

### Local readiness

Bundled data + `.pth` present → **M1 is unblocked** on this host (CPU OK).

---

## M2 — GS-ProCams (unchanged)

### Install (README exact)

```bash
cd /Users/ostino/Documents/Code/Projector_studies/GS-ProCams
conda env create -f environment.yml
conda activate gs-procams
pip install -r requirements.txt
pip install -e submodules/diff-gaussian-rasterization -e submodules/simple-knn
```

Requires: NVIDIA GPU, CUDA Toolkit 11.8, C++ compiler; COLMAP for real-world registration.

### Dataset (README)

Download SharePoint real-world/compensation data and/or Nepmap synthetic; place under `data/` (`README.md` L32).

### Smallest documented examples

```bash
bash ./scripts/synthetic.sh
# or
bash ./scripts/real-world.sh
# or
bash ./scripts/compensate.sh
```

Custom example (`README.md` L80–105):

```bash
python scripts/registrate.py -r 'data/compensation' -s example
python train.py -r 'data/compensation' -s example -m 'output/example'
python render.py -r 'data/compensation' -s example -m 'output/example' -o 'output/example/render' --views 26
python compensate.py -r 'data/compensation' -s example -m 'output/example' --view_id 26
```

Prefer README `-r`/`-s` form over shell scripts (scripts often pass full path in `-s` only).

### Success criterion

- Trained or rendered/compensated PNG outputs under `output/`  
- Or **exact external blocker** documented with evidence (no silent version changes)

### Local readiness (this host)

| Blocker | Evidence |
|---------|----------|
| No NVIDIA CUDA on Apple Silicon Mac | `README.md` L11–13; `requirements.txt` cu118 |
| No `data/` directory | listing empty / missing |
| No pretrained checkpoints | must train 20k iters |

→ **M2 expected outcome: document blocker** unless a CUDA machine + datasets are provided.

---

## Rules

1. Do not invent missing assets.  
2. Do not silently change dependency versions.  
3. Any patch → separate note in `report.md` of the run directory.  
4. Reproduction ≠ process started; requires calibration/render/eval artifact.
