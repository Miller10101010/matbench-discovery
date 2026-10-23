# DeePMD

DPA3 is an advanced interatomic potential leveraging the message-passing architecture, implemented within the DeePMD-kit framework, available on [GitHub](https://github.com/deepmodeling/deepmd-kit/tree/dpa3-alpha).

## DeePMD-DPA3

### Model

- DPA-3.1-3M-FT：model pretrained on the `OpenLAM` datasest (using DPA-3.1-3M from [DPA3-paper](https://arxiv.org/abs/2506.01686)), and finetuned with `MPtraj` & `Sub-Alex` datasets.

  ```bash
  wget https://figshare.com/files/55141895
  ```

- DPA-3.1-MPtrj：model trained only on the `MPtraj` dataset.

  ```bash
  wget https://figshare.com/files/55141124
  ```

#### How to install

```bash
uv pip install git+https://github.com/deepmodeling/deepmd-kit@devel
```

#### How to use

```py
from ase import Atoms
from deepmd.calculator import DP

water = Atoms(
    "H2O",
    positions=[(0.7601, 1.9270, 1), (1.9575, 1, 1), (1.0, 1.0, 1.0)],
    cell=[100, 100, 100],
    calculator=DP(model="dpa-3.1-mptrj.pth"),
)

print(water.get_potential_energy())
print(water.get_forces())
```

### Results

Discovery metrics are recorded in each model YAML. Relaxed structures are available
from Figshare:

```sh
# DPA-3.1-3M-FT
wget https://figshare.com/files/55141109

# DPA-3.1-MPtrj
wget https://figshare.com/files/55141127
```

## DeePMD-DPA4

DPA4 is an SE(3)-equivariant interatomic-potential architecture built on an EMFA
(Edge-conditioned, Multi-Focus, Attention) SO(2)-equivariant convolution. It
combines a low-rank edge–node SO(2)-equivariant product, a multi-focus design
for message nonlinearity, and envelope-gated attention for message aggregation.
A Lebedev-grid projection preserves SO(3)-equivariance in the nonlinearity to
machine precision, and a compiler-friendly conservative energy-gradient training
path gives up to ~3× wall-clock speedup under `torch.compile`. See the
[DPA4 paper](https://arxiv.org/abs/2606.02419).

### DPA-4.0.1-Pro-MPtrj

DPA-4.0.1-Pro-MPtrj is the DPA4-Pro model trained only on MPtrj for this
submission; it supersedes DPA-4.0-Pro-MPtrj. Following the DPA4-Pro column of
Table S-17 in the paper, the model parameters are: feature dim 64, 6 interaction
layers, 2 focuses, 4 SO(2) layers, a rank-2 per-channel edge–node product, a
Bessel radial basis with 16 bases, lmax=5, mmax=1, Lebedev quadrature, SiLU+GLU
activations, a 6.0 Å cutoff, and up to 384 selected neighbors (~22.8M
parameters). It was trained for 2×10⁶ steps with the HybridMuon optimizer, a WSD
learning-rate schedule, MAE energy/force/virial loss weights 20/20/5, bf16 AMP,
TF32 matmul, and `torch.compile` enabled.

### DPA-4.1-OAM inference

The three DPA-4.1 model cards are review drafts. At the submitter's direction,
authors, trained-by information, the CC-BY-4.0 checkpoint license, training task
and targets follow the DPA-4.0.1-Pro-MPtrj card. The submitter confirms training
on MPtrj, OMat24 and sAlex, with OSOD openness. Public artifact URLs remain
pending; schema validity alone does not establish publication readiness.
Known Air/Max short-range diatomic anomalies and frozen/eager precision
limitations are recorded in the cards and retained in the metrics.

Model-specific hyperparameters come from the original checkpoints, using only
keys already present in the MPtrj reference card. The original optimizer state
identifies HybridMuon with `muon_mode: slice` and `weight_decay: 0.001` for all
three models. Their checkpoint steps are 446,624 (Air), 314,359 (Max) and
217,103 (C Air); these identify the selected checkpoints, not a planned total
training schedule. GPU counts, training hours, loss weights, batch sizes and
learning-rate schedules were not saved in the supplied checkpoints and are
omitted. The reference model's training costs and hyperparameter values are
not copied. The full model configuration remains embedded in each released
inference checkpoint after removal of optimizer, auxiliary EMA and training
history.

The DPA-4.1-Air-OAM and DPA-4.1-Max-OAM checkpoints use DeePMD's `--pt`
export backend. DPA-4.1C-Air-OAM uses `--pt-expt`. Their shared calculator
exports the original training checkpoint to a `.pt2` archive. The pinned DPA4
exporter produces an `edge_vec` artifact and uses `DeepPot`'s native neighbor
list builder. DPA4C produces a graph artifact and uses the native cell-list
graph builder. ASE energy, force, and stress conversion uses DeePMD's
`DP.calculate` implementation.

The DPA4 models use `models/deepmd/dpa41-env` with PyTorch `2.11.0+cu128`;
their export pipeline requires PyTorch 2.11–2.13. DPA4C uses
`models/deepmd/dpa41c-env` with PyTorch `2.10.0+cu126`. Both environments pin
DeePMD commit `b94c0043eac81c06c123563295875055e37c43ae` and Python 3.14.
They build DeePMD from source and require Linux x86-64, a CUDA toolkit with
`nvcc`, and a compatible C++ compiler. Select the target GPU before building;
native extensions target the visible GPU architecture.

These Python 3.14 projects passed isolated Linux host installation, dependency
compatibility checks, GPU imports, Cu/H2 force/stress finite differences and
translation checks, and energy comparisons on 12 historical WBM structures per
model. The shared MD runner passed a 200-step smoke test and the diatomic runner
passed an H/He/Li smoke test for each model. Air and C used fresh exports of the
inference-only release checkpoints; Max reused its previously hash-verified
same-source/PyTorch A800 export. These bounded checks do not establish numerical
parity for every diatomic curve or reproduce a complete benchmark campaign.

Installation used current-platform `uv pip install --project <project> --python
<environment>/bin/python -r <project>/pyproject.toml`, with the matching PyTorch
CUDA index and a package mirror/cache. The server could not download the
canonical GitHub archive URL, so the validation project used an official
codeload archive with identical verified source contents; the repository path
was made absolute. Dependency declarations and build settings were preserved.
Canonical-URL `uv sync` and universal lockfile generation remain unverified.
The A800 validation does not reproduce the recorded RTX PRO 6000 costs.

The reused Air/Max MD and diatomic measurements were
made with Python 3.13, PyTorch 2.13.0+cu130 and archived DeePMD revisions on
NVIDIA RTX PRO 6000 Blackwell Server Edition. Their recorded runtimes describe
those historical runs, not the candidate environments below.

DPA-4.1C-Air-OAM retains physical MD/diatomic statistics from the validated
A800 study. Its replacement costs come from a complete RTX PRO 6000 rerun:
3,029.08 s summed over 17 single-GPU MD systems, and 29.59 s for all 92
diatomic curves. The timing campaign scheduled independent systems on five
GPUs; the reported sum is not the campaign's elapsed wall time. This rerun used
Python 3.13.15, PyTorch 2.13.0+cu130 and DeePMD
`de3fc8f62fe8879a9803d11b39f30ac789c5793f` with uncommitted modifications
preserved in the source workspace's `runtime/dpa4c-timing-20261022/` snapshot.
That revision alone does not reproduce the measured runtime. The model card
records the separate origins of physical statistics and cost measurements.

From the repository root, validate a local checkpoint with the shared runner:

```sh
CUDA_VISIBLE_DEVICES=0 DP_TRITON_INFER=1 DP_CUDA_INFER=1 \
  uv run --project models/deepmd/dpa41-env python \
  models/run_md.py --model dpa-4.1-max-oam \
  --checkpoint /path/to/checkpoint.pt --dry-run
```

Use the same environment with `models/run_diatomics.py` for diatomic curves.
Both runners accept an already validated `.pt2` through `--checkpoint`.
The example selects level-1 Triton inference; keep those settings for both
export and evaluation when reproducing that configuration.
Omit `--dry-run` for a full benchmark. `--print-cmd` prints the isolated
command using the model card's environment. Public checkpoint downloading
uses the model card's `checkpoint_url` when no override is supplied.

First-use export can take substantially longer than inference, particularly
for the Max model's kernel tuning. Exports are cached by checkpoint content,
backend, GPU model and compute capability, kernel policy, and package versions.
Set `DP_TRITON_INFER` and `DP_CUDA_INFER` before export to select supported
kernel policies; the selected policy is baked into the archive. Keep it fixed
and record it when reporting timings. Re-export for a different GPU model or
runtime. DeePMD selects its device at import time: use `CUDA_VISIBLE_DEVICES`
to select one GPU, or set `DEVICE=cpu` before importing for CPU inference.
