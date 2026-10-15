# BEANet

PyTorch model implementation for paired reflectance and fluorescence hyperspectral classification.
The default configuration predicts 24 joint classes: eight bacterial species and three burden levels.

This repository contains the model and a synthetic example. Experimental data, trained weights,
dataset preparation, and the full training and evaluation pipeline are outside the scope of this release.
The example checks that the model runs; it does not reproduce the paper's reported accuracy.

## Setup

Use Python 3.10 or newer. The example has been checked with Python 3.12 and PyTorch 2.7.1.

```bash
python -m pip install -r requirements.txt
python example.py
```

To check a backward pass and optimizer step:

```bash
python example.py --backward
```

Use `--device cuda` with a CUDA-enabled PyTorch installation.

## Model

```python
import torch
from beanet import BEANet

model = BEANet(in_channels=300, num_classes=24).eval()
reflectance = torch.randn(2, 300, 16, 16)
fluorescence = torch.randn_like(reflectance)

with torch.no_grad():
    logits = model(reflectance, fluorescence)  # [2, 24]
    details = model(reflectance, fluorescence, return_diagnostics=True)
```

Inputs are preprocessed, spatially paired patches in `[batch, bands, height, width]` order.
The released configuration uses 300 bands and a fixed 16 × 16 spatial grid.
Raw sensor calibration and spectral normalization must be handled before calling the model.
The example uses randomly initialized weights and synthetic inputs.

The model contains a wavelet-based reflectance branch, a spectral-spatial fluorescence branch,
and four interaction experts. Bio-optical statistics and a gated spectral convolution encoder
condition expert routing. A separate descriptor projection weights the interaction, reflectance,
and fluorescence decision sources at the output stage. The spectral encoder uses PyTorch
convolutions and does not require a Mamba package.

Diagnostic outputs include:

| Key | Shape | Meaning |
| --- | --- | --- |
| `logits` | `[batch, 24]` | Joint-class logits |
| `gates` | `[batch, 4]` | Interaction-expert weights |
| `late_gates` | `[batch, 3]` | Weights for interaction, reflectance, and fluorescence sources |
| `qc_token` | `[batch, 32]` | Projected bio-optical descriptor |
| `embedding` | `[batch, 168]` | Features available for auxiliary objectives |

For the default task, joint labels follow `species_index * 3 + burden_index`.
Species and burden probabilities can be obtained by reshaping joint probabilities to
`[batch, 8, 3]` and summing over the corresponding axis. The species order must match the
dataset or checkpoint used by the caller.

## Files

```text
beanet/
  model.py          # Model entry point and source arbitration
  reflectance.py    # Wavelet decomposition and subband processing
  fluorescence.py   # Spectral-spatial extraction and activation blocks
  fusion.py         # Bio-optical descriptors and expert routing
  _heads.py         # Grouped spectral residual decision heads
example.py          # Synthetic forward and backward example
requirements.txt
```

Research checkpoint parameter names are retained for the released scalar source-fusion
configuration. Other experimental fusion modes and ablation variants are not included.
