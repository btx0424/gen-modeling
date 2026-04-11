# gen-modeling

Small generative modeling experiments in PyTorch, including energy-based models, equilibrium models, flow matching, image datasets, and LAFAN1 motion data.

## Related Works

### Generative Models

- [Equilibrium Matching: Generative Modeling with Implicit Energy-Based Models
](https://arxiv.org/abs/2510.02300)

- [Is Noise Conditioning Necessary for Denoising Generative Models?
](https://arxiv.org/abs/2502.13129)

- [Back to Basics: Let Denoising Generative Models Denoise
](https://arxiv.org/abs/2511.13720)

- [Introduction to Flow Matching and Diffusion Models 2026
](https://diffusion.csail.mit.edu/2026/)


## Setup

### Option 1: `conda`

```bash
conda create -n gen-modeling python=3.11
conda activate gen-modeling
pip install -e .
```

### Option 2: `uv`

```bash
uv sync
```

Run commands with `uv run ...`, for example:

```bash
uv run python main.py
```

## Data

Image examples download datasets through `torchvision` when needed.

For LAFAN1, you can just clone the retargeted dataset into `data/`. The loader supports either:

- `data/LAFAN1_Retargeting_Dataset/<robot>/*.csv`
- `data/<robot>/*.csv`

Example:

```bash
git clone https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset data/LAFAN1_Retargeting_Dataset
```

## Usage Examples

Run a 2D synthetic experiment:

```bash
uv run python examples/main.py
```

Train flow matching on MNIST:

```bash
uv run python examples/FM_mnist.py
```

Run a LAFAN1 motion experiment:

```bash
uv run python examples/FM_lafan1.py
```

Compute dataset normalization statistics:

```bash
uv run python scripts/compute_image_stats.py --dataset mnist
```

Type-check the codebase:

```bash
uv run pyright
```
