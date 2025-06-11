# DCNv2 CPU Version

CPU implementation of DCNv2 (Deformable ConvNets v2) for systems without GPU support.

## Requirements

- **Python**: 3.7+
- **PyTorch**: 1.11+ (CPU version)
- **Xcode Command Line Tools** (macOS)

## Installation

### 1. Install Dependencies

```bash
# Install Xcode command line tools (macOS only)
xcode-select --install

# Create virtual environment
python -m venv dcnv2-cpu
source dcnv2-cpu/bin/activate

# Install PyTorch CPU version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 2. Build DCNv2

```bash
# Clone repository
git clone https://github.com/yytang2012/DCNv2.git
cd DCNv2

# Build CPU version
export CUDA_HOME=""
python setup.py build develop
```

### 3. Verify Installation

```bash
python testcpu.py
```

## Quick Test

```bash
# Basic functionality test
python -c "
import torch
from dcn_v2 import DCN
dcn = DCN(3, 16, 3, 1, 1)
x = torch.randn(1, 3, 32, 32)
y = dcn(x)
print(f'✅ DCNv2 CPU working: {x.shape} -> {y.shape}')
"
```

## Additional Resources

- **Examples**: See `examples/` folder for detailed usage examples
- **Original Paper**: [Deformable ConvNets v2](https://arxiv.org/abs/1811.11168)

