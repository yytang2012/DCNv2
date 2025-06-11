# DCNv2 CPU Examples

This folder contains detailed examples and guides for using DCNv2 CPU implementation.

## Files

### Test Scripts
- **`test_cpu_comprehensive.py`** - Comprehensive test suite for validating DCNv2 CPU implementation
- **`usage_example.py`** - Basic and advanced usage examples showing integration into PyTorch models

### Documentation
- **`troubleshooting_guide.md`** - Common issues and solutions for DCNv2 CPU
- **`README.md`** - This file

## Running Examples

### Comprehensive Test Suite
```bash
cd examples
python test_cpu_comprehensive.py
```

This will run:
- Environment verification
- Forward/backward pass tests
- Manual offset/mask control tests
- Zero offset equivalence tests
- Gradient numerical verification
- Performance benchmarks

### Usage Examples
```bash
cd examples
python usage_example.py
```

This demonstrates:
- Basic DCN integration
- Advanced DCNv2 with manual control
- Training loop example
- Inference mode example

## Quick Tests

### Basic Functionality Test
```bash
python -c "
import torch
from dcn_v2 import DCN
dcn = DCN(3, 16, 3, 1, 1)
x = torch.randn(1, 3, 32, 32)
y = dcn(x)
print(f'✅ DCNv2 CPU working: {x.shape} -> {y.shape}')
"
```

### Gradient Test
```bash
python -c "
import torch
from dcn_v2 import DCN
dcn = DCN(3, 16, 3, 1, 1)
x = torch.randn(1, 3, 32, 32, requires_grad=True)
y = dcn(x)
loss = y.sum()
loss.backward()
print(f'Input gradient shape: {x.grad.shape}')
print('✅ Gradient test passed!')
"
```

## Architecture Details

### Core Components
- **`dcn_v2_cpu.cpp`**: Main forward/backward pass coordination
- **`dcn_v2_im2col_cpu.cpp`**: Core deformable convolution algorithms
- **CPU Kernel Functions**:
  - `modulated_deformable_im2col_cpu_kernel`: Forward pass
  - `modulated_deformable_col2im_cpu_kernel`: Backward pass (data gradients)
  - `modulated_deformable_col2im_coord_cpu_kernel`: Backward pass (offset/mask gradients)

### Key Differences from CUDA Version
- **No Thread Parallelization**: CPU version uses sequential loops instead of CUDA thread blocks
- **Memory Layout**: Simplified memory access patterns optimized for CPU cache
- **Bilinear Interpolation**: CPU-optimized `dmcn_im2col_bilinear_cpu` function
- **No Atomic Operations**: Direct memory writes instead of `atomicAdd`