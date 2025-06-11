# DCNv2 CPU Troubleshooting Guide

## Common Issues and Solutions

### 1. Compilation Errors

#### Error: `fatal error: 'ATen/ATen.h' file not found`
```bash
# Reinstall PyTorch with proper headers
conda uninstall pytorch
conda install pytorch torchvision torchaudio cpuonly -c pytorch

# Or rebuild with verbose output to debug
python setup.py build develop --verbose
```

#### Error: C++ compiler issues on macOS
```bash
# Ensure Xcode command line tools are installed
xcode-select --install

# Check compiler
which g++
g++ --version
```

### 2. Runtime Errors

#### Error: `ModuleNotFoundError: No module named '_ext'`
```bash
# Clean and rebuild
python setup.py clean --all
rm -rf build/
export CUDA_HOME=""  # Force CPU build
python setup.py build develop
```

#### Error: `CUDA kernel launch failed`
```bash
# Verify CPU-only environment
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
# Should print: CUDA available: False
```

### 3. Performance Issues

#### Slow Performance
```bash
# Optimize CPU performance
export OMP_NUM_THREADS=8  # Adjust based on your CPU cores
export MKL_NUM_THREADS=8

# Use optimized BLAS
conda install mkl mkl-include

# Check CPU utilization during tests
top -pid $(pgrep Python)
```

### 4. Memory Issues

#### Out of Memory Errors
```python
# Reduce batch size for testing
input_tensor = torch.randn(1, 64, 64, 64)  # Smaller batch

# Monitor memory usage
import psutil
process = psutil.Process()
print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.1f} MB")
```

## Validation Checklist

### Environment Setup
- [ ] Xcode command line tools installed
- [ ] PyTorch CPU version installed
- [ ] DCNv2 compiled without CUDA

### Basic Functionality
- [ ] Forward pass works without errors
- [ ] Output shapes are correct
- [ ] No NaN or infinite values in output

### Gradient Computation
- [ ] Backward pass completes successfully
- [ ] Gradients are computed for all parameters
- [ ] Gradient check passes (numerical verification)

### Equivalence Tests
- [ ] Zero offset equals standard convolution
- [ ] Manual offset/mask control works
- [ ] Results are deterministic (same input -> same output)

### Performance
- [ ] Reasonable execution time for your use case
- [ ] Memory usage is acceptable
- [ ] CPU utilization is efficient

## Performance Expectations

| Input Size | CPU Time | Expected GPU Time | Ratio |
|------------|----------|-------------------|--------|
| 64x64      | ~10ms    | ~1ms             | 10x    |
| 128x128    | ~40ms    | ~2ms             | 20x    |
| 256x256    | ~160ms   | ~5ms             | 32x    |

*Note: Times are approximate and depend on hardware*

## When to Use CPU vs GPU

### Use CPU Implementation When:
- 🔬 **Development/Testing**: Quick iteration without GPU dependency
- 🖥️ **Deployment**: CPU-only servers or edge devices
- 🧪 **Research**: Small-scale experiments and algorithm validation
- 🐛 **Debugging**: Easier to debug CPU code with standard tools

### Consider GPU When:
- 🚀 **Production**: Large-scale inference or training
- ⏱️ **Performance Critical**: Real-time applications
- 📊 **Large Models**: High-resolution images or large batch sizes
- 💰 **Cost Effective**: GPU compute is often more cost-efficient at scale