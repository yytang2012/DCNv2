#!/usr/bin/env python3
"""
Comprehensive DCNv2 CPU functionality test suite
"""
import torch
import numpy as np
from dcn_v2 import DCN, DCNv2, dcn_v2_conv

def test_cpu_only_environment():
    """Verify we're running in CPU-only mode"""
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CPU threads: {torch.get_num_threads()}")
    
    # Ensure all tensors are on CPU
    device = torch.device('cpu')
    print(f"Using device: {device}")
    return device

def test_dcn_forward_pass():
    """Test basic DCN forward pass"""
    print("\n=== Testing DCN Forward Pass ===")
    
    # Create test data
    batch_size, in_channels, height, width = 2, 16, 32, 32
    out_channels = 32
    
    # Initialize DCN layer
    dcn = DCN(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        deformable_groups=1
    )
    
    # Create input tensor
    input_tensor = torch.randn(batch_size, in_channels, height, width)
    
    # Forward pass
    try:
        output = dcn(input_tensor)
        print(f"✅ Forward pass successful")
        print(f"Input shape: {input_tensor.shape}")
        print(f"Output shape: {output.shape}")
        
        # Verify output properties
        assert output.shape == (batch_size, out_channels, height, width)
        assert torch.isfinite(output).all(), "Output contains non-finite values"
        assert not torch.isnan(output).any(), "Output contains NaN values"
        
        print(f"Output stats: min={output.min():.4f}, max={output.max():.4f}, mean={output.mean():.4f}")
        return True
        
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        return False

def test_dcn_backward_pass():
    """Test DCN backward pass (gradient computation)"""
    print("\n=== Testing DCN Backward Pass ===")
    
    # Create test data with gradient tracking
    input_tensor = torch.randn(2, 8, 16, 16, requires_grad=True)
    
    # Initialize DCN layer
    dcn = DCN(8, 16, kernel_size=3, stride=1, padding=1)
    
    try:
        # Forward pass
        output = dcn(input_tensor)
        
        # Create loss and backward pass
        loss = output.sum()
        loss.backward()
        
        print(f"✅ Backward pass successful")
        
        # Verify gradients exist and are finite
        assert input_tensor.grad is not None, "Input gradient is None"
        assert torch.isfinite(input_tensor.grad).all(), "Input gradient contains non-finite values"
        
        # Check parameter gradients
        for name, param in dcn.named_parameters():
            if param.grad is not None:
                assert torch.isfinite(param.grad).all(), f"Parameter {name} gradient contains non-finite values"
                print(f"Parameter {name}: grad_norm={param.grad.norm():.6f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Backward pass failed: {e}")
        return False

def test_dcnv2_manual_offset():
    """Test DCNv2 with manual offset and mask control"""
    print("\n=== Testing DCNv2 Manual Offset Control ===")
    
    # Test parameters
    batch_size, in_channels, height, width = 1, 4, 8, 8
    out_channels = 8
    kernel_size = 3
    deformable_groups = 1
    
    # Initialize DCNv2
    dcnv2 = DCNv2(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=1,
        padding=1,
        deformable_groups=deformable_groups
    )
    
    try:
        # Create input
        input_tensor = torch.randn(batch_size, in_channels, height, width)
        
        # Create offset and mask tensors
        # offset: [N, 2*deformable_groups*kH*kW, H, W]
        offset_channels = 2 * deformable_groups * kernel_size * kernel_size
        offset = torch.randn(batch_size, offset_channels, height, width) * 0.1  # Small offsets
        
        # mask: [N, deformable_groups*kH*kW, H, W]
        mask_channels = deformable_groups * kernel_size * kernel_size
        mask = torch.sigmoid(torch.randn(batch_size, mask_channels, height, width))
        
        # Forward pass with manual control
        output = dcnv2(input_tensor, offset, mask)
        
        print(f"✅ DCNv2 manual control successful")
        print(f"Input shape: {input_tensor.shape}")
        print(f"Offset shape: {offset.shape}")
        print(f"Mask shape: {mask.shape}")
        print(f"Output shape: {output.shape}")
        
        # Verify output
        expected_shape = (batch_size, out_channels, height, width)
        assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"
        assert torch.isfinite(output).all(), "Output contains non-finite values"
        
        return True
        
    except Exception as e:
        print(f"❌ DCNv2 manual control failed: {e}")
        return False

def test_zero_offset_equivalence():
    """Test that DCN with zero offset equals standard convolution"""
    print("\n=== Testing Zero Offset Equivalence ===")
    
    # Small test case for easier verification
    batch_size, channels, height, width = 1, 2, 4, 4
    kernel_size = 3
    
    try:
        # Create input
        input_tensor = torch.randn(batch_size, channels, height, width)
        
        # Initialize DCNv2 with identity weight
        dcnv2 = DCNv2(channels, channels, kernel_size, stride=1, padding=1)
        
        # Set identity weights (diagonal in channel dimension)
        with torch.no_grad():
            dcnv2.weight.zero_()
            dcnv2.bias.zero_()
            center = kernel_size // 2
            for i in range(channels):
                dcnv2.weight[i, i, center, center] = 1.0
        
        # Create zero offset and unit mask
        offset = torch.zeros(batch_size, 2*kernel_size*kernel_size, height, width)
        mask = torch.ones(batch_size, kernel_size*kernel_size, height, width)
        
        # DCN output
        dcn_output = dcnv2(input_tensor, offset, mask)
        
        # Standard convolution output
        conv_output = torch.nn.functional.conv2d(
            input_tensor, dcnv2.weight, dcnv2.bias, stride=1, padding=1
        )
        
        # Compare outputs
        diff = torch.abs(dcn_output - conv_output).max().item()
        print(f"Max difference between DCN and Conv2d: {diff:.2e}")
        
        # Should be very close (within numerical precision)
        if diff < 1e-6:
            print(f"✅ Zero offset equivalence test passed")
            return True
        else:
            print(f"❌ Zero offset equivalence test failed: difference too large")
            return False
            
    except Exception as e:
        print(f"❌ Zero offset equivalence test failed: {e}")
        return False

def test_gradient_numerical_verification():
    """Verify gradients using numerical differentiation"""
    print("\n=== Testing Gradient Numerical Verification ===")
    
    try:
        from torch.autograd import gradcheck
        
        # Small problem for gradcheck
        batch_size, in_channels, height, width = 1, 2, 3, 3
        out_channels = 2
        kernel_size = 3
        
        # Create test inputs
        input_tensor = torch.randn(batch_size, in_channels, height, width, 
                                 dtype=torch.double, requires_grad=True) * 0.01
        
        offset = torch.randn(batch_size, 2*kernel_size*kernel_size, height, width, 
                           dtype=torch.double, requires_grad=True) * 0.01
        
        mask = torch.sigmoid(torch.randn(batch_size, kernel_size*kernel_size, height, width, 
                                       dtype=torch.double, requires_grad=True))
        
        weight = torch.randn(out_channels, in_channels, kernel_size, kernel_size, 
                           dtype=torch.double, requires_grad=True) * 0.01
        
        bias = torch.randn(out_channels, dtype=torch.double, requires_grad=True) * 0.01
        
        # Test gradient
        test_passed = gradcheck(
            dcn_v2_conv,
            (input_tensor, offset, mask, weight, bias, 1, 1, 1, 1),  # stride, padding, dilation, groups
            eps=1e-4,
            atol=1e-3,
            rtol=1e-2
        )
        
        if test_passed:
            print(f"✅ Gradient numerical verification passed")
        else:
            print(f"❌ Gradient numerical verification failed")
            
        return test_passed
        
    except Exception as e:
        print(f"❌ Gradient verification failed: {e}")
        return False

def test_performance_benchmark():
    """Basic performance benchmark for CPU implementation"""
    print("\n=== Testing Performance Benchmark ===")
    
    import time
    
    # Test different input sizes
    test_cases = [
        (1, 32, 64, 64),
        (2, 64, 128, 128),
        (4, 128, 256, 256),
    ]
    
    for batch_size, channels, height, width in test_cases:
        print(f"\nTesting {batch_size}x{channels}x{height}x{width}")
        
        # Initialize DCN
        dcn = DCN(channels, channels, kernel_size=3, stride=1, padding=1)
        input_tensor = torch.randn(batch_size, channels, height, width)
        
        # Warmup
        for _ in range(3):
            _ = dcn(input_tensor)
        
        # Benchmark
        num_runs = 10
        start_time = time.time()
        
        for _ in range(num_runs):
            output = dcn(input_tensor)
        
        end_time = time.time()
        avg_time = (end_time - start_time) / num_runs
        
        # Calculate throughput
        pixels_processed = batch_size * height * width * num_runs
        pixels_per_second = pixels_processed / (end_time - start_time)
        
        print(f"  Average time per forward pass: {avg_time*1000:.2f}ms")
        print(f"  Throughput: {pixels_per_second/1e6:.2f} Mpixels/sec")

def main():
    """Run all CPU tests"""
    print("DCNv2 CPU Implementation Test Suite")
    print("=" * 50)
    
    # Test environment
    device = test_cpu_only_environment()
    
    # Run all tests
    tests = [
        test_dcn_forward_pass,
        test_dcn_backward_pass,
        test_dcnv2_manual_offset,
        test_zero_offset_equivalence,
        test_gradient_numerical_verification,
        test_performance_benchmark,
    ]
    
    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test_func.__name__} crashed: {e}")
            results.append(False)
    
    # Summary
    print(f"\n{'='*50}")
    print(f"Test Summary: {sum(results)}/{len(results)} tests passed")
    
    if all(results):
        print("🎉 All tests passed! DCNv2 CPU implementation is working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
    
    return all(results)

if __name__ == "__main__":
    main()