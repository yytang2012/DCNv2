#!/usr/bin/env python3
"""
DCNv2 CPU Usage Example - Integration into PyTorch models
"""
import torch
import torch.nn as nn
from dcn_v2 import DCN, DCNv2

class SimpleModel(nn.Module):
    """Simple model demonstrating DCN usage"""
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, 2, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        
        # Use DCN for adaptive feature extraction
        self.dcn_layer = DCN(64, 128, 3, 1, 1, deformable_groups=2)
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        adaptive_features = self.dcn_layer(features)
        return self.classifier(adaptive_features)

class AdvancedModel(nn.Module):
    """Advanced model with manual offset/mask control"""
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, 2, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        
        # Offset and mask prediction networks
        self.offset_net = nn.Conv2d(64, 18, 3, 1, 1)  # 2 * 3 * 3 = 18 for 3x3 kernel
        self.mask_net = nn.Conv2d(64, 9, 3, 1, 1)    # 1 * 3 * 3 = 9 for 3x3 kernel
        
        # DCNv2 layer
        self.dcnv2 = DCNv2(64, 128, 3, 1, 1, deformable_groups=1)
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        
        # Predict offsets and masks
        offset = self.offset_net(features)
        mask = torch.sigmoid(self.mask_net(features))
        
        # Apply deformable convolution
        adaptive_features = self.dcnv2(features, offset, mask)
        
        return self.classifier(adaptive_features)

def demo_basic_usage():
    """Demonstrate basic DCN usage"""
    print("=== Basic DCN Usage Demo ===")
    
    # Create model and input
    model = SimpleModel(in_channels=3, num_classes=10)
    input_tensor = torch.randn(4, 3, 224, 224)
    
    # Forward pass
    output = model(input_tensor)
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")
    print("✅ Basic usage successful!")

def demo_advanced_usage():
    """Demonstrate advanced DCNv2 usage with manual control"""
    print("\n=== Advanced DCNv2 Usage Demo ===")
    
    # Create model and input
    model = AdvancedModel(in_channels=3, num_classes=10)
    input_tensor = torch.randn(2, 3, 224, 224)
    
    # Forward pass
    output = model(input_tensor)
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")
    print("✅ Advanced usage successful!")

def demo_training_loop():
    """Demonstrate training with DCN layers"""
    print("\n=== Training Loop Demo ===")
    
    # Create model, loss, optimizer
    model = SimpleModel(in_channels=3, num_classes=10)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Dummy data
    inputs = torch.randn(8, 3, 64, 64)
    targets = torch.randint(0, 10, (8,))
    
    # Training step
    model.train()
    optimizer.zero_grad()
    
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    
    print(f"Training loss: {loss.item():.4f}")
    print("✅ Training demo successful!")

def demo_inference():
    """Demonstrate inference mode"""
    print("\n=== Inference Mode Demo ===")
    
    model = SimpleModel(in_channels=3, num_classes=10)
    model.eval()
    
    # Single image inference
    with torch.no_grad():
        input_tensor = torch.randn(1, 3, 224, 224)
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1)
    
    print(f"Predicted class: {predicted_class.item()}")
    print(f"Confidence: {probabilities.max().item():.4f}")
    print("✅ Inference demo successful!")

if __name__ == "__main__":
    print("DCNv2 CPU Usage Examples")
    print("=" * 40)
    
    demo_basic_usage()
    demo_advanced_usage()
    demo_training_loop()
    demo_inference()
    
    print("\n🎉 All demos completed successfully!")