# DeiT Vision Transformer on CIFAR-10

A PyTorch implementation of Data-efficient Image Transformers (DeiT) for image classification on the CIFAR-10 dataset. This project demonstrates the application of Vision Transformers with distillation tokens for efficient training on smaller datasets.

## 📋 Project Overview

This implementation features a DeiT (Data-efficient Image Transformer) architecture trained on CIFAR-10. DeiT introduces a distillation token alongside the traditional class token, enabling knowledge distillation and more efficient training. The model uses patch-based image processing and multi-head self-attention mechanisms to classify images into 10 categories.

## ✨ Features

- **Dual-Token Architecture**: Implements both classification and distillation tokens for improved learning
- **Patch-Based Processing**: Converts 32×32 CIFAR-10 images into 8×8 patches
- **Multi-Head Self-Attention**: 24 attention heads with 768-dimensional embeddings
- **Transformer Blocks**: 6 transformer encoder blocks with LayerNorm and feedforward layers
- **GELU Activation**: Uses Gaussian Error Linear Units for non-linearity
- **GPU Accelerated**: Automatic CUDA device detection and utilization

## 🔧 Installation

### Prerequisites

- Python 3.7+
- CUDA-compatible GPU (recommended)

### Dependencies

Install the required packages:

```bash
pip install torch torchvision
```

Or use the complete requirements:

```bash
pip install torch>=1.10.0 torchvision>=0.11.0
```

## 📊 Dataset

The model is trained on **CIFAR-10**, which consists of:
- **Training set**: 50,000 images
- **Test set**: 10,000 images
- **Image size**: 32×32 pixels (RGB)
- **Classes**: 10 (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)

The dataset is automatically downloaded when running the training script.

## 🚀 Usage

### Training the Model

Run the Jupyter notebook or execute the following Python code:

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# Hyperparameters
batch_size = 64
img_size = 32
patch_size = 8
num_channels = 3
num_patches = (img_size // patch_size) ** 2
num_heads = 24
embed_dim = 768
transformer_units = 6

# Load CIFAR-10 dataset
transform = transforms.Compose([transforms.ToTensor()])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
valset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                      download=True, transform=transform)

train_data = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True)
val_data = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                        shuffle=False)

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DeiT().to(device)
optimizer = optim.AdamW(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

# Training loop
for epoch in range(5):
    model.train()
    for images, labels in train_data:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        
        cls_out, dist_out = model(images)
        loss_cls = criterion(cls_out, labels)
        loss_dist = criterion(dist_out, labels)
        loss = 0.5 * loss_cls + 0.5 * loss_dist
        
        loss.backward()
        optimizer.step()
```

### Inference

```python
# Evaluation mode
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in val_data:
        images, labels = images.to(device), labels.to(device)
        cls_out, dist_out = model(images)
        preds = (cls_out + dist_out).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

test_acc = 100.0 * correct / total
print(f"Validation Accuracy: {test_acc:.2f}%")
```

## 🏗️ Model Architecture

### DeiT Components

1. **Patch Embedding Layer**
   - Converts input images into patch embeddings using Conv2d
   - Kernel size: 8×8, Stride: 8
   - Output: 768-dimensional embeddings

2. **Special Tokens**
   - Classification token (cls_token): Learnable parameter for class prediction
   - Distillation token (dist_token): Learnable parameter for distillation

3. **Positional Embeddings**
   - Learnable position encodings added to patch embeddings
   - Shape: (1, num_patches + 2, embed_dim)

4. **Transformer Encoder Blocks** (×6)
   - Layer Normalization
   - Multi-Head Self-Attention (24 heads)
   - Residual connections
   - Feedforward network with GELU activation
   - Dimension: 768 → 3072 → 768

5. **Dual Classification Heads**
   - Classification head: Projects cls_token to 10 classes
   - Distillation head: Projects dist_token to 10 classes
   - Final prediction: Average of both heads

## 📈 Results

Training results on CIFAR-10 (5 epochs):

| Epoch | Training Accuracy | Total Loss |
|-------|------------------|------------|
| 1     | 32.38%          | 1449.46    |
| 2     | 46.85%          | 1161.54    |
| 3     | 51.65%          | 1060.37    |
| 4     | 55.21%          | 980.73     |
| 5     | 58.49%          | 912.04     |

**Final Validation Accuracy**: 54.72%

### Training Configuration

- **Optimizer**: AdamW
- **Learning Rate**: 1e-4
- **Batch Size**: 64
- **Epochs**: 5
- **Loss Function**: Cross-Entropy (averaged over both heads)

*Note: Training for more epochs with data augmentation and learning rate scheduling would significantly improve accuracy.*

## 🛠️ Requirements

```
torch>=1.10.0
torchvision>=0.11.0
```

## 📝 Model Hyperparameters

| Parameter | Value |
|-----------|-------|
| Image Size | 32×32 |
| Patch Size | 8×8 |
| Embedding Dimension | 768 |
| Number of Heads | 24 |
| Transformer Blocks | 6 |
| MLP Dimension | 3072 |
| Batch Size | 64 |
| Learning Rate | 1e-4 |

## 🤝 Contributing

Contributions are welcome! Here are some ways you can contribute:

- Add data augmentation techniques (random flips, crops, color jitter)
- Implement learning rate scheduling
- Add support for other datasets (CIFAR-100, ImageNet)
- Improve model architecture with more advanced DeiT variants
- Add model checkpointing and logging
- Implement mixed precision training

### Steps to Contribute

1. Fork the repository
2. Create a new branch (`git checkout -b feature/improvement`)
3. Make your changes
4. Commit your changes (`git commit -am 'Add new feature'`)
5. Push to the branch (`git push origin feature/improvement`)
6. Create a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 References

- [DeiT: Data-efficient Image Transformers](https://arxiv.org/abs/2012.12877) - Touvron et al., 2021
- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) - Dosovitskiy et al., 2020
- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)

## 👤 Author

**Nakshatra1729yuvi**

## 🙏 Acknowledgments

- Thanks to the PyTorch team for the excellent deep learning framework
- CIFAR-10 dataset creators
- Original DeiT and Vision Transformer authors

---

*For questions or issues, please open an issue on GitHub.*
