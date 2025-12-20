# Classifier Module - CelebA Attribute Classification

Module này chứa các công cụ cho việc phân loại thuộc tính trên CelebA dataset sử dụng ResNet50 + CBAM architecture.

## Cấu trúc thư mục

```
classifiers/
├── __init__.py                 # Module initialization
├── model.py                    # ResNet50 + CBAM model architecture
├── dataset.py                  # CelebA dataset class
├── gradcam.py                  # Grad-CAM và Grad-CAM++ implementation
├── integrated_gradients.py     # Integrated Gradients implementation
├── inference.py                # Inference utilities
├── example_usage.py            # Script ví dụ sử dụng module
└── trainer/                    # Training scripts (nếu có)
```

## Components

### 1. Model (`model.py`)
- **ResNet50_CBAM**: Mô hình ResNet50 với CBAM attention blocks
- **SquarePadResize**: Transform để resize ảnh với padding (giữ nguyên aspect ratio)
- **ChannelAttention**: Channel attention module
- **SpatialAttention**: Spatial attention module
- **CBAMBlock**: Kết hợp channel và spatial attention

### 2. Dataset (`dataset.py`)
- **CelebADataset**: PyTorch Dataset cho CelebA multi-label classification
- Tự động chuyển đổi labels từ {-1, 1} sang {0, 1}

### 3. Grad-CAM (`gradcam.py`)
- **GradCAM**: Implementation của Gradient-weighted Class Activation Mapping
- **GradCAMPlusPlus**: Phiên bản cải tiến của Grad-CAM
- **visualize_gradcam**: Hàm visualization cho cả hai methods
- Hỗ trợ cả positive và negative class visualization

### 4. Integrated Gradients (`integrated_gradients.py`)
- **integrated_gradients**: Tính attribution map sử dụng Integrated Gradients
- **visualize_integrated_gradients**: Visualization function
- Hỗ trợ cả positive và negative class

### 5. Inference (`inference.py`)
- **inference_single_image**: Chạy inference cho một ảnh và hiển thị kết quả chi tiết

## Cách sử dụng

### Sử dụng trong Python Script

```python
from classifiers import (
    ResNet50_CBAM,
    SquarePadResize,
    CelebADataset,
    visualize_gradcam,
    visualize_integrated_gradients,
    inference_single_image
)

# Tạo model
model = ResNet50_CBAM(num_classes=40)

# Tạo transforms
from torchvision import transforms

train_transform = transforms.Compose([
    SquarePadResize(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

viz_transform = transforms.Compose([
    SquarePadResize(224),
    transforms.ToTensor()
])

# Load model
model.load_state_dict(torch.load("path/to/checkpoint.pth"))
model.eval()

# Inference
probs, preds, gt = inference_single_image(
    model=model,
    image_path="path/to/image.jpg",
    transform=train_transform,
    attribute_names=attribute_names,
    device=device,
    df_attr=df_attr
)

# Visualize Grad-CAM
visualize_gradcam(
    model=model,
    image_path="path/to/image.jpg",
    attribute_idx=0,
    attribute_name="Young",
    transform=train_transform,
    viz_transform=viz_transform,
    device=device,
    target_layer=model.layer4,
    target_class=1,
    method='gradcam'
)

# Visualize Integrated Gradients
visualize_integrated_gradients(
    model=model,
    image_path="path/to/image.jpg",
    attribute_idx=0,
    attribute_name="Young",
    transform=train_transform,
    viz_transform=viz_transform,
    device=device,
    target_class=1,
    steps=50
)
```

### Sử dụng trong Jupyter Notebook

Xem file `classifier_test_refactored.ipynb` để có ví dụ đầy đủ.

### Chạy example script

```bash
python classifiers/example_usage.py
```

## Parameters Quan Trọng

### Grad-CAM / Grad-CAM++
- `target_layer`: Layer để extract CAM (thường là `model.layer4` hoặc `model.cbam4`)
- `target_class`: 
  - `1`: Visualize cho positive prediction (attribute có mặt)
  - `0`: Visualize cho negative prediction (attribute không có)
- `method`: `'gradcam'` hoặc `'gradcam++'`

### Integrated Gradients
- `steps`: Số bước tích phân (20-300, mặc định 50)
- `baseline`: Ảnh baseline (mặc định là ảnh đen)
- `target_class`: 0 hoặc 1 như Grad-CAM

## Notes

1. **Target Class**: 
   - `target_class=1`: Model highlight vùng ảnh làm cho attribute **có mặt**
   - `target_class=0`: Model highlight vùng ảnh làm cho attribute **không có**

2. **Transform**: 
   - Luôn cần 2 transforms: một cho model (có normalization), một cho visualization (không có normalization)

3. **Target Layer**: 
   - Grad-CAM thường dùng `model.layer4` (layer cuối của ResNet)
   - Grad-CAM++ có thể dùng `model.cbam4` để tận dụng CBAM attention

4. **Memory**: 
   - Integrated Gradients tốn nhiều memory hơn Grad-CAM
   - Giảm `batch_size` trong IG nếu gặp OOM

## Dependencies

```
torch
torchvision
numpy
pandas
opencv-python
matplotlib
pillow
scikit-learn
```

## Citation

Nếu sử dụng module này, vui lòng cite các papers sau:

- **Grad-CAM**: Selvaraju et al. "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization" (ICCV 2017)
- **Grad-CAM++**: Chattopadhay et al. "Grad-CAM++: Generalized Gradient-Based Visual Explanations for Deep Convolutional Networks" (WACV 2018)
- **Integrated Gradients**: Sundararajan et al. "Axiomatic Attribution for Deep Networks" (ICML 2017)
- **CBAM**: Woo et al. "CBAM: Convolutional Block Attention Module" (ECCV 2018)
