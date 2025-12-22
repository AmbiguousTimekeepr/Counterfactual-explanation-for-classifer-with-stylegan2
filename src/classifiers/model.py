"""
ResNet18 + CBAM Classifier Model
"""
import torch
import torch.nn as nn
from torchvision import models
from PIL import Image


class SquarePadResize:
    """
    Resize chiều lớn nhất về target_size, sau đó thêm padding đen 
    để ảnh trở thành hình vuông (target_size x target_size).
    Giữ nguyên tỉ lệ ảnh (Aspect Ratio).
    """
    def __init__(self, target_size):
        self.target_size = target_size

    def __call__(self, img):
        w, h = img.size
        max_dim = max(w, h)
        scale = self.target_size / max_dim
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.BICUBIC)
        
        # Tạo ảnh nền đen
        new_img = Image.new("RGB", (self.target_size, self.target_size), (0, 0, 0))
        # Paste ảnh gốc vào giữa
        new_img.paste(img, ((self.target_size - new_w) // 2, (self.target_size - new_h) // 2))
        return new_img


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAMBlock(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAMBlock, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        out = out * self.sa(out)
        return out


class ResNet50_CBAM(nn.Module):
    def __init__(self, num_classes):
        super(ResNet50_CBAM, self).__init__()
        # Load pre-trained ResNet50
        base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        # Tách các layer để chèn CBAM vào giữa các stage
        self.stem = nn.Sequential(
            base_model.conv1, base_model.bn1, base_model.relu, base_model.maxpool
        )
        self.layer1 = base_model.layer1
        self.cbam1 = CBAMBlock(256)  # ResNet50 layer1 out channels = 256
        
        self.layer2 = base_model.layer2
        self.cbam2 = CBAMBlock(512)  # ResNet50 layer2 out channels = 512
        
        self.layer3 = base_model.layer3
        self.cbam3 = CBAMBlock(1024)  # ResNet50 layer3 out channels = 1024
        
        self.layer4 = base_model.layer4
        self.cbam4 = CBAMBlock(2048)  # ResNet50 layer4 out channels = 2048
        
        self.avgpool = base_model.avgpool
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.stem(x)
        
        x = self.layer1(x)
        x = self.cbam1(x)  # Apply CBAM
        
        x = self.layer2(x)
        x = self.cbam2(x)  # Apply CBAM
        
        x = self.layer3(x)
        x = self.cbam3(x)  # Apply CBAM
        
        x = self.layer4(x)
        x = self.cbam4(x)  # Apply CBAM
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class ResNet18_CBAM(nn.Module):
    def __init__(self, num_classes):
        super(ResNet18_CBAM, self).__init__()
        # Load pre-trained ResNet18
        base_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        # Tách các layer để chèn CBAM vào giữa các stage
        self.stem = nn.Sequential(
            base_model.conv1, base_model.bn1, base_model.relu, base_model.maxpool
        )
        self.layer1 = base_model.layer1
        self.cbam1 = CBAMBlock(64) # ResNet18 layer1 out channels = 64
        
        self.layer2 = base_model.layer2
        self.cbam2 = CBAMBlock(128) # ResNet18 layer2 out channels = 128
        
        self.layer3 = base_model.layer3
        self.cbam3 = CBAMBlock(256) # ResNet18 layer3 out channels = 256
        
        self.layer4 = base_model.layer4
        self.cbam4 = CBAMBlock(512) # ResNet18 layer4 out channels = 512
        
        self.avgpool = base_model.avgpool
        self.fc = nn.Linear(512, num_classes)
    def forward(self, x):
        x = self.stem(x)
        
        x = self.layer1(x)
        x = self.cbam1(x) # Apply CBAM
        
        x = self.layer2(x)
        x = self.cbam2(x) # Apply CBAM
        
        x = self.layer3(x)
        x = self.cbam3(x) # Apply CBAM
        
        x = self.layer4(x)
        x = self.cbam4(x) # Apply CBAM
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x