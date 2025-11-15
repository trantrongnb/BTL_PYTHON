import torch
import torch.nn as nn
import torch.nn.functional as F

# Squeeze-and-Excitation block
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = F.adaptive_avg_pool2d(x, 1).view(b, c)
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y)).view(b, c, 1, 1)
        return x * y

# Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SEBlock(out_channels)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class My_Advanced_Model(nn.Module):
    def __init__(self, num_class):
        super().__init__()
        self.block1 = nn.Sequential(
            ResidualBlock(1, 64),
            nn.MaxPool2d(2),
            nn.Dropout(0.3)
        )
        self.block2 = nn.Sequential(
            ResidualBlock(64, 128),
            nn.MaxPool2d(2),
            nn.Dropout(0.3)
        )
        self.block3 = nn.Sequential(
            ResidualBlock(128, 256),
            nn.MaxPool2d(2),
            nn.Dropout(0.4)
        )
        self.block4 = nn.Sequential(
            ResidualBlock(256, 512),
            nn.MaxPool2d(2),
            nn.Dropout(0.4)
        )
        self.Adap = nn.AdaptiveAvgPool2d((1,1))
        self.flatten = nn.Flatten()
        self.fc1 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.fc2 = nn.Linear(256, num_class)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.Adap(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
