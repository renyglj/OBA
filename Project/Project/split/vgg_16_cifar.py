import torch
from torch import nn
import torch.nn.functional as F

from split_layer import split_layer_dec
@split_layer_dec(__file__)
class VGG(nn.Module):
    def __init__(self, num_classes=10, init_weights=False):
        super(VGG, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.conv7 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn7 = nn.BatchNorm2d(256)

        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv8 = nn.Conv2d(256, 512, 3, padding=1)
        self.bn8 = nn.BatchNorm2d(512)
        self.conv9 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn9 = nn.BatchNorm2d(512)
        self.conv10 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn10 = nn.BatchNorm2d(512)

        self.pool4 = nn.MaxPool2d(2, 2)

        self.conv11 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn11 = nn.BatchNorm2d(512)
        self.conv12 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn12 = nn.BatchNorm2d(512)
        self.conv13 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn13 = nn.BatchNorm2d(512)

        self.pool5 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(512*1*1, 512)
        self.fc2 = nn.Sequential(
            # nn.Dropout(p=0.5),
            nn.Linear(512, 256)
        )
        self.fc3 = nn.Sequential(
            # nn.Dropout(p=0.5),
            nn.Linear(256, num_classes)
        )

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = F.relu(self.bn7(self.conv7(x)))
        x = self.pool3(x)
        x = F.relu(self.bn8(self.conv8(x)))
        x = F.relu(self.bn9(self.conv9(x)))
        x = F.relu(self.bn10(self.conv10(x)))
        x = self.pool4(x)
        x = F.relu(self.bn11(self.conv11(x)))
        x = F.relu(self.bn12(self.conv12(x)))
        x = F.relu(self.bn13(self.conv13(x)))
        x = self.pool5(x)
        # print(x.shape)
        # x = x.view(-1, 512)
        x = F.relu(self.fc1(torch.flatten(x, start_dim=1)))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
# class VGG(nn.Module):
#     def __init__(self):
#         super(VGG, self).__init__()
#         # 2个卷积层和1个最大池化层
#         self.layer1 = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),  # (32-3+2)/1+1 = 32  32*32*64
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#
#             nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),  # (32-3+2)/1+1 = 32  32*32*64
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#
#             nn.MaxPool2d(2, 2)  # (32-2)/2+1 = 16    16*16*64
#
#         )
#         # 2个卷积层和1个最大池化层
#         self.layer2 = nn.Sequential(
#             nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # (16-3+2)/1+1 = 16  16*16*128
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True),
#
#             nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),  # (16-3+2)/1+1 = 16  16*16*128
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True),
#
#             nn.MaxPool2d(2, 2)  # (16-2)/2+1 = 8    8*8*128
#         )
#         # 3个卷积层和1个最大池化层
#         self.layer3 = nn.Sequential(
#             nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # (8-3+2)/1+1 = 8  8*8*256
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True),
#
#             nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),  # (8-3+2)/1+1 = 8  8*8*256
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True),
#
#             nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),  # (8-3+2)/1+1 = 8  8*8*256
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True),
#
#             nn.MaxPool2d(2, 2),  # (8-2)/2+1 = 4    4*4*256
#         )
#         # 3个卷积层和1个最大池化层
#         self.layer4 = nn.Sequential(
#             nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),  # (4-3+2)/1+1 = 4  4*4*512
#             nn.BatchNorm2d(512),
#             nn.ReLU(inplace=True),
#
#             nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),  # (4-3+2)/1+1 = 4  4*4*512
#             nn.BatchNorm2d(512),
#             nn.ReLU(inplace=True),
#
#             nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),  # (4-3+2)/1+1 = 4  4*4*512
#             nn.BatchNorm2d(512),
#             nn.ReLU(inplace=True),
#
#             nn.MaxPool2d(2, 2)  # (4-2)/2+1 = 2    2*2*512
#         )
#         # 3个卷积层和1个最大池化层
#         self.layer5 = nn.Sequential(
#             nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),  # (2-3+2)/1+1 = 2  2*2*512
#             nn.BatchNorm2d(512),
#             nn.ReLU(inplace=True),
#
#             nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),  # (2-3+2)/1+1 = 2  2*2*512
#             nn.BatchNorm2d(512),
#             nn.ReLU(inplace=True),
#
#             nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),  # (2-3+2)/1+1 = 2  2*2*512
#             nn.BatchNorm2d(512),
#             nn.ReLU(inplace=True),
#
#             nn.MaxPool2d(2, 2)  # (2-2)/2+1 = 1    1*1*512
#         )
#         self.conv = nn.Sequential(
#             self.layer1,
#             self.layer2,
#             self.layer3,
#             self.layer4,
#             self.layer5
#         )
#         self.fc = nn.Sequential(
#             nn.Linear(512, 512),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.5),
#
#             nn.Linear(512, 256),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.5),
#
#             nn.Linear(256, 10)
#         )
#
#     def forward(self, x):
#         x = self.conv(x)
#         x = x.view(-1, 512)
#         x = self.fc(x)
#         return x

