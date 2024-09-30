import torch
from torch import nn
import torch.nn.functional as F

from split_layer import split_layer_dec
@split_layer_dec(__file__)
class AlexNet(nn.Module):
    def __init__(self, num_classes=10, init_weights=False):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=1)

        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, padding=2)

        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, padding=1)

        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, padding=1)

        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, padding=1)

        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.linear1 = nn.Linear(256 * 5 * 5, 4096)
        self.dropout1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(4096, 4096)
        self.dropout2 = nn.Dropout(p=0.5)
        self.linear3 = nn.Linear(4096, num_classes)
        # self.features = nn.Sequential(
        #     nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2),  # input[3, 224, 224]  output[48, 55, 55]
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=3, stride=2),                  # output[48, 27, 27]
        #     nn.Conv2d(48, 128, kernel_size=5, padding=2),           # output[128, 27, 27]
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=3, stride=2),                  # output[128, 13, 13]
        #     nn.Conv2d(128, 192, kernel_size=3, padding=1),          # output[192, 13, 13]
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(192, 192, kernel_size=3, padding=1),          # output[192, 13, 13]
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(192, 128, kernel_size=3, padding=1),          # output[128, 13, 13]
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=3, stride=2),                  # output[128, 6, 6]
        # )
        # self.classifier = nn.Sequential(
        #     nn.Dropout(p=0.5),
        #     nn.Linear(128 * 6 * 6, 2048),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(p=0.5),
        #     nn.Linear(2048, 2048),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(2048, num_classes),
        # )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.conv5(x)
        x = F.relu(x)
        x = self.pool3(x)
        x = F.relu(self.linear1(torch.flatten(x, start_dim=1)))
        x = self.dropout1(x)
        x = F.relu(self.linear2(x))
        x = self.dropout2(x)
        x = self.linear3(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
