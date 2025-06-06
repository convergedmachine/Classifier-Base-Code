import torch
import torch.nn as nn
from .mixstyle import MixStyle

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, fc_dims = [], mixstyle_layers=[], mixstyle_p=0.5, mixstyle_alpha=0.3, num_classes=1000):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.mixstyle_layers = mixstyle_layers
        if mixstyle_layers:
            self.mixstyle = MixStyle(p=mixstyle_p, alpha=mixstyle_alpha, mix='random')
            print('Insert MixStyle after the following layers: {}'.format(mixstyle_layers))
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_dims = fc_dims
        self.feature_dim = 512 * block.expansion
        if fc_dims:
            self.fc = self._construct_fc_layer(
                fc_dims, 512 * block.expansion
            )
        self.classifier = nn.Linear(self.feature_dim, num_classes)    
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _construct_fc_layer(self, fc_dims, input_dim):
        """Constructs fully connected layer

        Args:
            fc_dims (list or tuple): dimensions of fc layers, if None, no fc layers are constructed
            input_dim (int): input dimension
            dropout_p (float): dropout probability, if None, dropout is unused
        """
        if fc_dims is None:
            self.feature_dim = input_dim
            return None

        assert isinstance(
            fc_dims, (list, tuple)
        ), 'fc_dims must be either list or tuple, but got {}'.format(
            type(fc_dims)
        )

        layers = []
        for dim in fc_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU(inplace=True))
            input_dim = dim

        self.feature_dim = fc_dims[-1]

        return nn.Sequential(*layers)        

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        if 'layer1' in self.mixstyle_layers:
            x = self.mixstyle(x)

        x = self.layer2(x)
        if 'layer2' in self.mixstyle_layers:
            x = self.mixstyle(x)

        x = self.layer3(x)
        if 'layer3' in self.mixstyle_layers:
            x = self.mixstyle(x)

        x = self.layer4(x)
        if 'layer4' in self.mixstyle_layers:
            x = self.mixstyle(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        if self.fc_dims:
            x = self.fc(x)
        x = self.classifier(x)

        return x

def resnet50(num_classes):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)

def resnet50_fc(num_classes):
    return ResNet(Bottleneck, [3, 4, 6, 3], fc_dims = [512], num_classes=num_classes)

def resnet50_fc_ml(num_classes):
    return ResNet(Bottleneck, [3, 4, 6, 3], fc_dims = [512], mixstyle_layers=['layer1', 'layer2', 'layer3'], num_classes=num_classes)

# Example usage:
# model = resnet50()
# print(model)