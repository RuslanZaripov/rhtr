"""
all credits to @nizhib
"""
import torch
import torch.nn as nn
from torchvision.models.resnet import \
    resnet18,\
    resnet34,\
    resnet50,\
    resnet101,\
    resnet152
from torch.nn import Conv2d

nonlinearity = nn.ReLU


ENCODERS = {
    'resnet18': resnet18,
    'resnet34': resnet34,
    'resnet50': resnet50,
    'resnet101': resnet101,
    'resnet152': resnet152
}


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super().__init__()

        # B, C, H, W -> B, C/4, H, W
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity(inplace=True)

        # B, C/4, H, W -> B, C/4, H, W
        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3,
                                          stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity(inplace=True)

        # B, C/4, H, W -> B, C, H, W
        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x


class LinkResNet(nn.Module):
    def __init__(self, k=50, input_channels=3, output_channels=1, dropout2d_p=0.5,
                 pretrained=True, encoder='resnet50'):
        assert input_channels > 0
        assert encoder in ENCODERS
        super().__init__()

        self.k = k

        if encoder in ['resnet18', 'resnet34']:
            filters = [64, 128, 256, 512]
        else:
            filters = [256, 512, 1024, 2048]

        resnet = ENCODERS[encoder](pretrained=pretrained)

        if input_channels != 3:
            resnet.conv1 = Conv2d(input_channels, 64, kernel_size=(7, 7),
                                  stride=(2, 2), padding=(3, 3), bias=False)

        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.dropout2d1 = nn.Dropout2d(p=dropout2d_p)
        self.dropout2d2 = nn.Dropout2d(p=dropout2d_p)
        self.dropout2d3 = nn.Dropout2d(p=dropout2d_p)

        # Decoder
        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        # Final Classifier
        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 3, stride=2)
        self.finalrelu1 = nonlinearity(inplace=True)
        self.finalconv2 = nn.Conv2d(32, 32, 3)
        self.finalrelu2 = nonlinearity(inplace=True)
        self.finalconv3 = nn.Conv2d(32, output_channels, 2, padding=1)
        self.sigmoid = nn.Sigmoid()

    def step_function(self, x, y):
        return torch.reciprocal(1 + torch.exp(-self.k * (x - y)))

    # noinspection PyCallingNonCallable
    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)

        e4 = self.encoder4(e3)

        # Decoder with Skip Connections
        d4 = self.decoder4(e4) + self.dropout2d1(e3)
        # d4 = e3
        d3 = self.decoder3(d4) + self.dropout2d2(e2)
        d2 = self.decoder2(d3) + self.dropout2d3(e1)
        d1 = self.decoder1(d2)

        # Final Classification
        f1 = self.finaldeconv1(d1)
        f2 = self.finalrelu1(f1)
        f3 = self.finalconv2(f2)
        f4 = self.finalrelu2(f3)
        f5 = self.finalconv3(f4)

        result = self.sigmoid(f5)

        # # result shape is (B, C, H, W)
        probability_map = result[:, 0, :, :]
        threshold_map = result[:, 1, :, :]

        thresh_binary = self.step_function(probability_map, threshold_map)  # (B, 1, H, W)

        # stack threshold binary for each batch as last  (B, C, H, W) -> (B, C + 1, H, W)
        thresh_binary = thresh_binary.unsqueeze(1)
        result = torch.cat([result, thresh_binary], dim=1)
        return result
