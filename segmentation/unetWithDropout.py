import torch.nn as nn
from monai.networks.blocks import Convolution
from monai.networks.nets import UNet
import torch
from monai.networks.blocks import UnetBasicBlock
from monai.networks.layers import Conv

class UNetWithDropout(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=(16, 32, 64, 128, 256), dropout_prob=0.3):
        super().__init__()

        self.encoder_blocks = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout2d(p=dropout_prob)

        # Encoder
        prev_channels = in_channels
        for f in features:
            block = UnetBasicBlock(
                spatial_dims=2,
                in_channels=prev_channels,
                out_channels=f,
                kernel_size=3,
                stride=1,
                norm_name="BATCH"
            )
            self.encoder_blocks.append(block)
            prev_channels = f

        # Bottleneck
        self.bottleneck = UnetBasicBlock(
            spatial_dims=2,
            in_channels=features[-1],
            out_channels=features[-1] * 2,
            kernel_size=3,
            stride=1,
            norm_name="BATCH"
        )

        # Decoder
        self.upconvs = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()

        for f in reversed(features):
            self.upconvs.append(nn.ConvTranspose2d(prev_channels, f, kernel_size=2, stride=2))
            self.decoder_blocks.append(UnetBasicBlock(
                spatial_dims=2,
                in_channels=prev_channels,
                out_channels=f,
                kernel_size=3,
                stride=1,
                norm_name="BATCH"
            ))
            prev_channels = f

        self.final_conv = Conv[Conv.CONV, 2](features[0], out_channels, kernel_size=1)

    def forward(self, x):
        enc_features = []

        for enc in self.encoder_blocks:
            x = enc(x)
            enc_features.append(x)
            x = self.pool(x)
            x = self.dropout(x)

        x = self.bottleneck(x)

        for i in range(len(self.upconvs)):
            x = self.upconvs[i](x)
            skip = enc_features[-(i + 1)]
            x = torch.cat((x, skip), dim=1)
            x = self.decoder_blocks[i](x)
            x = self.dropout(x)

        return self.final_conv(x)


