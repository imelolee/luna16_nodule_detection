from .module import Atten_Conv_Block
from .multi_scale import conv_2nV1, conv_3nV1
from .transformer import Transformer
from .position_encoding import build_position_encoding

import torch
from torch import Tensor, nn
from typing import Optional, Sequence, Tuple, Type, Union, Dict, List
from monai.networks.blocks.feature_pyramid_network import ExtraFPNBlock, FeaturePyramidNetwork, LastLevelMaxPool


bn_momentum = 0.1

class ResBlock3d(nn.Module):
    def __init__(self, n_in, n_out, stride=1, ):
        super(ResBlock3d, self).__init__()
        self.conv1 = nn.Conv3d(n_in, n_out, kernel_size=3,
                               stride=stride, padding=1)
        self.bn1 = nn.BatchNorm3d(n_out, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(n_out, n_out, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(n_out, momentum=bn_momentum)

        if stride != 1 or n_out != n_in:
            self.shortcut = nn.Sequential(
                nn.Conv3d(n_in, n_out, kernel_size=1, stride=stride),
                nn.BatchNorm3d(n_out, momentum=bn_momentum))
        else:
            self.shortcut = None

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)
        return out


class FeatureNet(nn.Module):
    def __init__(
            self, 
            in_channels: int = 1, 
            out_channels: int = 128, 
            hidden_dim: int = 64,
            position_embedding: str = 'sine',  
            dropout: float = 0.1,
            nheads: int = 8,
            num_queries: int = 512,
            dim_feedforward: int = 256,
            num_encoder_layers: int = 6,
            num_decoder_layers: int = 6,
            normalize_before: str = None,
            return_intermediate_dec: bool =True       
        ):
        super(FeatureNet, self).__init__()
        self.preBlock = nn.Sequential(
            nn.Conv3d(in_channels, 24, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm3d(24, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.Conv3d(24, 24, kernel_size=3, padding=1),
            nn.BatchNorm3d(24, momentum=bn_momentum),
            nn.ReLU(inplace=True))

        self.forw1 = nn.Sequential(
            ResBlock3d(24, 32),
            ResBlock3d(32, 32),
            Atten_Conv_Block(32),
        )

        self.forw2 = nn.Sequential(
            ResBlock3d(32, 64),
            ResBlock3d(64, 64),
            Atten_Conv_Block(64),
        )

        self.forw3 = nn.Sequential(
            ResBlock3d(64, 64),
            ResBlock3d(64, 64),
            ResBlock3d(64, 64),
            Atten_Conv_Block(64),
        )

        self.forw4 = nn.Sequential(
            ResBlock3d(64, 64),
            ResBlock3d(64, 64),
            ResBlock3d(64, 64),
            Atten_Conv_Block(64),
        )

        # skip connection in U-net
        self.back1 = nn.Sequential(
            # 64 + 64 + 3, where 3 is the channel dimension of coord
            ResBlock3d(128, 128),
            ResBlock3d(128, 128),
            ResBlock3d(128, out_channels),
            Atten_Conv_Block(out_channels),
        )

        # skip connection in U-net
        self.back2 = nn.Sequential(
            ResBlock3d(192, 64),
            ResBlock3d(64, 64),
            ResBlock3d(64, 64),
            Atten_Conv_Block(64),
        )

        # skip connection in U-net
        self.back3 = nn.Sequential(
            ResBlock3d(128, 128),
            ResBlock3d(128, 128),
            ResBlock3d(128, 128),
            Atten_Conv_Block(128),
        )

        self.maxpool1 = nn.MaxPool3d(kernel_size=2, stride=2,
                                     return_indices=True)
        self.maxpool2 = nn.MaxPool3d(kernel_size=2, stride=2,
                                     return_indices=True)
        self.maxpool3 = nn.MaxPool3d(kernel_size=2, stride=2,
                                     return_indices=True)
        self.maxpool4 = nn.MaxPool3d(kernel_size=2, stride=2,
                                     return_indices=True)
        self.position_embedding = build_position_encoding(hidden_dim, position_embedding)
        self.transformer = Transformer(
            d_model = hidden_dim,
            dropout = dropout,
            nhead = nheads,
            num_queries = num_queries,
            dim_feedforward = dim_feedforward,
            num_encoder_layers = num_encoder_layers,
            num_decoder_layers = num_decoder_layers,
            normalize_before = normalize_before,
            return_intermediate_dec = return_intermediate_dec,
        )

        self.scale1 = conv_3nV1(32, 64, 64)

        self.scale2 = conv_3nV1(64, 64, 64)

        self.scale3 = conv_2nV1(64, 64)

        # upsampling in U-net
        self.path1 = nn.Sequential(
            nn.ConvTranspose3d(128, 128, kernel_size=2, stride=2),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True))

        # upsampling in U-net
        self.path2 = nn.Sequential(
            nn.ConvTranspose3d(64, 64, kernel_size=2, stride=2),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True))

    def forward(self, x):
        out = self.preBlock(x)  # 24, 1/2
        out_pool = out
        out1 = self.forw1(out_pool)  # 32
        out1_pool, _ = self.maxpool2(out1)
        out2 = self.forw2(out1_pool)  # 64
        # out2 = self.drop(out2)
        out2_pool, _ = self.maxpool3(out2)
        out3 = self.forw3(out2_pool)  # 64
        out3_pool, _ = self.maxpool4(out3)
        out4 = self.forw4(out3_pool)  # 64
        # out4 = self.drop(out4)
        pe = self.position_embedding(out4)
        out4_tr = self.transformer(out4, pe)

        out2_scale = self.scale1(out1, out2, out3)
        out3_scale = self.scale2(out2, out3, out4)
        out4_scale = self.scale3(out3, out4)

        comb3 = self.back3(torch.cat((out4_tr, out4_scale), 1))
        rev2 = self.path1(comb3)
        # rev2 = self.path1(out4_tr)
        comb2 = self.back2(torch.cat((rev2, out3_scale), 1))  # 96+96
        rev1 = self.path2(comb2)
        comb1 = self.back1(torch.cat((rev1, out2_scale), 1))  # 64+64

        return {'0': comb1, '1': comb2}


