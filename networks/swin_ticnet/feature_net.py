from .module import Atten_Conv_Block, ResBlock3d
from .multi_scale import conv_2nV1, conv_3nV1
from .swin_transformer import SwinTransformer

import torch
from torch import nn


class FeatureNet(nn.Module):
    def __init__(
            self, 
            in_channels: int = 1, 
            out_channels: int = 128,      
        ):
        super(FeatureNet, self).__init__()
        self.preBlock = nn.Sequential(
            nn.Conv3d(in_channels, 24, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm3d(24, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Conv3d(24, 24, kernel_size=3, padding=1),
            nn.BatchNorm3d(24, momentum=0.1),
            nn.ReLU(inplace=True))

        self.forw1 = nn.Sequential(
            ResBlock3d(24, 32),
            ResBlock3d(32, 32),
            Atten_Conv_Block(32),
        )

        self.forw2 = nn.Sequential(
            ResBlock3d(32, 48),
            ResBlock3d(48, 48),
            Atten_Conv_Block(48),
        )

        self.forw3 = nn.Sequential(
            ResBlock3d(48, 48),
            ResBlock3d(48, 48),
            ResBlock3d(48, 48),
            Atten_Conv_Block(48),
        )

        self.forw4 = nn.Sequential(
            ResBlock3d(48, 48),
            ResBlock3d(48, 48),
            ResBlock3d(48, 48),
            Atten_Conv_Block(48),
        )

        self.forw_tr = nn.Sequential(
            ResBlock3d(768, 384),
            ResBlock3d(384, 384),
            ResBlock3d(384, 384),
            Atten_Conv_Block(384),
        )

        # skip connection in U-net
        self.back1 = nn.Sequential(
            # 64 + 64 + 3, where 3 is the channel dimension of coord
            ResBlock3d(384 + 48, 192),
            ResBlock3d(192, 192),
            ResBlock3d(192, 192),
            Atten_Conv_Block(192),
        )

        # skip connection in U-net
        self.back2 = nn.Sequential(
            ResBlock3d(384+48, 384),
            ResBlock3d(384, 384),
            ResBlock3d(384, 384),
            Atten_Conv_Block(384),
        )

        # skip connection in U-net
        self.back3 = nn.Sequential(
            ResBlock3d(384 + 48, 384),
            ResBlock3d(384, 384),
            ResBlock3d(384, 384),
            Atten_Conv_Block(384),
        )

        self.maxpool1 = nn.MaxPool3d(kernel_size=2, stride=2,
                                     return_indices=True)
        self.maxpool2 = nn.MaxPool3d(kernel_size=2, stride=2,
                                     return_indices=True)
        self.maxpool3 = nn.MaxPool3d(kernel_size=2, stride=2,
                                     return_indices=True)
        self.maxpool4 = nn.MaxPool3d(kernel_size=2, stride=2,
                                     return_indices=True)
        
        self.swinViT = SwinTransformer(
            in_chans=48,
            embed_dim=48,
            window_size=(3,3,3),
            patch_size=(2,2,2),
            depths=(2, 2, 2, 2),
            num_heads=(3, 6, 12, 24),
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=0.1,
            attn_drop_rate=0.1,
            drop_path_rate=0.1,
            norm_layer=nn.LayerNorm,
            use_checkpoint=False,
            spatial_dims=3,
        )

        self.scale1 = conv_3nV1(32, 48, 48, 48)

        self.scale2 = conv_3nV1(48, 48, 48, 48)

        self.scale3 = conv_2nV1(48, 48, 48)

        # upsampling in U-net
        self.path1 = nn.Sequential(
            nn.ConvTranspose3d(384, 384, kernel_size=2, stride=2),
            nn.BatchNorm3d(384),
            nn.ReLU(inplace=True))

        # upsampling in U-net
        self.path2 = nn.Sequential(
            nn.ConvTranspose3d(384, 384, kernel_size=2, stride=2),
            nn.BatchNorm3d(384),
            nn.ReLU(inplace=True))
        
        self.path3 = nn.Sequential(
            nn.Conv3d(192, 192, kernel_size=3, padding=1),
            nn.BatchNorm3d(192),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.preBlock(x)  # 24
        out1 = self.forw1(out)  # 32
        out1_pool, _ = self.maxpool2(out1)
        out2 = self.forw2(out1_pool)  # 48
      
        out2_pool, _ = self.maxpool3(out2)
        out3 = self.forw3(out2_pool)  # 48
        out3_pool, _ = self.maxpool4(out3)
        out4 = self.forw4(out3_pool)  # 48
       
        out4_tr = self.swinViT(out4) # 768
        rev3 = self.forw_tr(out4_tr) # 384

        out2_scale = self.scale1(out1, out2, out3) # 48
        out3_scale = self.scale2(out2, out3, out4) # 48
        out4_scale = self.scale3(out3, out4) # 48

        comb3 = self.back3(torch.cat((rev3, out4_scale), 1)) # 384
        rev2 = self.path1(comb3) # 384

        comb2 = self.back2(torch.cat((rev2, out3_scale), 1)) # 384
        rev1 = self.path2(comb2) # 384
        comb1 = self.back1(torch.cat((rev1, out2_scale), 1)) # 192

        comb1 = self.path3(comb1) # 192

        return {'0': comb1, '1': comb2}


