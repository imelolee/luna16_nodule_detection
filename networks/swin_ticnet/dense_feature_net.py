from .module import Atten_Conv_Block, _DenseBlock, _Transition
from .multi_scale import conv_2nV1, conv_3nV1
from .swin_transformer import SwinTransformer

import torch
from torch import nn


class DenseFeatureNet(nn.Module):
    def __init__(
            self, 
            in_channels: int = 1, 
            out_channels: int = 128,  
            num_layers: int = 8,    
            bn_size: int = 4,
            growth_rate: int = 12,
        ):
        super(DenseFeatureNet, self).__init__()
        self.preBlock = nn.Sequential(
            nn.Conv3d(in_channels, 24, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm3d(24, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Conv3d(24, 24, kernel_size=3, padding=1),
            nn.BatchNorm3d(24, momentum=0.1),
            nn.ReLU(inplace=True)
        )

        self.forw1 = nn.Sequential(
            _DenseBlock(num_layers=num_layers, num_input_features=24, bn_size=bn_size, growth_rate=growth_rate),
            _Transition(num_input_features=24+growth_rate*num_layers, num_output_features=32),
            Atten_Conv_Block(32),
        )

        self.forw2 = nn.Sequential(
            _DenseBlock(num_layers=num_layers, num_input_features=32, bn_size=bn_size, growth_rate=growth_rate),
            _Transition(num_input_features=32+growth_rate*num_layers, num_output_features=64),
            Atten_Conv_Block(64),
        )

        self.forw3 = nn.Sequential(
            _DenseBlock(num_layers=num_layers, num_input_features=64, bn_size=bn_size, growth_rate=growth_rate),
            _Transition(num_input_features=64+growth_rate*num_layers, num_output_features=64),
            Atten_Conv_Block(64),
        )

        self.forw4 = nn.Sequential(
            _DenseBlock(num_layers=num_layers, num_input_features=64, bn_size=bn_size, growth_rate=growth_rate),
            _Transition(num_input_features=64+growth_rate*num_layers, num_output_features=48),
            Atten_Conv_Block(48),
        )

        self.forw_tr = nn.Sequential(
            nn.Conv3d(768, 256, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm3d(256, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128, momentum=0.1),
            nn.ReLU(inplace=True)
        )

        # skip connection in U-net
        self.back1 = nn.Sequential(
            # 64 + 64 + 3, where 3 is the channel dimension of coord
            _DenseBlock(num_layers=num_layers, num_input_features=128, bn_size=bn_size, growth_rate=growth_rate),
            _Transition(num_input_features=128+growth_rate*num_layers, num_output_features=out_channels),
            Atten_Conv_Block(out_channels),
        )

        # skip connection in U-net
        self.back2 = nn.Sequential(
            _DenseBlock(num_layers=num_layers, num_input_features=192, bn_size=bn_size, growth_rate=growth_rate),
            _Transition(num_input_features=192+growth_rate*num_layers, num_output_features=128),
            Atten_Conv_Block(128),
        )

        # skip connection in U-net
        self.back3 = nn.Sequential(
            _DenseBlock(num_layers=num_layers, num_input_features=192, bn_size=bn_size, growth_rate=growth_rate),
            _Transition(num_input_features=192+growth_rate*num_layers, num_output_features=128),
            Atten_Conv_Block(128),
        )

        self.avgpool = nn.AvgPool3d(kernel_size=2, stride=2)

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

        self.scale1 = conv_3nV1(32, 64, 64)

        self.scale2 = conv_3nV1(64, 64, 48)

        self.scale3 = conv_2nV1(64, 48)

        # upsampling in U-net
        self.path1 = nn.Sequential(
            nn.ConvTranspose3d(128, 128, kernel_size=2, stride=2),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True))

        # upsampling in U-net
        self.path2 = nn.Sequential(
            nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True))

    def forward(self, x):
        out0 = self.preBlock(x)  
        out1 = self.forw1(out0) 
        out1_pool = self.avgpool(out1)
        out2 = self.forw2(out1_pool)  
      
        out2_pool = self.avgpool(out2)
        out3 = self.forw3(out2_pool)  
        out3_pool = self.avgpool(out3)
        out4 = self.forw4(out3_pool)  
       
        out4_tr = self.swinViT(out4)
        rev3 = self.forw_tr(out4_tr)  
  
        out4_tr = self.swinViT(out4)
        rev3 = self.forw_tr(out4_tr)  

        out2_scale = self.scale1(out1, out2, out3) # 1/8, 64
        out3_scale = self.scale2(out2, out3, out4)
        out4_scale = self.scale3(out3, out4)

        comb3 = self.back3(torch.cat((rev3, out4_scale), 1))
        rev2 = self.path1(comb3)

        comb2 = self.back2(torch.cat((rev2, out3_scale), 1))  
        rev1 = self.path2(comb2)
        comb1 = self.back1(torch.cat((rev1, out2_scale), 1))  

        return {'0': comb1, '1': comb2, '2': comb3}


