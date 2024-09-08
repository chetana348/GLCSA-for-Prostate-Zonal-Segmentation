import torch
import torch.nn as nn
from scripts.models.utils import *
from scripts.models.GLCSA_ps import *

class Attention(nn.Module):
    def __init__(self, input_encoder, input_decoder, output_dim, apply_norm='bn'):
        super().__init__()
        if apply_norm == 'gn':
            self.norm1 = nn.GroupNorm(32 if (input_encoder >= 32 and input_encoder % 32 == 0) else input_encoder,
                                      input_encoder)
            self.norm2 = nn.GroupNorm(32 if (input_decoder >= 32 and input_decoder % 32 == 0) else input_decoder,
                                      input_decoder)
            self.norm3 = nn.GroupNorm(32 if (output_dim >= 32 and output_dim % 32 == 0) else output_dim,
                                      output_dim)

        if apply_norm == 'bn':
            self.norm1 = nn.BatchNorm2d(input_encoder)
            self.norm2 = nn.BatchNorm2d(input_decoder)
            self.norm3 = nn.BatchNorm2d(output_dim)
        
        else:
            self.norm1, self.norm2, self.norm3 = nn.Identity(), nn.Identity(), nn.Identity()

        self.conv_encoder = nn.Sequential(
            self.norm1,
            nn.ReLU(),
            nn.Conv2d(input_encoder, output_dim, 3, padding=1),
            nn.MaxPool2d(2, 2),
        )

        self.conv_decoder = nn.Sequential(
            self.norm2,
            nn.ReLU(),
            nn.Conv2d(input_decoder, output_dim, 3, padding=1),
        )

        self.conv_attn = nn.Sequential(
            self.norm3,
            nn.ReLU(),
            nn.Conv2d(output_dim, 1, 1),
        )

    def forward(self, x1, x2):
        out = self.conv_encoder(x1) + self.conv_decoder(x2)
        out = self.conv_attn(out)
        return out * x2

    
class NormAct(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.norm = nn.BatchNorm2d(features)
        self.act = nn.ReLU()
    def forward(self ,x):
        return self.act(self.norm(x))

class ResConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv = nn.Sequential(NormAct(in_channels),
                                  nn.Conv2d(in_channels=in_channels, 
                                            out_channels=out_channels, 
                                            kernel_size=3, 
                                            padding=1, 
                                            stride=stride),
                                    NormAct(out_channels), 
                                  nn.Conv2d(in_channels=out_channels, 
                                            out_channels=out_channels, 
                                            kernel_size=3, 
                                            padding=1, 
                                            stride=1)                                     
                                  )
        self.skip = nn.Conv2d(in_channels=in_channels, 
                              out_channels=out_channels, 
                              kernel_size=1, 
                              padding=0, 
                              stride=stride)


    def forward(self, x):
        return self.conv(x) + self.skip(x)

class ResPlus(nn.Module):
    def __init__(self,
                n_blocks=1,
                in_channels=1, 
                out_channels=3, 
                k=0.5,
                input_size=(128, 128),
                fusion_out=None,
                patch_size=4,
                spatial_att=True,
                channel_att=True,
                spatial_head_dim=[4, 4, 4],
                channel_head_dim=[1, 1, 1], 
                device='cuda', 
                ):
        super().__init__()
  
        patch = input_size[0] // patch_size

        self.input_layer = nn.Sequential(Conv(in_channels=in_channels, out_channels=int(64 * k),),
            nn.Conv2d(int(64 * k), int(64 * k), kernel_size=3, padding=1),)
        self.input_skip = nn.Sequential(
            nn.Conv2d(in_channels, int(64 * k), kernel_size=3, padding=1))

        self.squeeze_excite1 = SEBlock(int(64 * k),reduction=int(16 * k),)

        self.res_conv1 = ResConv(int(64 * k),int(128 * k),stride=2,)

        self.squeeze_excite2 = SEBlock(int(128 * k),reduction=int(32 * k),)

        self.res_conv2 = ResConv(int(128 * k),int(256 * k), stride=2,)

        self.squeeze_excite3 = SEBlock(int(256 * k),reduction=int(32 * k),)

        self.res_conv3 = ResConv(int(256 * k), int(512 * k), stride=2,)

        self.aspp_bridge = ASPP(int(512 * k),int(1024 * k), apply_norm='bn',
                                    activation=False
                                    )
        
        self.glcsa = GLCSA(n_blocks=n_blocks,                                            
                                features = [int(64 * k), int(128 * k), int(256 * k)],                                                                                                              
                                strides=[patch_size, patch_size // 2, patch_size // 4],
                                patch=patch,
                                spatial_att=spatial_att,
                                channel_att=channel_att, 
                                spatial_head=spatial_head_dim,
                                channel_head=channel_head_dim,
                                ) 
        self.attn1 = Attention(int(256 * k), int(1024 * k), int(1024 * k),)
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest',)

        self.up_res_conv1 = ResConv(int(1024 * k) + int(256 * k), int(512 * k),)

        self.attn2 = Attention(int(128 * k),int(512 * k),int(512 * k),)
        self.up2 = nn.Upsample(scale_factor=2,  mode='nearest',)

        self.up_res_conv2 = ResConv(int(512 * k) + int(128 * k),  int(256 * k),)

        self.attn3 = Attention(int(64 * k),int(256 * k),int(256 * k),)
        self.up3 = nn.Upsample(scale_factor=2,    mode='nearest',)

        self.up_res_conv3 = ResConv(int(256 * k) + int(64 * k), int(128 * k),)

        self.aspp_out = ASPP(int(128 * k), int(64 * k))

        self.output_layer = nn.Conv2d(int(64 * k), out_channels, kernel_size=1,   padding=0)

    def forward(self, x):
        x1 = self.input_layer(x) + self.input_skip(x)

        x2 = self.squeeze_excite1(x1)
        x2 = self.res_conv1(x2)

        x3 = self.squeeze_excite2(x2)
        x3 = self.res_conv2(x3)

        x4 = self.squeeze_excite3(x3)
        x4 = self.res_conv3(x4)

        x5 = self.aspp_bridge(x4)

        x1, x2, x3 = self.glcsa([x1, x2, x3])       

        x6 = self.attn1(x3, x5)

        x6 = self.up1(x6)
        x6 = torch.cat([x6, x3], dim=1)
        x6 = self.up_res_conv1(x6)

        x7 = self.attn2(x2, x6)
        x7 = self.up2(x7)
        x7 = torch.cat([x7, x2], dim=1)
        x7 = self.up_res_conv2(x7)

        x8 = self.attn3(x1, x7)
        x8 = self.up3(x8)
        x8 = torch.cat([x8, x1], dim=1)
        x8 = self.up_res_conv3(x8)

        x9 = self.aspp_out(x8)
        out = self.output_layer(x9)

        return out

