import torch
import torch.nn as nn
from scripts.models.utils import *
from scripts.models.GLCSA_ps import *
import einops

class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, apply_norm='bn', activation=True, t=2):
        super().__init__()
        self.t = t
        self.conv = Conv(in_channels=in_channels, out_channels=out_channels, apply_norm=apply_norm, activation=activation)

    def forward(self, x):
        x1 = self.conv(x)
        for _ in range(self.t):     
            x1 = self.conv(x + x1)
        return x1


class rconv_block(nn.Module):
    def __init__(self, 
                in_channels, 
                out_channels, 
                apply_norm='bn', 
                activation=True, 
                t=2):
        super().__init__()
        self.conv = Conv(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding=0, apply_norm=None, activation=False)
        self.block = nn.Sequential(
            conv_block(in_channels=out_channels,out_channels=out_channels, t=t, apply_norm=apply_norm, activation=activation),
            conv_block(in_channels=out_channels,  out_channels=out_channels, t=t, apply_norm=None, activation=False))
        self.norm = nn.BatchNorm2d(out_channels)
        self.norm_c = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()
    def forward(self, x):
        x = self.conv(x)
        x1 = self.norm_c(x)
        x1 = self.act(x1)
        x1 = self.block(x1)
        xs = x + x1
        x = self.norm(xs)
        x = self.act(x)
        return x, xs
    
    
class RUNet(nn.Module):
    def __init__(self,
                n_blocks=1,
                in_channels=1, 
                out_channels=3, 
                k=0.5,
                input_size=(128,128),
                patch_size=8,
                spatial_att=True,
                channel_att=True,
                spatial_head_dim=[4, 4, 4, 4],
                channel_head_dim=[1, 1, 1, 1],
                device='cuda', 
                ):
        super().__init__()
        patch = input_size[0] // patch_size


        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        
        self.rconv1 = rconv_block(in_channels=in_channels, out_channels=int(64 * k))
        self.rconv2 = rconv_block(in_channels=int(64 * k), out_channels=int(128 * k))
        self.rconv3 = rconv_block(in_channels=int(128 * k), out_channels=int(256 * k))
        self.rconv4 = rconv_block(in_channels=int(256 * k), out_channels=int(512 * k))
        self.rconv5 = rconv_block(in_channels=int(512 * k), out_channels=int(1024 * k))


        self.GLCSA = GLCSA(n_blocks=n_blocks,                                            
                                features = [int(64 * k), int(128 * k), int(256 * k), int(512 * k)],                                                                                                              
                                strides=[patch_size, patch_size // 2, patch_size // 4, patch_size // 8],
                                patch=patch,
                                spatial_att=spatial_att,
                                channel_att=channel_att, 
                                spatial_head=spatial_head_dim,
                                channel_head=channel_head_dim,
                                                            )  
        self.up1 = UpConv(in_channels=int(1024 * k),out_channels=int(512 * k), )          
        self.rconv6 = rconv_block(in_channels=int(1024 * k),out_channels=int(512 * k))
        self.up2 = UpConv(in_channels=int(512 * k), out_channels=int(256 * k),)          
        self.rconv7 = rconv_block(in_channels=int(512 * k), out_channels=int(256 * k))
        self.up3 = UpConv(in_channels=int(256 * k), out_channels=int(128 * k),)          
        self.rconv8 = rconv_block(in_channels=int(256 * k), out_channels=int(128 * k))
        self.up4 = UpConv(in_channels=int(128 * k), out_channels=int(64 * k),)          
        self.rconv9 = rconv_block(in_channels=int(128 * k), out_channels=int(64 * k))   
        self.out = nn.Conv2d(in_channels=int(64 * k),  out_channels=out_channels,kernel_size=(1),padding=(0))                                                          
    def forward(self, x):
        x1, x1_ = self.rconv1(x)
        x2 = self.pool(x1)
        x2, x2_ = self.rconv2(x2)
        x3 = self.pool(x2)
        x3, x3_ = self.rconv3(x3)
        x4 = self.pool(x3)
        x4, x4_ = self.rconv4(x4)
        x = self.pool(x4)
        x, _ = self.rconv5(x)
        x1, x2, x3, x4 = self.GLCSA([x1_, x2_, x3_, x4_])
        x = self.up1(x)
        x = torch.cat((x, x4), dim=1)
        x, _ = self.rconv6(x)
        x = self.up2(x)
        x = torch.cat((x, x3), dim=1)
        x, _ = self.rconv7(x)
        x = self.up3(x)
        x = torch.cat((x, x2), dim=1)
        x, _ = self.rconv8(x)
        x = self.up4(x)
        x = torch.cat((x, x1), dim=1)
        x, _ = self.rconv9(x)
        x = self.out(x)
        return x

