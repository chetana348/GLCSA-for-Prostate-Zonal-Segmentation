import torch
import torch.nn as nn
from scripts.models.utils import *
from scripts.models.GLCSA_ps import *

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, apply_norm=True, activation=True):
        super().__init__()
        self.apply_norm = apply_norm
        self.activation = activation
        self.conv = nn.Conv2d(in_channels=in_channels, 
                                            out_channels=out_channels, 
                                            kernel_size=5, 
                                            padding=2)
        self.norm = nn.BatchNorm2d(out_channels) if self.apply_norm else None
        self.act = nn.PReLU() if self.activation else None

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x) if self.apply_norm else x
        x = self.act(x) if self.activation else x
        return x 

class ConvIn(nn.Module):
    def __init__(self, in_channels, out_channels, apply_norm=True, activation=True):
        super().__init__()
        self.apply_norm = apply_norm
        self.activation = activation
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=5,padding=2)
        self.skip = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=1,padding=0)
        self.norm = nn.BatchNorm2d(out_channels) if self.apply_norm else None
        self.act = nn.PReLU() if self.activation else None

    def forward(self, x):
        x_skip = self.skip(x)
        x = self.conv(x)
        xs = x + x_skip
        x = self.norm(x)
        x = self.act(x)
        return x, xs

class DownConv(nn.Module):
    def __init__(self, in_channels, out_channels, n_layers):
        super().__init__()
        self.n_layers = n_layers
        self.down = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=2)

        self.conv = nn.ModuleList([ConvBlock(in_channels=out_channels, 
                                            out_channels=out_channels)
                                            for _ in range(n_layers - 1)])

        self.conv.append(ConvBlock(in_channels=out_channels, out_channels=out_channels, apply_norm=False, activation=False))
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.PReLU()

    def forward(self, x):
        x_d = self.down(x)
        x = x_d.clone()
        for i in range(self.n_layers):
            x = self.conv[i](x)
        x_s = x + x_d     
        x = self.norm(x_s)
        x = self.act(x)
        return x, x_s

class UpConv(nn.Module):
    def __init__(self, in_channels, inter_channels, out_channels, n_layers) :
        super().__init__()
        self.n_layers = n_layers
        self.up = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=2)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.PReLU()
        self.conv = nn.ModuleList([ConvBlock(in_channels=out_channels + inter_channels, 
                                                    out_channels=out_channels)])

        for _ in range(n_layers - 2):
            self.conv.append(ConvBlock(in_channels=out_channels, 
                                              out_channels=out_channels))

        self.conv.append(ConvBlock(in_channels=out_channels, 
                                          out_channels=out_channels, 
                                          apply_norm=False, 
                                          activation=False))
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.PReLU()

    def forward(self, x_e, x_d):
        x_d = self.up(x_d)
        x = self.norm1(x_d)
        x = self.act1(x)
        x = torch.cat((x_e, x), dim=1)
        for i in range(self.n_layers):
            x = self.conv[i](x)
        x += x_d
        x = self.norm2(x)
        x = self.act2(x)
        return x    

class ConvOut(nn.Module):
    def __init__(self, in_channels, inter_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=2)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.PReLU()                              
        self.conv = nn.Conv2d(in_channels=out_channels + inter_channels,out_channels=out_channels,kernel_size=5,padding=2)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.PReLU()

    def forward(self, x_e, x_d):
        x_d = self.up(x_d)
        x = self.norm1(x_d)
        x = self.act1(x)
        x = torch.cat((x_e, x), dim=1)
        x = self.conv(x)
        x += x_d
        x = self.norm2(x)
        x = self.act2(x)
        return x

class VNet(nn.Module):
    def __init__(self,
                n_blocks=1,
                in_channels=1, 
                out_channels=3, 
                k=0.5,
                input_size=(128, 128),
                patch_size=8,
                spatial_att=True,
                channel_att=True,
                spatial_head_dim=[4, 4, 4, 4],
                channel_head_dim=[1, 1, 1, 1],
                emb = 'patch',
                fusion = 'summation',
                device='cuda', 
                ):
        super().__init__()
        k = 1
    
        patch = input_size[0] // patch_size
  
        
        self.conv1 = ConvIn(in_channels=in_channels, out_channels=int(32 * k))
        self.conv2 = DownConv(in_channels=int(32 * k), out_channels=int(64 * k), n_layers=2)
        self.conv3 = DownConv(in_channels=int(64 * k), out_channels=int(128 * k), n_layers=3)
        self.conv4 = DownConv(in_channels=int(128 * k), out_channels=int(256 * k), n_layers=3)
        self.conv5 = DownConv(in_channels=int(256 * k), out_channels=int(512 * k), n_layers=3)
        
       
        self.glcsa = GLCSA(n_blocks=n_blocks,                                            
                                features = [int(32 * k), int(64 * k), int(128 * k), int(256 * k)],                                                                                                              
                                strides=[patch_size, patch_size // 2, patch_size // 4, patch_size // 8],
                                patch=patch,
                                emb=emb,
                                fusion=fusion,
                                spatial_att=spatial_att,
                                channel_att=channel_att, 
                                spatial_head=spatial_head_dim,
                                channel_head=channel_head_dim,
                                )  
        self.up1 = UpConv(in_channels=int(512 * k),inter_channels=int(256 * k), out_channels=int(256 * k), n_layers=3)
        self.up2 = UpConv(in_channels=int(256 * k),inter_channels=int(128 * k), out_channels=int(128 * k), n_layers=3)
        self.up3 = UpConv(in_channels=int(128 * k),inter_channels=int(64 * k), out_channels=int(64 * k), n_layers=2)
        self.up4 = ConvOut(in_channels=int(64 * k),inter_channels=int(32 * k), out_channels=int(32 * k))

        self.out = nn.Conv2d(in_channels=int(32 * k),out_channels=out_channels, kernel_size=(1, 1), padding=(0, 0))
    def forward(self, x):
        x1, x1_ = self.conv1(x)
        x2, x2_ = self.conv2(x1)
        x3, x3_ = self.conv3(x2)
        x4, x4_ = self.conv4(x3)
        x, _ = self.conv5(x4)
        x1, x2, x3, x4 = self.glcsa([x1_, x2_, x3_, x4_])
        x = self.up1(x4, x)
        x = self.up2(x3, x)
        x = self.up3(x2, x)
        x = self.up4(x1, x)
        x = self.out(x)
        return x


if __name__ == '__main__':
    model = VNet()
    in1 = torch.rand(1, 1, 128, 128)
    out = model(in1)
    total_params = sum(p.numel() for p in model.parameters())
    print(total_params)
    print(out.shape)