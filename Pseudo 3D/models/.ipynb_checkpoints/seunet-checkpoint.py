import torch
import torch.nn as nn
from torchvision.models import vgg19
from scripts.models.utils import *
from scripts.models.GLCSA_ps import *


class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = Conv(in_channels=in_channels, out_channels=out_channels)
        self.conv2 = Conv(in_channels=out_channels, out_channels=out_channels, apply_norm=None, activation=False)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()
        self.se = SEBlock(in_channels=out_channels, reduction=8)
    def forward(self, x):
        x = self.conv1(x)
        xs = self.conv2(x)
        x = self.norm(xs)
        x = self.act(x)
        x = self.se(x)
        return x, xs

class SEUNet(nn.Module):
    def __init__(self,
                n_blocks=1,
                in_channels=1, 
                out_channels=3, 
                k=1,
                pretrained = False, 
                input_size=(128, 128),
                patch_size=8,
                spatial_att=True,
                channel_att=True,
                spatial_head_dim=[4, 4, 4, 4],
                channel_head_dim=[1, 1, 1, 1], 
                device = 'cuda'
                ):
        super().__init__()    
        patch = input_size[0] // patch_size

        self.mu = torch.tensor([0.485, 0.456, 0.406], requires_grad=False).to(device).view((1, 3, 1, 1))
        self.sigma = torch.tensor([0.229, 0.224, 0.225], requires_grad=False).to(device).view((1, 3, 1, 1))
        self.act = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.up = nn.Upsample(scale_factor=(2),mode='bilinear',align_corners=True)

        self.vgg1 = vgg19(pretrained=pretrained).features[:3]
        self.vgg2 = vgg19(pretrained=pretrained).features[4:8]
        self.vgg3 = vgg19(pretrained=pretrained).features[9:17]
        self.vgg4 = vgg19(pretrained=pretrained).features[18:26]
        self.vgg5 = vgg19(pretrained=pretrained).features[27:-2]

        for m in [self.vgg1, self.vgg2, self.vgg3, self.vgg4, self.vgg5]:
            for param in m.parameters():
                param.requires_grad = True

        self.aspp_1 = ASPP(in_channels=512, out_channels=64)
        self.G_vgg1 = GLCSA(n_blocks=n_blocks,features = [64, 128, 256, 512],                                                                               
                                strides=[patch_size, patch_size // 2, patch_size // 4, patch_size // 8],
                                patch=patch,
                                spatial_att=spatial_att,
                                channel_att=channel_att, 
                                spatial_head=spatial_head_dim,
                                channel_head=channel_head_dim,
                                ) 
        self.G_vgg2 = GLCSA(n_blocks=n_blocks, features = [64, 128, 256, 512],                                                                                                              strides=[patch_size, patch_size // 2, patch_size // 4, patch_size // 8],
                                patch=patch,
                                spatial_att=spatial_att,
                                channel_att=channel_att, 
                                spatial_head=spatial_head_dim,
                                channel_head=channel_head_dim,
                                ) 
        self.GLCSA =  GLCSA(n_blocks=n_blocks, features = [32, 64, 128, 256],                                                                             
                                strides=[patch_size, patch_size // 2, patch_size // 4, patch_size // 8],
                                patch=patch,
                                spatial_att=spatial_att,
                                channel_att=channel_att, 
                                spatial_head=spatial_head_dim,
                                channel_head=channel_head_dim,
                                ) 

        self.decode1 = conv_block(in_channels=64 + 512, out_channels=256)
        self.decode2 = conv_block(in_channels=256 + 256, out_channels=128)
        self.decode3 = conv_block(in_channels=128 + 128, out_channels=64)
        self.decode4 = conv_block(in_channels=64 + 64, out_channels=32)
        self.out1 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=in_channels, kernel_size=1, padding=0), 
                              nn.Sigmoid())
        self.encode2_1 = conv_block(in_channels=in_channels, out_channels=32)
        self.encode2_2 = conv_block(in_channels=32, out_channels=64)
        self.encode2_3 = conv_block(in_channels=64, out_channels=128)
        self.encode2_4 = conv_block(in_channels=128, out_channels=256)

        self.aspp_2 = ASPP(in_channels=256, out_channels=64)

        self.decode2_1 = conv_block(in_channels=64 + 512 + 256, out_channels=256)
        self.decode2_2 = conv_block(in_channels=256 + 256 + 128, out_channels=128)
        self.decode2_3 = conv_block(in_channels=128 + 128 + 64, out_channels=64)
        self.decode2_4 = conv_block(in_channels=64 + 64 + 32, out_channels=32)
        self.out = nn.Conv2d(in_channels=32, out_channels=out_channels, kernel_size=1, padding=0)
        
    def forward(self, x_in):
        if x_in.shape[1] == 1:
            x_in = torch.cat((x_in, x_in, x_in), dim=1)
        x_in = self.normalize(x_in)
        x1 = self.vgg1(x_in)
        x = self.act(x1)
        x2 = self.vgg2(x)
        x = self.act(x2)
        x3 = self.vgg3(x)
        x = self.act(x3)
        x4 = self.vgg4(x)
        x = self.act(x4)
        x = self.vgg5(x)
        x = self.act(x)
        x = self.aspp_1(x)
        x1, x2, x3, x4 = self.G_vgg1([x1, x2, x3, x4])
        x12, x22, x32, x42 = self.G_vgg2([x1, x2, x3, x4])
        x = self.up(x)
        x = torch.cat((x4, x), dim=1)
        x, _ = self.decode1(x)
        x = self.up(x)
        x = torch.cat((x3, x), dim=1)
        x, _ = self.decode2(x)
        x = self.up(x)
        x = torch.cat((x2, x), dim=1)
        x, _ = self.decode3(x)
        x = self.up(x)
        x = torch.cat((x1, x), dim=1)
        x, _ = self.decode4(x)
        x = self.out1(x)
        out = x * x_in
        x, x1_2 = self.encode2_1(out)
        x = self.pool(x)
        x, x2_2 = self.encode2_2(x)
        x = self.pool(x)
        x, x3_2 = self.encode2_3(x)
        x = self.pool(x)
        x, x4_2 = self.encode2_4(x)
        x = self.pool(x)
        x = self.aspp_2(x)
        x1_2, x2_2, x3_2, x4_2 = self.GLCSA([x1_2, x2_2, x3_2, x4_2])
        x = self.up(x)
        x = torch.cat((x42, x4_2, x), dim=1)
        x, _ = self.decode2_1(x)
        x = self.up(x)
        x = torch.cat((x32, x3_2, x), dim=1)
        x, _ = self.decode2_2(x)
        x = self.up(x)
        x = torch.cat((x22, x2_2, x), dim=1)
        x, _ = self.decode2_3(x)
        x = self.up(x)
        x = torch.cat((x12, x1_2, x), dim=1)
        x, _ = self.decode2_4(x)
        x = self.out(x)            
        return x

    def normalize(self, x):
        return (x - self.mu) / self.sigma