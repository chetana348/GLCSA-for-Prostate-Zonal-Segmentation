import math
import torch
import torch.nn as nn
import einops
import torch.nn.functional as F

class Conv(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=3,stride=1,padding=1,dilation= 
                                   1,apply_norm='bn',activation=True, use_bias = True):
        super(Conv, self).__init__()        
        self.conv = nn.Conv2d (in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,stride=stride,padding=padding,
                               dilation=dilation,bias=True)        
        self.apply_norm = apply_norm
        self.activation = activation
        
        self.norm = nn.GroupNorm(out_channels, out_channels) if self.apply_norm == 'gn' else nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU(inplace=False) if self.activation else False
            

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x) if self.apply_norm is not None else x
        x = self.relu(x) if self.activation else x
        return x

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels1, out_channels2, norm1=None, norm2=None, act1=None, act2=None, activation=False):
        super(DoubleConv, self).__init__()
        if activation:
            self.conv1 = Conv(in_channels=in_channels, out_channels=out_channels1, apply_norm=norm1, activation=act1)
            self.conv2 = Conv(in_channels=out_channels1, out_channels=out_channels2, apply_norm=norm2, activation=act2)
        else:
            self.conv1 = Conv(in_channels=in_channels, out_channels=out_channels1, apply_norm='bn')
            self.conv2 = Conv(in_channels=out_channels1, out_channels=out_channels2, apply_norm='bn')

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x



class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels,activation=True, apply_norm='bn',scale=2, glcsa=False):
        super(UpConv, self).__init__()
        
        self.up = nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=True)
        
        if glcsa:
            self.conv = Conv(in_channels=in_channels, 
                                out_channels=out_channels, 
                                kernel_size=1,
                                padding=0,
                                apply_norm=apply_norm, 
                                activation=activation)
        else:        
            self.conv = Conv(in_channels=in_channels, out_channels=out_channels, apply_norm=apply_norm, activation=activation)

    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        return x
    



class MainConv(nn.Module):
    def __init__(self, 
                in_channels, 
                out_channels, 
                groups,
                kernel_size=1, 
                padding=0, 
                apply_norm=None, 
                activation=False, 
                ):
        super(MainConv, self).__init__()

        self.conv = ConvBlock(in_channels=in_channels, 
                                        out_channels=out_channels, 
                                        kernel_size=kernel_size,
                                        padding=padding,
                                        groups=groups,
                                        apply_norm=apply_norm,
                                        activation=activation)
                            
    def forward(self, x):
        P = int(x.shape[0] ** 0.5)
        x = einops.rearrange(x, 'B (H W) C-> B C H W', H=P) 
        x = self.conv(x)
        x = einops.rearrange(x, 'B C H W -> B (H W) C')      
        return x


class PatchEmbedding(nn.Module):
    def __init__(self,pooling,
                patch_size
                ):
        super(PatchEmbedding, self).__init__()
        self.projection = pooling(output_size=(patch_size, patch_size))

    def forward(self, x):
        x = self.projection(x)
        x = einops.rearrange(x, 'B C H W -> B (H W) C')
        return x

class Embedding(nn.Module):
    def __init__(self, dim=16):
        super(Embedding, self).__init__()
        self.dim = dim

    def forward(self, x):
        # Resize the input tensor to the specified output size
        x_resized = F.interpolate(x, size=self.dim, mode='nearest')
        
        # Rearrange the tensor to have the same output shape
        x_reshaped = einops.rearrange(x_resized, 'B C H W -> B (H W) C')
        
        return x_reshaped
    
    
class DotProduct(nn.Module):
    def __init__(self) -> None:
        super(DotProduct, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
                                                    
    def forward(self, x1, x2, x3, scale):
        x2 = x2.transpose(-2, -1)
        x12 = torch.einsum('bhcw,bhwk->bhck', x1, x2) * scale
        att = self.softmax(x12)
        x123 = torch.einsum('bhcw,bhwk->bhck', att, x3) 
        return x123


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels,kernel_size=3,stride=1,padding=1,dilation=1, groups=None, apply_norm='bn',                                    activation=True):
        super(ConvBlock, self).__init__()
        
        self.apply_norm=apply_norm
        self.activation = activation
        
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels if groups is None else groups,
            dilation=dilation, 
            bias=True)

        self.norm = nn.GroupNorm(out_channels, out_channels) if self.apply_norm == 'gn' else nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU(inplace=False) if self.activation else False

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x) if self.apply_norm is not None else x
        x = self.relu(x) if self.activation else x
        return x


class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction:int=16):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(in_channels, int(in_channels // reduction), bias=False),
                                nn.ReLU(),
                                nn.Linear(int(in_channels // reduction), in_channels, bias=False),
                                nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avgpool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        out = x * y.expand_as(x)
        return out
    
    
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, apply_norm='bn', activation=True, rate=[1, 6, 12, 18]):
        super().__init__()

        self.conv1 = Conv(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding=0, apply_norm=apply_norm,
            activation=activation)

        self.conv2 = Conv(in_channels=in_channels, out_channels=out_channels, padding=rate[0], dilation=rate[0], apply_norm=apply_norm,
            activation=activation, use_bias=False)
        
        self.conv3 = Conv(in_channels=in_channels,out_channels=out_channels,padding=rate[1],dilation=rate[1],apply_norm=apply_norm,
            activation=activation, use_bias=False)
        
        self.conv4 = Conv(in_channels=in_channels, out_channels=out_channels, padding=rate[2], dilation=rate[2],                                     apply_norm=apply_norm, activation=activation, use_bias=False)
        
        self.conv5 = Conv(in_channels=in_channels, out_channels=out_channels, padding=rate[3], dilation=rate[3], apply_norm=apply_norm,
            activation=activation, use_bias=False)

        self.out = Conv(in_channels=out_channels * 5, out_channels=out_channels, kernel_size=1, padding=0, apply_norm=apply_norm,
            activation=activation,  use_bias=False)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        x5 = self.conv5(x)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x = self.out(x)
        return x