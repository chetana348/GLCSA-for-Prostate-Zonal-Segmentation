import torch
import torch.nn as nn
from scripts.models.utils import *
from scripts.models.GLCSA_ps import *

class ResConv(nn.Module):
    def __init__(self, in_channels, features):
        super().__init__()
        f1 = int(1.67 * features * 0.167)
        f2 = int(1.67 * features * 0.333)
        f3 = int(1.67 * features * 0.5)
        fout = f1 + f2 + f3

        self.skip = Conv(in_channels=in_channels, out_channels=fout, kernel_size=1, padding=0,apply_norm='bn',activation=False)
        self.conv1 = Conv(in_channels=in_channels,out_channels=f1,kernel_size=3, padding=1,apply_norm='bn',activation=True)
        self.conv2 = Conv(in_channels=f1,out_channels=f2,kernel_size=3,padding=1,apply_norm='bn',activation=True)
        self.conv3 = Conv(in_channels=f2, out_channels=f3, kernel_size=3, padding=1,apply_norm='bn',activation=True)
        self.norm = nn.BatchNorm2d(fout)
        self.act = nn.ReLU()

    def forward(self, x):
        x_skip = self.skip(x)
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x = torch.cat((x1, x2, x3), dim=1)
        x = self.norm(x)
        x += x_skip
        x = self.act(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_layers):
        super().__init__()
        self.n_layers = n_layers

        self.norm = nn.ModuleList([nn.BatchNorm2d(out_channels) for _ in range(n_layers)])

        self.skip = nn.ModuleList([Conv(in_channels=in_channels, 
                                                out_channels=out_channels, 
                                                kernel_size=1, 
                                                padding=0, 
                                                apply_norm=None, 
                                                activation=False)])   

        self.conv = nn.ModuleList([Conv(in_channels=in_channels, 
                                                out_channels=out_channels, 
                                                kernel_size=3, 
                                                padding=1, 
                                                apply_norm=None, 
                                                activation=False), 
                                                ])

        for _ in range(n_layers - 1):
            self.skip.append(Conv(in_channels=out_channels, 
                                                    out_channels=out_channels, 
                                                    kernel_size=1, 
                                                    padding=0, 
                                                    apply_norm=None, 
                                                    activation=False))                                                
            self.conv.append(Conv(in_channels=out_channels, 
                                                    out_channels=out_channels, 
                                                    kernel_size=3, 
                                                    padding=1, 
                                                    apply_norm=None, 
                                                    activation=False))    
        self.act = nn.ReLU()

    def forward(self, x):
        for i in range(self.n_layers):
            x_skip = self.skip[i](x)
            x = self.conv[i](x)
            x_s = x + x_skip
            x = self.norm[i](x_s)
            x = self.act(x)
        return x, x_s

class ResUNet(nn.Module):
    def __init__(self,
                n_blocks=1,
                in_channels=1, 
                out_channels=3, 
                k=1,
                input_size=(128, 128),
                patch_size=8,
                spatial_att=False,
                channel_att=False,
                spatial_head_dim=[4, 4, 4, 4],
                channel_head_dim=[1, 1, 1, 1],
                emb = 'patch',
                fusion = 'summation',
                device='cuda', 
                ):
        super().__init__()
        
        patch = input_size[0] // patch_size

        alpha = 1.67
        k = 1
        features1 = int(32*alpha*0.167)+int(32*alpha*0.333)+int(32*alpha* 0.5)                                
        features2 = int(32*2*alpha*0.167)+int(32*2*alpha*0.333)+int(32*2*alpha* 0.5)
        features3 = int(32*4*alpha*0.167)+int(32*4*alpha*0.333)+int(32*4*alpha* 0.5)
        features4 = int(32*8*alpha*0.167)+int(32*8*alpha*0.333)+int(32*8*alpha* 0.5)
        features5 = int(32*16*alpha*0.167)+int(32*16*alpha*0.333)+int(32*16*alpha* 0.5)        
        features6 = int(32*8*alpha*0.167)+int(32*8*alpha*0.333)+int(32*8*alpha* 0.5)        
        features7 = int(32*4*alpha*0.167)+int(32*4*alpha*0.333)+int(32*4*alpha* 0.5)        
        features8 = int(32*2*alpha*0.167)+int(32*2*alpha*0.333)+int(32*2*alpha* 0.5)        
        features9 = int(32*alpha*0.167)+int(32*alpha*0.333)+int(32*alpha* 0.5)        

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.res1 = ResConv(in_channels=in_channels, features=int(32 * k))
        self.block1 = ResBlock(in_channels=features1, out_channels=32, n_layers=4)
        self.res2 = ResConv(in_channels=features1, features=int(32 * k * 2))
        self.block2 = ResBlock(in_channels=features2, out_channels=32 * 2, n_layers=3)
        self.res3 = ResConv(in_channels=features2, features=int(32 * k * 4))
        self.block3 = ResBlock(in_channels=features3, out_channels=32 * 4, n_layers=2)
        self.res4 = ResConv(in_channels=features3, features=int(32 * k * 8))
        self.block4 = ResBlock(in_channels=features4,  out_channels=32 * 8, n_layers=1)
        self.res5 = ResConv(in_channels=features4, features=int(32 * k * 16))

        self.glcsa = GLCSA(n_blocks=n_blocks,                                            
                                features = [int(32 * k), int(64 * k), int(128 * k), int(256 * k)],                                                                                                              
                                strides=[patch_size, patch_size // 2, patch_size // 4, patch_size // 8],
                                patch=patch,
                                emb = emb,
                                fusion = fusion,
                                spatial_att=spatial_att,
                                channel_att=channel_att, 
                                spatial_head=spatial_head_dim,
                                channel_head=channel_head_dim,
                                                            ) 

        self.up1 = nn.Sequential(nn.ConvTranspose2d(features5, 32*8, kernel_size=2,stride=2), nn.BatchNorm2d(32 * 8),nn.ReLU())

        self.res6 = ResConv(in_channels=32 * 8 * 2, features=int(32 * k * 8))  

        self.up2 = nn.Sequential(nn.ConvTranspose2d(features6,32*4,kernel_size=2,stride=2), nn.BatchNorm2d(32 * 4),nn.ReLU())     

        self.res7 = ResConv(in_channels=32 * 4 * 2, features=int(32 * k * 4)) 

        self.up3 = nn.Sequential(nn.ConvTranspose2d(features7, 32*2,kernel_size=2,stride=2),nn.BatchNorm2d(32 * 2), nn.ReLU())    
                                                      
        self.res8 = ResConv(in_channels=32 * 2 * 2, features=int(32 * k * 2)) 
        
        self.up4 = nn.Sequential(nn.ConvTranspose2d(features8,32,kernel_size=2,stride=2),nn.BatchNorm2d(32), nn.ReLU())   
  
        self.res9 = ResConv(in_channels=32 * 2, features=int(32 * k))    

        self.out = nn.Conv2d(in_channels=features9, out_channels=out_channels, kernel_size=1, padding=0)                  

    def forward(self, x):
        x1 = self.res1(x)
        xp1 = self.maxpool(x1)
        x1, x1_ = self.block1(x1)     
        x2 = self.res2(xp1)
        xp2 = self.maxpool(x2)
        x2, x2_ = self.block2(x2)   

        x3 = self.res3(xp2)
        xp3 = self.maxpool(x3)
        x3, x3_ = self.block3(x3)   

        x4 = self.res4(xp3)
        xp4 = self.maxpool(x4)
        x4, x4_ = self.block4(x4)  

        x = self.res5(xp4)
        
        x1, x2, x3, x4 = self.glcsa([x1_, x2_, x3_, x4_])

        x = self.up1(x)
        x = torch.cat((x, x4), dim=1)
        x = self.res6(x)
        x = self.up2(x)
        x = torch.cat((x, x3), dim=1)
        x = self.res7(x)
        x = self.up3(x)
        x = torch.cat((x, x2), dim=1)
        x = self.res8(x)
        x = self.up4(x)
        x = torch.cat((x, x1), dim=1)
        x = self.res9(x)
        x = self.out(x)
        return x

        