import torch
import torch.nn as nn
from scripts.models.utils import *
from scripts.models.GLCSA import *
import einops


class UNet(nn.Module):
    def __init__(self,
                n_blocks=1,
                in_channels=1, 
                out_channels=3, 
                width_factor=0.5,
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
   
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        
        norm2 = None
        self.conv1 = DoubleConv(in_channels=in_channels, 
                                        out_channels1=int(64 * width_factor), 
                                        out_channels2=int(64 * width_factor), 
                                        activation=True,
                                        norm1='bn', 
                                        norm2=norm2, 
                                        act1=True, 
                                        act2=False)
        self.norm1 = nn.BatchNorm2d(int(64 * width_factor))
        
        self.conv2 = DoubleConv(in_channels=int(64 * width_factor), 
                                        out_channels1=int(128 * width_factor), 
                                        out_channels2=int(128 * width_factor), 
                                        activation=True,
                                        norm1='bn', 
                                        norm2=norm2, 
                                        act1=True, 
                                        act2=False)
        self.norm2 = nn.BatchNorm2d(int(128 * width_factor))

        self.conv3 = DoubleConv(in_channels=int(128 * width_factor), 
                                        out_channels1=int(256 * width_factor), 
                                        out_channels2=int(256 * width_factor), 
                                        activation=True,
                                        norm1='bn', 
                                        norm2=norm2, 
                                        act1=True, 
                                        act2=False)
        self.norm3 = nn.BatchNorm2d(int(256 * width_factor))

        self.conv4 = DoubleConv(in_channels=int(256 * width_factor), 
                                        out_channels1=int(512 * width_factor), 
                                        out_channels2=int(512 * width_factor), 
                                        activation=True,
                                        norm1='bn', 
                                        norm2=norm2, 
                                        act1=True, 
                                        act2=False)  
        self.norm4 = nn.BatchNorm2d(int(512 * width_factor))
    
        self.conv5 = DoubleConv(in_channels=int(512 * width_factor), 
                                        out_channels1=int(1024 * width_factor), 
                                        out_channels2=int(1024 * width_factor), 
                                        )   

        
        self.glcsa = GLCSA(n_blocks=n_blocks,                                            
                        features=[int(64 * width_factor), int(128 * width_factor), int(256 * width_factor), int(512 * width_factor)],                                                                                                              
                        strides=[patch_size, patch_size // 2, patch_size // 4, patch_size // 8],
                        patch=patch,
                        emb = emb,
                        fusion=fusion,
                        spatial_att=spatial_att,
                        channel_att=channel_att, 
                        spatial_head=spatial_head_dim,
                        channel_head=channel_head_dim

                        )  
          
        self.up1 = UpConv(in_channels=int(1024 * width_factor), 
                            out_channels=int(512 * width_factor), 
                            apply_norm='bn')

        self.upconv1 =DoubleConv(in_channels=int(512 * width_factor + 512 * width_factor), 
                                        out_channels1=int(512 * width_factor), 
                                        out_channels2=int(512 * width_factor)
                                        )

        self.up2 = UpConv(in_channels=int(512 * width_factor), 
                            out_channels=int(256 * width_factor), 
                            apply_norm='bn')

        self.upconv2 = DoubleConv(in_channels=int(256 * width_factor + 256 * width_factor), 
                                        out_channels1=int(256 * width_factor), 
                                        out_channels2=int(256 * width_factor) 
                                        )

        self.up3 = UpConv(in_channels=int(256 * width_factor), 
                            out_channels=int(128 * width_factor), 
                            apply_norm='bn')

        self.upconv3 = DoubleConv(in_channels=int(128 * width_factor + 128 * width_factor), 
                                        out_channels1=int(128 * width_factor), 
                                        out_channels2=int(128 * width_factor)
                                        )

        self.up4 = UpConv(in_channels=int(128 * width_factor), 
                            out_channels=int(64 * width_factor), 
                            apply_norm='bn')

        self.upconv4 = DoubleConv(in_channels=int(64 * width_factor + 64 * width_factor), 
                                        out_channels1=int(64 * width_factor), 
                                        out_channels2=int(64 * width_factor) 
                                        )    

        self.out = Conv(in_channels=int(64 * width_factor), 
                            out_channels=out_channels, 
                            apply_norm=None,
                            activation=False, 
                            kernel_size=1, 
                            padding=0)   

    def forward(self, x):
        x1 = self.conv1(x)
        x1_n = self.norm1(x1)
        x1_a = self.relu(x1_n)
        
        x2 = self.maxpool(x1_a)
        x2 = self.conv2(x2)
        x2_n = self.norm2(x2)
        x2_a = self.relu(x2_n)
        
        x3 = self.maxpool(x2_a) 
        x3 = self.conv3(x3)
        x3_n = self.norm3(x3)
        x3_a = self.relu(x3_n)
        
        x4 = self.maxpool(x3_a)
        x4 = self.conv4(x4)
        x4_n = self.norm4(x4)
        x4_a = self.relu(x4_n)
        
        x5 = self.maxpool(x4_a)
        x = self.conv5(x5)
        
        x1, x2, x3, x4 = self.glcsa([x1, x2, x3, x4])
        
        x = self.up1(x)
        x = torch.cat((x, x4), dim=1)
        x = self.upconv1(x)
        
        x = self.up2(x)
        x = torch.cat((x, x3), dim=1)
        x = self.upconv2(x)
        
        x = self.up3(x)
        x = torch.cat((x, x2), dim=1)
        x = self.upconv3(x)
        
        x = self.up4(x)
        x = torch.cat((x, x1), dim=1)
        x = self.upconv4(x)
        
        x = self.out(x)
        return x



