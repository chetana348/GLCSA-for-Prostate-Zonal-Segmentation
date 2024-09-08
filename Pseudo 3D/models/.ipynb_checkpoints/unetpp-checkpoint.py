import torch
import torch.nn as nn
from scripts.models.utils import *
from scripts.models.GLCSA_ps import *
import einops


class UNetPlusPlus(nn.Module):
    def __init__(self,
                n_blocks=1,
                in_channels=1, 
                out_channels=2, 
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
                ) -> None:
        super().__init__()
   
        patch = input_size[0] // patch_size
   
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.relu = nn.ReLU()
        norm2 = None
        self.conv1 = DoubleConv(in_channels=in_channels, 
                                activation = True,
                                        out_channels1=int(64 * k), 
                                        out_channels2=int(64 * k), 
                                        norm1='bn', 
                                        norm2=norm2, 
                                        act1=True, 
                                        act2=False)
        self.norm1 = nn.BatchNorm2d(int(64 * k))
        self.conv2 = DoubleConv(in_channels=int(64 * k), 
                                        out_channels1=int(128 * k), 
                                        out_channels2=int(128 * k), 
                                        norm1='bn', 
                                        norm2=norm2,
                                        activation = True,
                                        act1=True, 
                                        act2=False)
        self.norm2 = nn.BatchNorm2d(int(128 * k))

        self.conv3 = DoubleConv(in_channels=int(128 * k), 
                                        out_channels1=int(256 * k), 
                                        out_channels2=int(256 * k), 
                                        norm1='bn', 
                                        norm2=norm2, 
                                        activation = True,
                                        act1=True, 
                                        act2=False)
        self.norm3 = nn.BatchNorm2d(int(256 * k))

        self.conv4 = DoubleConv(in_channels=int(256 * k), 
                                        out_channels1=int(512 * k), 
                                        out_channels2=int(512 * k), 
                                        norm1='bn', 
                                        norm2=norm2, 
                                        act1=True,
                                        activation = True,
                                        act2=False)  
        self.norm4 = nn.BatchNorm2d(int(512 * k))
    
        self.conv5 = DoubleConv(in_channels=int(512 * k), 
                                        out_channels1=int(1024 * k), 
                                        out_channels2=int(1024 * k), 
                                        )   

        self.glcsa = GLCSA(n_blocks=n_blocks,                                            
                                features = [int(64 * k), int(128 * k), int(256 * k), int(512 * k)],                                                                                                              
                                strides=[patch_size, patch_size // 2, patch_size // 4, patch_size // 8],
                                patch=patch,
                                spatial_att=spatial_att,
                                channel_att=channel_att,
                                emb = emb,
                                fusion = fusion,                             
                                spatial_head=spatial_head_dim,
                                channel_head=channel_head_dim,
                                                            )  

        # Define skip connections
        self.up_convs = nn.ModuleList([
            UpConv(in_channels=int(1024 * k), out_channels=int(512 * k), apply_norm='bn'),
            UpConv(in_channels=int(512 * k), out_channels=int(256 * k), apply_norm='bn'),
            UpConv(in_channels=int(256 * k), out_channels=int(128 * k), apply_norm='bn'),
            UpConv(in_channels=int(128 * k), out_channels=int(64 * k), apply_norm='bn')
        ])

        self.convs = nn.ModuleList([
            DoubleConv(in_channels=int(1024 * k), out_channels1=int(512 * k), out_channels2=int(512 * k)),
            DoubleConv(in_channels=int(512 * k), out_channels1=int(256 * k), out_channels2=int(256 * k)),
            DoubleConv(in_channels=int(256 * k), out_channels1=int(128 * k), out_channels2=int(128 * k)),
            DoubleConv(in_channels=int(128 * k), out_channels1=int(64 * k), out_channels2=int(64 * k))
        ])

        self.out = Conv(in_channels=int(64 * k), out_channels=out_channels, apply_norm=None, activation=False,  kernel_size=1, padding=0)   

        # self.initialize_weights()                                     

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

        # Applying U-Net++ skip connections
        skips = [x1, x2, x3, x4]
        for i, (up_conv, conv, skip) in enumerate(zip(self.up_convs, self.convs, reversed(skips))):
            x = up_conv(x)
            x = torch.cat((x, skip), dim=1)
            x = conv(x)

        x = self.out(x)
        return x

if __name__ == '__main__':
    model = UNetPlusPlus()
    in1 = torch.rand(1, 1, 128, 128)
    out = model(in1)
    total_params = sum(p.numel() for p in model.parameters())
    print(total_params)
    print(out.shape)