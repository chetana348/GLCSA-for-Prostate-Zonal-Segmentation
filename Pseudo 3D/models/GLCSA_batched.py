import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *

import torch
import torch.nn as nn
import torch.nn.functional as F

class GlobalLocalChannelAttention(nn.Module):
    def __init__(self, in_features, out_features, n_heads=4, window_size=1):
        super().__init__()
        self.n_heads = n_heads
        self.window_size = window_size

        # Global attention
        self.q_map_global = MainConv1(in_channels=out_features, 
                                     out_channels=out_features, 
                                     groups=out_features)
        self.k_map_global = MainConv1(in_channels=in_features, 
                                     out_channels=in_features, 
                                     groups=in_features)
        self.v_map_global = MainConv1(in_channels=in_features, 
                                     out_channels=in_features, 
                                     groups=in_features) 

        # Local attention
        if self.window_size > 0:
            self.q_map_local = MainConv1(in_channels=out_features, 
                                        out_channels=out_features, 
                                        groups=out_features)
            self.k_map_local = MainConv1(in_channels=in_features, 
                                        out_channels=in_features, 
                                        groups=in_features)
            self.v_map_local = MainConv1(in_channels=in_features, 
                                        out_channels=in_features, 
                                        groups=in_features) 

        self.projection = MainConv(in_channels=out_features, 
                                   out_channels=out_features, 
                                   groups=out_features)
        self.sdp = DotProduct()        

    def forward(self, x):
        q, k, v = x[0], x[1], x[2]
        b, hw, c = q.shape
        slices = b

        # Initialize accumulators for global and local attention
        global_attention_accum = torch.zeros_like(q)
        local_attention_accum = torch.zeros_like(q)
        
        for i in range(slices):            
            # Global attention
            q_global = self.q_map_global(q[i, :, :])
            k_global = self.k_map_global(k[i,:,:])
            v_global = self.v_map_global(v[i,:,:])
            att_global = self.compute_global_attention(q_global, k_global, v_global)
            
            # Local attention
            if self.window_size > 0:
                att_local = self.compute_local_attention(q[i,:,:], k[i,:,:], v[i,:,:])
            else:
                att_local = 0
            
            # Combine global and local attention
            att = att_global + att_local
            
            # Aggregate the results
            global_attention_accum[i,:,:] = att
            local_attention_accum[i,:,:] = att

        # Final projection
        att = self.projection(global_attention_accum + local_attention_accum)
        #print(att.shape)
        
        return att
    
    def compute_global_attention(self, q, k, v):
        hw, c_q = q.shape
        c = k.shape[1]
        scale = c ** -0.5
        q = q.reshape(hw, self.n_heads, c_q // self.n_heads).permute(1, 0, 2).transpose(1, 2)
        k = k.reshape(hw, self.n_heads, c // self.n_heads).permute(1, 0, 2).transpose(1, 2)
        v = v.reshape(hw, self.n_heads, c // self.n_heads).permute(1,0,2).transpose(1, 2)
        att_global = self.sdp(q, k, v, scale).permute(2, 0, 1).flatten(1)
        return att_global
    
    def compute_local_attention(self, q, k, v):
        q_local = self.q_map_local(q)
        k_local = self.k_map_local(k)
        v_local = self.v_map_local(v)
        hw, c_q = q_local.shape
        c = k_local.shape[1]
        scale = c ** -0.5

        # Extract local patches
        q_local_patches = self.extract_local_patches(q_local)
        k_local_patches = self.extract_local_patches(k_local)
        v_local_patches = self.extract_local_patches(v_local)

        # Reshape and permute for attention computation
        q_local_patches = q_local_patches.reshape(-1, self.n_heads, c_q // self.n_heads).permute(1, 0, 2).transpose(1, 2)
        k_local_patches = k_local_patches.reshape(-1, self.n_heads, c // self.n_heads).permute(1, 0, 2).transpose(1, 2)
        v_local_patches = v_local_patches.reshape(-1, self.n_heads, c // self.n_heads).permute(1, 0, 2).transpose(1, 2)
        
        # Compute local attention
        att_local = self.sdp(q_local_patches, k_local_patches, v_local_patches, scale).permute(2, 0, 1).flatten(1)
        
        return att_local
        
    def extract_local_patches(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        h, w, c = x.shape
        pad = (self.window_size - 1) // 2
        if h < pad or w < pad:
            raise ValueError("Input tensor dimensions are smaller than required for the specified window_size and padding.")
        x = F.pad(x, (pad, pad, pad, pad), mode='constant', value=0)
        x = x.unfold(1, self.window_size, 1).unfold(2, self.window_size, 1)
        x = x.permute(0, 2, 3, 1, 4).reshape(h, w, -1, c)
        return x


    
class GlobalLocalSpatialAttention(nn.Module):
    def __init__(self, in_features, out_features, n_heads=4, global_ratio=0.5):
        super().__init__()
        self.n_heads = n_heads
        self.global_ratio = global_ratio

        # Global Attention
        self.global_q_map = MainConv1(in_channels=in_features, 
                                                 out_channels=in_features, 
                                                 groups=in_features)
        self.global_k_map = MainConv1(in_channels=in_features, 
                                                 out_channels=in_features, 
                                                 groups=in_features)
        self.global_v_map = MainConv1(in_channels=out_features, 
                                                 out_channels=out_features, 
                                                 groups=out_features)       

        # Local Attention
        self.local_q_map = MainConv1(in_channels=in_features, 
                                                out_channels=in_features, 
                                                groups=in_features)
        self.local_k_map = MainConv1(in_channels=in_features, 
                                                out_channels=in_features, 
                                                groups=in_features)
        self.local_v_map = MainConv1(in_channels=out_features, 
                                                out_channels=out_features, 
                                                groups=out_features)       

        # Final projection
        self.projection = MainConv(in_channels=out_features, 
                                               out_channels=out_features, 
                                               groups=out_features)                                             
        self.sdp = DotProduct()        

    def forward(self, x):
        q, k, v = x[0], x[1], x[2]
        
        # Initialize accumulators for global and local attention
        global_attention_accum = torch.zeros_like(v)
        local_attention_accum = torch.zeros_like(v)
        
        for i in range(20):        
            # Global Attention
            global_q = self.global_q_map(q[i,:,:])
            global_k = self.global_k_map(k[i,:,:])
            global_v = self.global_v_map(v[i,:,:])  

            hw, c = global_q.shape
            c_v = global_v.shape[1]
            scale_global = (c // self.n_heads) ** -0.5        
            global_q = global_q.reshape(hw, self.n_heads, c // self.n_heads).permute(1, 0, 2)
            global_k = global_k.reshape(hw, self.n_heads, c // self.n_heads).permute(1, 0, 2)
            global_v = global_v.reshape(hw, self.n_heads, c_v // self.n_heads).permute(1, 0, 2)
            global_att = self.sdp(global_q, global_k, global_v, scale_global).transpose(0,1).flatten(1)    

            # Local Attention
            local_q = self.local_q_map(q[i,:,:])
            local_k = self.local_k_map(k[i,:,:])
            local_v = self.local_v_map(v[i,:,:])  

            hw, c = local_q.shape
            c_v = local_v.shape[1]
            scale_local = (c // self.n_heads) ** -0.5        
            local_q = local_q.reshape(hw, self.n_heads, c // self.n_heads).permute(1, 0, 2)
            local_k = local_k.reshape(hw, self.n_heads, c // self.n_heads).permute(1, 0, 2)
            local_v = local_v.reshape(hw, self.n_heads, c_v // self.n_heads).permute(1, 0, 2)
            local_att = self.sdp(local_q, local_k, local_v, scale_local).transpose(0,1).flatten(1)    

            # Combine Global and Local Attention
            combined_att = (1 - self.global_ratio) * local_att + self.global_ratio * global_att
            
            # Aggregate the results
            global_attention_accum[i,:,:] = combined_att
            local_attention_accum[i,:,:] = combined_att

            # Final projection
        x = self.projection(global_attention_accum + local_attention_accum)
        return x



class GSA_LSA_Block(nn.Module):
    def __init__(self, features, channel_head, spatial_head, enable_spatial_att=True, enable_channel_att=True):
        super().__init__()
        self.enable_channel_att = enable_channel_att
        self.enable_spatial_att = enable_spatial_att
        
        if self.enable_channel_att:
            self.channel_norm = nn.ModuleList([nn.LayerNorm(in_features, eps=1e-6) for in_features in features])   
            self.c_attention = nn.ModuleList([GlobalLocalChannelAttention(in_features=sum(features),
                                                                          out_features=feature,
                                                                          n_heads=head) 
                                              for feature, head in zip(features, channel_head)])
                                        
        if self.enable_spatial_att:
            self.spatial_norm = nn.ModuleList([nn.LayerNorm(in_features, eps=1e-6) for in_features in features])          
            self.s_attention = nn.ModuleList([GlobalLocalSpatialAttention(in_features=sum(features),
                                                                          out_features=feature,
                                                                          n_heads=head) 
                                              for feature, head in zip(features, spatial_head)])

    def forward(self, x):
        if self.enable_channel_att:
            x_ca = self.channel_attention(x)
            x = [xi + xj for xi, xj in zip(x, x_ca)]
            
        if self.enable_spatial_att:
            x_sa = self.spatial_attention(x)
            x = [xi + xj for xi, xj in zip(x, x_sa)]  
        
        return x

    def channel_attention(self, x):
        x_c = [self.channel_norm[i](j) for i, j in enumerate(x)]
        x_cin = self.concatenate(*x_c)
        x_in = [[q, x_cin, x_cin] for q in x_c]
        x_att = [self.c_attention[i](j) for i, j in enumerate(x_in)]
        return x_att    

    def spatial_attention(self, x):
        x_s = [self.spatial_norm[i](j)for i, j in enumerate(x)]
        x_sin = self.concatenate(*x_s)
        x_in = [[x_sin, x_sin, v] for v in x_s]        
        x_att = [self.s_attention[i](j) for i, j in enumerate(x_in)]
        return x_att 

    def concatenate(self, *args):
        return torch.cat((args), dim=2)



class GLCSA(nn.Module):
    def __init__(self, features, fusion, strides, emb='patch', patch=8, channel_att=True, spatial_att=True, n_blocks=1, channel_head=[1, 1, 1, 1], spatial_head=[4, 4, 4, 4]):   #default embedding set to pool
        super().__init__()
        self.n_blocks = n_blocks
        self.features = features
        self.spatial_head = spatial_head
        self.channel_head = channel_head
        self.channel_att = channel_att
        self.spatial_att = spatial_att
        self.patch = patch
        self.emb = emb
        self.fusion = fusion

        if emb=='patch':  
            print('Using patch embedding..')
            self.embed = nn.ModuleList([PatchEmbedding(
                                                pooling=nn.AdaptiveAvgPool2d, patch_size=patch) for _ in features])  
        else:
            print('using default embedding..')
            self.embed = nn.ModuleList([Embedding() for _ in features]) 
            
        self.avg_map = nn.ModuleList([MainConv(
                                            in_channels=feature,
                                            out_channels=feature, 
                                            kernel_size=1,
                                            padding=0, 
                                            groups=feature
                                            )
                                            for feature in features])         
                            
        if fusion == 'summation':
            print('Using summation based fusion...')
            self.attention = nn.ModuleList([
                GSA_LSA_Block(features=features, 
                              channel_head=channel_head, 
                              spatial_head=spatial_head, 
                              enable_channel_att=channel_att, 
                              enable_spatial_att=spatial_att) 
                              for _ in range(n_blocks)])
            
        elif fusion == 'sequential':
            print('Using Sequential based fusion...')
            self.channel_attention = nn.ModuleList([
                GSA_LSA_Block(features=features, 
                              channel_head=channel_head, 
                              spatial_head=[1] * len(features), 
                              enable_channel_att=channel_att, 
                              enable_spatial_att=False) 
                              for _ in range(n_blocks)])
            self.spatial_attention = nn.ModuleList([
                GSA_LSA_Block(features=features, 
                              channel_head=[1] * len(features), 
                              spatial_head=spatial_head, 
                              enable_channel_att=False, 
                              enable_spatial_att=spatial_att) 
                              for _ in range(n_blocks)])

        if fusion == 'parallel':
            print('Using parallel based fusion...')
            self.channel_attention = nn.ModuleList([
                GSA_LSA_Block(features=features, 
                              channel_head=channel_head, 
                              spatial_head=[1] * len(features), 
                              enable_channel_att=channel_att, 
                              enable_spatial_att=False) 
                              for _ in range(n_blocks)])
            self.spatial_attention = nn.ModuleList([
                GSA_LSA_Block(features=features, 
                              channel_head=[1] * len(features), 
                              spatial_head=spatial_head, 
                              enable_channel_att=False, 
                              enable_spatial_att=spatial_att) 
                              for _ in range(n_blocks)])
            
  
            
        self.upconvs = nn.ModuleList([UpConv(in_channels=feature, 
                                                    out_channels=feature,
                                                    apply_norm=None,
                                                    activation=False,
                                                    scale=stride,
                                                    glcsa=True
                                                    )
                                                    for feature, stride in zip(features, strides)])                                                      
        self.bn_relu = nn.ModuleList([nn.Sequential(
                                            nn.BatchNorm2d(feature), 
                                            nn.ReLU()
                                            ) 
                                            for feature in features])
    
    def forward(self, raw):
        x = [self.embed[i](j) for i, j in enumerate(raw)]
        x = [self.avg_map[i](j) for i, j in enumerate(x)]
        
        if self.fusion == 'summation':
            for block in self.attention:
                x = block(x)
                    
        elif self.fusion == 'sequential':
            for i in range(self.n_blocks):
                x = self.channel_attention[i](x)
                x = self.spatial_attention[i](x)

        elif self.fusion == 'parallel':
            for i in range(self.n_blocks):
                channel_out = self.channel_attention[i](x)
                spatial_out = self.spatial_attention[i](x)
                x = [co + so for co, so in zip(channel_out, spatial_out)]
                
        x = [self.reshape(i) for i in x]
        x = [self.upconvs[i](j) for i, j in enumerate(x)]
        


        x_out = [xi + xj for xi, xj in zip(x, raw)]
            
        x_out = [self.bn_relu[i](j) for i, j in enumerate(x_out)]
        return (*x_out, )
        
    def reshape1(self, x):
        return einops.rearrange(x, 'B (H W) C ->B C H W', H=self.patch)
    def reshape(self, x):
        return einops.rearrange(x, 'B (H W) C -> B C H W', H=self.patch)