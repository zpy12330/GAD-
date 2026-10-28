import torch.nn as nn
import torch
import torch.nn.functional as F



class Conv2d_GN(nn.Module):
    def __init__(self, inc, outc, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bias=True):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(inc, outc, ks, stride, pad, dilation, groups, bias),
            nn.GroupNorm(outc//8, outc)
        )
    def forward(self, x):
        return self.block(x)



class MLP(nn.Module):
    def __init__(self, input_dim=2048, embed_dim=768, identity=False):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)
        if identity:
            self.proj = nn.Identity()

    def forward(self, x):
        n, _, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        x = x.permute(0,2,1).reshape(n, -1, h, w)
        
        return x
    

class SRModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        
        self.att = nn.Conv2d(in_channels=dim, out_channels=1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x1, x2, x3, x4):

        _, _, H, W = x1.shape

        att_map = self.sigmoid(self.att(x1))

        x2 = nn.Upsample(size=(H, W), mode="bilinear")(x2)
        x3 = nn.Upsample(size=(H, W), mode="bilinear")(x3)
        x4 = nn.Upsample(size=(H, W), mode="bilinear")(x4)  

        x2 = x2 * att_map
        x3 = x3 * att_map
        x4 = x4 * att_map

        return x2, x3, x4







class DirectionalEnhanceConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.horizontal = nn.Conv2d(dim, dim, kernel_size=(1, 3), padding=(0, 1), groups=dim)
        self.vertical = nn.Conv2d(dim, dim, kernel_size=(3, 1), padding=(1, 0), groups=dim)
        self.diagonal_1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.diagonal_2 = nn.Conv2d(dim, dim, kernel_size=3, padding=2, dilation=2, groups=dim)
        self.fusion = nn.Sequential(
            nn.Conv2d(dim * 4, dim, kernel_size=1),
            nn.GroupNorm(dim // 8, dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x_h = self.horizontal(x)
        x_v = self.vertical(x)
        x_d1 = self.diagonal_1(x)
        x_d2 = self.diagonal_2(x)
        return self.fusion(torch.cat([x_h, x_v, x_d1, x_d2], dim=1))


class LightweightAttention2d(nn.Module):
    def __init__(self, dim, num_heads=4, pool_size=16):
        super().__init__()
        self.num_heads = num_heads
        self.pool_size = pool_size
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.k_proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.v_proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.out_proj = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, query, context=None):
        if context is None:
            context = query

        n, c, h, w = query.shape
        context = F.adaptive_avg_pool2d(context, output_size=(self.pool_size, self.pool_size))

        q = self.q_proj(query).flatten(2).transpose(1, 2)
        k = self.k_proj(context).flatten(2).transpose(1, 2)
        v = self.v_proj(context).flatten(2).transpose(1, 2)

        q = q.reshape(n, h * w, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(n, self.pool_size * self.pool_size, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(n, self.pool_size * self.pool_size, self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.softmax((q @ k.transpose(-2, -1)) * self.scale, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(n, h * w, c)
        out = out.transpose(1, 2).reshape(n, c, h, w)
        return self.out_proj(out)


class SiameseRefinement(nn.Module):
    def __init__(self, dim, num_heads=4, pool_size=16):
        super().__init__()
        self.shared_encoder = Conv2d_GN(inc=dim, outc=dim, ks=3, pad=1)
        self.pos_emb = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.shared_self_attn = LightweightAttention2d(dim, num_heads=num_heads, pool_size=pool_size)
        self.cross_raw = LightweightAttention2d(dim, num_heads=num_heads, pool_size=pool_size)
        self.cross_deg = LightweightAttention2d(dim, num_heads=num_heads, pool_size=pool_size)
        self.gate = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=1),
            nn.Sigmoid()
        )
        self.out_proj = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, f_raw, f_deg):
        f_raw_enc = self.shared_encoder(f_raw) + self.pos_emb
        f_deg_enc = self.shared_encoder(f_deg) + self.pos_emb

        f_raw_s = self.shared_self_attn(f_raw_enc) + f_raw_enc
        f_deg_s = self.shared_self_attn(f_deg_enc) + f_deg_enc

        f_raw_c = self.cross_raw(f_raw_s, f_deg_s) + f_raw_s
        f_deg_c = self.cross_deg(f_deg_s, f_raw_s) + f_deg_s

        gate = self.gate(torch.cat([f_raw_c, f_deg_c], dim=1))
        f_fused = gate * f_raw_c + (1 - gate) * f_deg_c
        return self.out_proj(f_fused + f_raw)



class StableFeatureSelectionMask(nn.Module):
    def __init__(self, dim):
        super().__init__()

        # 保持原有 structure，只改权重预测部分
        self.scale_refine = nn.ModuleList([
            nn.Sequential(
                Conv2d_GN(inc=dim, outc=dim, ks=3, pad=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(dim, 1, kernel_size=1)
            )
            for _ in range(3)
        ])

        # 改动1：去掉自适应池化，改用 depthwise + 全局池化分支并行
        self.weight_predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 全局信息
            nn.Conv2d(dim * 3, max(dim, 3), kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(dim, 3), 3, kernel_size=1)
        )

        # 改动2：增加一个轻量空间路径（depthwise 3x3，不增加参数量）
        self.spatial_path = nn.Sequential(
            nn.Conv2d(dim * 3, dim * 3, kernel_size=3, padding=1, groups=dim * 3),  # depthwise
            nn.Conv2d(dim * 3, 3, kernel_size=1),  # pointwise
        )

        # 改动3：融合时加入 max 和 mean 的统计信息（不增加参数）
        self.fusion = nn.Sequential(
            Conv2d_GN(inc=3, outc=dim, ks=3, pad=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, 1, kernel_size=1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x2, x3, x4, target_size):
        x2 = F.interpolate(x2, size=target_size, mode='bilinear', align_corners=False)
        x3 = F.interpolate(x3, size=target_size, mode='bilinear', align_corners=False)
        x4 = F.interpolate(x4, size=target_size, mode='bilinear', align_corners=False)

        features = [x2, x3, x4]
        scale_logits = [refine(feat) for refine, feat in zip(self.scale_refine, features)]
        scale_logits = torch.cat(scale_logits, dim=1)

        scale_context = torch.cat(features, dim=1)

        # 全局权重
        global_weights = torch.softmax(self.weight_predictor(scale_context), dim=1)

        # 空间权重（轻量）
        spatial_weights = torch.sigmoid(self.spatial_path(scale_context))

        # 融合两种权重（相乘，几乎无开销）
        final_weights = global_weights * spatial_weights
        selected_logits = scale_logits * final_weights

        # 统计增强（不增加参数）
        logits_mean = selected_logits.mean(dim=1, keepdim=True)
        logits_max = torch.max(selected_logits, dim=1, keepdim=True)[0]

        stable_mask = self.sigmoid(self.fusion(selected_logits) + 0.3 * logits_mean + 0.1 * logits_max)
        return stable_mask



class SRFModule(nn.Module):
    def __init__(self, embed_dims, mid_dim=8, size=(512,512)):
        super(SRFModule, self).__init__()
        
        self.size = size

        self.mlp1 = MLP(input_dim=embed_dims[0], embed_dim=mid_dim)
        self.mlp2 = MLP(input_dim=embed_dims[1], embed_dim=mid_dim)
        self.mlp3 = MLP(input_dim=embed_dims[2], embed_dim=mid_dim)
        self.mlp4 = MLP(input_dim=embed_dims[3], embed_dim=mid_dim)

        self.conv1 = Conv2d_GN(inc=mid_dim, outc=mid_dim)
        self.conv2 = Conv2d_GN(inc=mid_dim, outc=mid_dim)
        self.conv3 = Conv2d_GN(inc=mid_dim, outc=mid_dim)
        self.conv4 = Conv2d_GN(inc=mid_dim, outc=mid_dim)

        self.stable_mask_selector = StableFeatureSelectionMask(dim=mid_dim)
        self.sr_module = SRModule(dim=mid_dim)

        self.mask_alpha = nn.Parameter(torch.tensor(0.5))

        self.directional_enhance = DirectionalEnhanceConv(dim=mid_dim)
        self.siamese_refinement = SiameseRefinement(dim=mid_dim, num_heads=4, pool_size=16)


        self.block = nn.Sequential(
            Conv2d_GN(inc=mid_dim*4, outc=mid_dim*4),
            Conv2d_GN(inc=mid_dim*4, outc=mid_dim),
            nn.ReLU()
        )


        self.linear_pred = nn.Sequential(
            nn.Conv2d(mid_dim, mid_dim//4, 1),
            nn.Conv2d(mid_dim//4, mid_dim//4, 1 ,groups=mid_dim//4),
            nn.Conv2d(mid_dim//4, 1, kernel_size=1),
            nn.Conv2d(1, 1, kernel_size=1)
        )
        

    def forward(self, inputs):
 
        
        x1, x2, x3, x4 = inputs # [16, 128, 128] [32, 64, 64] [64, 32, 32] [128, 16, 16]

        x1 = self.mlp1(x1)
        x2 = self.mlp2(x2)
        x3 = self.mlp3(x3)
        x4 = self.mlp4(x4)

        # 保存未经过掩码引导的原始特征，用于保留原始特征支路
        x1_raw = x1.clone()
        x2_raw = x2.clone()
        x3_raw = x3.clone()
        x4_raw = x4.clone()

        # 支路1：掩码引导支路 - x1,x2,x3,x4经过DirectionalEnhanceConv和掩码引导
        stable_mask = self.stable_mask_selector(x2, x3, x4, target_size=x1.shape[2:])
        
        x1_guided = x1 * (1 + self.mask_alpha * (stable_mask - 0.5))
        x1_masked = self.conv1(x1_guided)
        x2_masked = self.conv2(x2)
        x3_masked = self.conv3(x3)
        x4_masked = self.conv4(x4)
        
        x2_masked, x3_masked, x4_masked = self.sr_module(x1_masked, x2_masked, x3_masked, x4_masked)
        
        # 对各尺度特征应用方向增强
        x1_masked = self.directional_enhance(x1_masked)
        x2_masked = self.directional_enhance(x2_masked)
        x3_masked = self.directional_enhance(x3_masked)
        x4_masked = self.directional_enhance(x4_masked)
        
        # 掩码引导支路特征拼接
        x_masked = torch.cat([
            F.interpolate(x1_masked, size=self.size, mode="bilinear", align_corners=False),
            F.interpolate(x2_masked, size=self.size, mode="bilinear", align_corners=False),
            F.interpolate(x3_masked, size=self.size, mode="bilinear", align_corners=False),
            F.interpolate(x4_masked, size=self.size, mode="bilinear", align_corners=False)
        ], dim=1)
        x_masked = self.block(x_masked)

        # 支路2：保留原始特征支路 - 未经过掩码引导的原始特征拼接
        x_raw = torch.cat([
            F.interpolate(x1_raw, size=self.size, mode="bilinear", align_corners=False),
            F.interpolate(x2_raw, size=self.size, mode="bilinear", align_corners=False),
            F.interpolate(x3_raw, size=self.size, mode="bilinear", align_corners=False),
            F.interpolate(x4_raw, size=self.size, mode="bilinear", align_corners=False)
        ], dim=1)
        x_raw = self.block(x_raw)

        # 通过SiameseRefinement融合两条支路
        x_fused = self.siamese_refinement(x_raw, x_masked)

        x = self.linear_pred(x_fused)

        return x
    


