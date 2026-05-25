import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from torch.autograd import Variable
from torch.distributions import Normal, Independent, kl
from PIL import Image
import warnings
try:
    from .backbones.vmamba import Backbone_VSSM
    _VMAMBA_IMPORT_ERROR = None
except ImportError as exc:
    Backbone_VSSM = None
    _VMAMBA_IMPORT_ERROR = exc
from .backbones.cnn_adapters import (
    DEFAULT_OUT_CHANNELS,
    SiameseConvNeXtAdapter,
    SiameseResNetAdapter,
)
warnings.filterwarnings('ignore')

class SiameseVMambaAdapter(nn.Module):

    def __init__(self, pretrained_path=None):
        super(SiameseVMambaAdapter, self).__init__()
        if Backbone_VSSM is None:
            raise ImportError(f"Failed to import VMamba backbone dependencies: {_VMAMBA_IMPORT_ERROR}")
        self.out_channels = DEFAULT_OUT_CHANNELS
        if pretrained_path is None:
            pretrained_path = '/mnt/ssd/sdb/Datasets/SARCD/pretrained/vssm_small_0229_ckpt_epoch_222.pth'
        
        self.backbone = Backbone_VSSM(
            out_indices=(0, 1, 2, 3),   
            pretrained=None,             
            norm_layer="ln2d",           
            depths=[2, 2, 15, 2],       
            dims=96,                    
            ssm_d_state=1,               
            ssm_ratio=2.0,               
            ssm_dt_rank="auto",
            ssm_act_layer="silu",
            ssm_conv=3,
            ssm_conv_bias=False,        
            ssm_drop_rate=0.0,
            ssm_init="v0",
            forward_type="v05_noz",      
            mlp_ratio=4.0,
            mlp_act_layer="gelu",
            mlp_drop_rate=0.0,
            drop_path_rate=0.3,          
            patch_norm=True,
            downsample_version="v3",     
            patchembed_version="v2",     
            use_checkpoint=False,
            imgsize=256,                
        )
        
        self._load_pretrained_weights(pretrained_path)

    def _load_pretrained_weights(self, ckpt_path):
        
        try:
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
            
            model_dict = self.backbone.state_dict()
            
            filtered_dict = {}
            skipped_keys = []
            for k, v in state_dict.items():
                if 'classifier' in k or 'head' in k:
                    skipped_keys.append(f"{k} (classifier)")
                    continue
                    
                if k in model_dict:
                    if v.shape == model_dict[k].shape:
                        filtered_dict[k] = v
                    else:
                        skipped_keys.append(f"{k} (shape mismatch: {v.shape} vs {model_dict[k].shape})")
                else:
                    skipped_keys.append(f"{k} (not in model)")

            model_dict.update(filtered_dict)
            self.backbone.load_state_dict(model_dict)
            
            print(f"Successfully loaded {len(filtered_dict)}/{len(state_dict)} pretrained weights")
                    
        except Exception as e:
            print(f"Failed to load pretrained weights: {e}")
            print("Initializing backbone randomly.")

    def forward(self, x):

        return self.backbone(x)


def build_siamese_backbone(backbone='vmamba', pretrained_path=None, out_channels=DEFAULT_OUT_CHANNELS):
    backbone = backbone.lower()

    if backbone in ('vmamba', 'vssm'):
        if tuple(out_channels) != DEFAULT_OUT_CHANNELS:
            raise ValueError("VMamba adapter currently uses fixed output channels (96, 192, 384, 768).")
        return SiameseVMambaAdapter(pretrained_path=pretrained_path)

    resnet_names = {'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'}
    if backbone in resnet_names:
        return SiameseResNetAdapter(
            arch=backbone,
            out_channels=out_channels,
            pretrained_path=pretrained_path,
        )

    convnext_aliases = {
        'convnext': 'convnext_tiny',
        'convnext_t': 'convnext_tiny',
        'convnext_tiny': 'convnext_tiny',
        'convnext_s': 'convnext_small',
        'convnext_small': 'convnext_small',
        'convnext_b': 'convnext_base',
        'convnext_base': 'convnext_base',
        'convnext_l': 'convnext_large',
        'convnext_large': 'convnext_large',
    }
    if backbone in convnext_aliases:
        return SiameseConvNeXtAdapter(
            arch=convnext_aliases[backbone],
            out_channels=out_channels,
            pretrained_path=pretrained_path,
        )

    raise ValueError(
        f"Unsupported backbone '{backbone}'. "
        "Use vmamba, resnet18/34/50/101/152, or convnext_tiny/small/base/large."
    )


class HyperDUM(nn.Module):
    def __init__(
        self,
        ndf,
        latent_dim=None,
        feat_channels=768,
        hyper_dim=2048,
        num_patches=16,
        momentum=0.99,
        image_change_ratio_threshold=1e-3,
        min_change_pixels=4,
    ):
        super(HyperDUM, self).__init__()
        self.feat_channels = feat_channels   
        self.hyper_dim = hyper_dim           
        self.num_patches = num_patches       
        self.momentum = momentum            
        self.num_labels = 2                  
        self.angular_weight = 0.35           
        self.image_change_ratio_threshold = image_change_ratio_threshold
        self.min_change_pixels = min_change_pixels

        p_side = int(num_patches ** 0.5)
        assert p_side * p_side == num_patches, "num_patches 须为完全平方数 (e.g. 1,4,16,64)"
        self.p_side = p_side

        Phi = torch.empty(hyper_dim, feat_channels)
        nn.init.orthogonal_(Phi)
        self.register_buffer("Phi", Phi)   # [d, C], 固定 buffer

        proto_cpb = torch.empty(self.num_labels, hyper_dim, feat_channels)
        for l in range(self.num_labels):
            nn.init.orthogonal_(proto_cpb[l])  # 每个标签独立正交初始化
        self.register_buffer("proto_cpb", proto_cpb)
        self.register_buffer("proto_cpb_logvar", torch.full_like(proto_cpb, -2.0))
        proto_ppb = torch.empty(self.num_labels, hyper_dim, num_patches)
        for l in range(self.num_labels):
            nn.init.orthogonal_(proto_ppb[l])
        self.register_buffer("proto_ppb", proto_ppb)
        self.register_buffer("proto_ppb_logvar", torch.full_like(proto_ppb, -2.0))

        self.register_buffer("proto_initialized", torch.zeros(self.num_labels, dtype=torch.bool))

        self.uncertainty_mixer = nn.Sequential(
            nn.Conv2d(2, ndf, kernel_size=1),
            nn.GroupNorm(num_groups=min(8, ndf), num_channels=ndf),
            nn.GELU(),
            nn.Conv2d(ndf, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def _cpb_project(self, feat):

        z_pooled = feat.mean(dim=[2, 3])                              # [B, C]

        h_cpb = self.Phi.unsqueeze(0) * z_pooled.unsqueeze(1)        # [B, d, C]

        h_cpb = F.normalize(h_cpb, p=2, dim=1)                       # [B, d, C]
        return h_cpb

    def _ppb_project(self, feat):

        B, C, H, W = feat.shape
        P = self.num_patches
        # Step1: Adaptive pool 到 P 个块 → [B, C, p_side, p_side] → [B, C, P]
        z_patch = F.adaptive_avg_pool2d(feat, (self.p_side, self.p_side))  # [B, C, ps, ps]
        z_patch = z_patch.reshape(B, C, P)                                 # [B, C, P]
        # Step2: 矩阵乘法: Phi[d,C] @ z_patch[B,C,P] → [B, d, P]
        h_ppb = torch.einsum('dc,bcp->bdp', self.Phi, z_patch)             # [B, d, P]
        # Step3: 对每个块的 d 维超向量做 L2 归一化
        h_ppb = F.normalize(h_ppb, p=2, dim=1)                             # [B, d, P]
        return h_ppb

    def _image_labels_from_gt(self, gt_labels):

        if gt_labels is None:
            return None
        if gt_labels.dim() == 3:
            gt = gt_labels.unsqueeze(1)
        else:
            gt = gt_labels
        gt = (gt.float() > 0.5).float()
        num_pixels = gt[0].numel()
        change_pixels = gt.flatten(1).sum(dim=1)
        change_ratio = change_pixels / float(num_pixels)
        ratio_threshold = max(
            float(self.image_change_ratio_threshold),
            float(self.min_change_pixels) / float(num_pixels),
        )
        return (change_ratio >= ratio_threshold).long()

    def _uncertainty_cpb(self, h_cpb):

        proto_mu = self.proto_cpb
        proto_var = self.proto_cpb_logvar.exp().clamp(min=1e-6, max=10.0)
        diff = h_cpb.unsqueeze(1) - proto_mu.unsqueeze(0)            
        mahal = (diff.pow(2) / proto_var.unsqueeze(0)).mean(dim=2)   
        best_mahal, _ = mahal.min(dim=1)                              
        dist_unc = 1.0 - torch.exp(-best_mahal)                       

        proto_norm = proto_mu.norm(dim=1, keepdim=True).clamp(min=1e-8)
        protos = proto_mu / proto_norm
        sims = torch.einsum('bdc,ldc->blc', h_cpb, protos)            
        top2 = torch.topk(sims, k=min(2, self.num_labels), dim=1).values
        if top2.shape[1] == 2:
            gap = (top2[:, 0, :] - top2[:, 1, :]).abs()               
        else:
            gap = torch.ones_like(top2[:, 0, :]) * 2.0
        ang_unc = 1.0 - (gap / 2.0).clamp(0, 1)                      

        unc = ((1.0 - self.angular_weight) * dist_unc + self.angular_weight * ang_unc).clamp(0, 1)
        return unc

    def _uncertainty_ppb(self, h_ppb):

        proto_mu = self.proto_ppb
        proto_var = self.proto_ppb_logvar.exp().clamp(min=1e-6, max=10.0)
        diff = h_ppb.unsqueeze(1) - proto_mu.unsqueeze(0)            
        mahal = (diff.pow(2) / proto_var.unsqueeze(0)).mean(dim=2)   
        best_mahal, _ = mahal.min(dim=1)                              
        dist_unc = 1.0 - torch.exp(-best_mahal)

        proto_norm = proto_mu.norm(dim=1, keepdim=True).clamp(min=1e-8)
        protos = proto_mu / proto_norm
        sims = torch.einsum('bdp,ldp->blp', h_ppb, protos)            # [B, L, P]
        top2 = torch.topk(sims, k=min(2, self.num_labels), dim=1).values
        if top2.shape[1] == 2:
            gap = (top2[:, 0, :] - top2[:, 1, :]).abs()
        else:
            gap = torch.ones_like(top2[:, 0, :]) * 2.0
        ang_unc = 1.0 - (gap / 2.0).clamp(0, 1)

        unc = ((1.0 - self.angular_weight) * dist_unc + self.angular_weight * ang_unc).clamp(0, 1)
        return unc

    def _bundling_update(self, h_cpb, h_ppb, gt_labels=None, confidence_mask=None, epistemic_map=None):

        with torch.no_grad():
            conf_score = None
            epi_score = None
            if confidence_mask is not None:
                conf_score = confidence_mask.float().mean(dim=[1, 2, 3])  # [B]
            if epistemic_map is not None:
                epi_score = epistemic_map.float().mean(dim=[1, 2, 3])      # [B]

            if gt_labels is not None:

                sample_labels = self._image_labels_from_gt(gt_labels)  # [B]
                for l in range(self.num_labels):
                    idx = (sample_labels == l).nonzero(as_tuple=True)[0]
                    if conf_score is not None:
                        idx = idx[conf_score[idx] > 0.02]
                    if len(idx) == 0:
                        continue
                    is_first_l = not self.proto_initialized[l].item()
                    local_mom = 0.0 if is_first_l else self.momentum

                    if epi_score is not None and not is_first_l:
                        novelty = epi_score[idx].mean().item()
                        if novelty > 0.6:
                            local_mom = max(0.90, self.momentum - 0.05)

                    batch_cpb = h_cpb[idx].mean(0)              # [d, C]
                    batch_cpb_var = h_cpb[idx].var(0, unbiased=False).clamp(min=1e-6)
                    self.proto_cpb[l].mul_(local_mom).add_(
                        batch_cpb, alpha=1.0 - local_mom)
                    cpb_var = self.proto_cpb_logvar[l].exp()
                    cpb_var.mul_(local_mom).add_(batch_cpb_var, alpha=1.0 - local_mom)
                    self.proto_cpb_logvar[l].copy_(cpb_var.clamp(min=1e-6, max=10.0).log())

                    batch_ppb = h_ppb[idx].mean(0)              # [d, P]
                    batch_ppb_var = h_ppb[idx].var(0, unbiased=False).clamp(min=1e-6)
                    self.proto_ppb[l].mul_(local_mom).add_(
                        batch_ppb, alpha=1.0 - local_mom)
                    ppb_var = self.proto_ppb_logvar[l].exp()
                    ppb_var.mul_(local_mom).add_(batch_ppb_var, alpha=1.0 - local_mom)
                    self.proto_ppb_logvar[l].copy_(ppb_var.clamp(min=1e-6, max=10.0).log())
                    if is_first_l:
                        self.proto_initialized[l] = True
            else:

                if conf_score is not None:
                    idx = (conf_score > 0.05).nonzero(as_tuple=True)[0]
                    if len(idx) > 0:
                        batch_cpb = h_cpb[idx].mean(0)          # [d, C]
                        batch_ppb = h_ppb[idx].mean(0)          # [d, P]
                    else:
                        batch_cpb = h_cpb.mean(0)
                        batch_ppb = h_ppb.mean(0)
                else:
                    batch_cpb = h_cpb.mean(0)                   # [d, C]
                    batch_ppb = h_ppb.mean(0)                   # [d, P]
                for l in range(self.num_labels):
                    is_first_l = not self.proto_initialized[l].item()
                    local_mom = 0.0 if is_first_l else self.momentum
                    self.proto_cpb[l].mul_(local_mom).add_(
                        batch_cpb, alpha=1.0 - local_mom)
                    cpb_var = self.proto_cpb_logvar[l].exp()
                    cpb_var.mul_(local_mom).add_(
                        h_cpb.var(0, unbiased=False).clamp(min=1e-6), alpha=1.0 - local_mom
                    )
                    self.proto_cpb_logvar[l].copy_(cpb_var.clamp(min=1e-6, max=10.0).log())
                    self.proto_ppb[l].mul_(local_mom).add_(
                        batch_ppb, alpha=1.0 - local_mom)
                    ppb_var = self.proto_ppb_logvar[l].exp()
                    ppb_var.mul_(local_mom).add_(
                        h_ppb.var(0, unbiased=False).clamp(min=1e-6), alpha=1.0 - local_mom
                    )
                    self.proto_ppb_logvar[l].copy_(ppb_var.clamp(min=1e-6, max=10.0).log())
                    if is_first_l:
                        self.proto_initialized[l] = True

    def compute_hyper_vectors(self, feat_diff):

        h_cpb = self._cpb_project(feat_diff)
        h_ppb = self._ppb_project(feat_diff)
        return h_cpb, h_ppb

    def prototype_separation_loss(self, h_cpb, h_ppb, gt_labels, sim_target=-0.6):

        if gt_labels is None:
            zero = h_cpb.new_zeros(())
            return zero, {'cpb_sim': zero.detach(), 'ppb_sim': zero.detach()}

        sample_labels = self._image_labels_from_gt(gt_labels)
        idx0 = (sample_labels == 0).nonzero(as_tuple=True)[0]
        idx1 = (sample_labels == 1).nonzero(as_tuple=True)[0]

        zero = h_cpb.new_zeros(())
        if len(idx0) == 0 or len(idx1) == 0:
            return zero, {'cpb_sim': zero.detach(), 'ppb_sim': zero.detach()}

        cpb_0 = F.normalize(h_cpb[idx0].mean(0).flatten(), p=2, dim=0)
        cpb_1 = F.normalize(h_cpb[idx1].mean(0).flatten(), p=2, dim=0)
        ppb_0 = F.normalize(h_ppb[idx0].mean(0).flatten(), p=2, dim=0)
        ppb_1 = F.normalize(h_ppb[idx1].mean(0).flatten(), p=2, dim=0)

        cpb_sim = (cpb_0 * cpb_1).sum()
        ppb_sim = (ppb_0 * ppb_1).sum()
        loss = 0.5 * (
            F.relu(cpb_sim - sim_target) +
            F.relu(ppb_sim - sim_target)
        )
        return loss, {'cpb_sim': cpb_sim.detach(), 'ppb_sim': ppb_sim.detach()}

    def forward(self, feat_diff, confidence_mask=None, gt_labels=None, update_prototypes=True):

        B, C, H, W = feat_diff.shape

        h_cpb, h_ppb = self.compute_hyper_vectors(feat_diff)

        unc_cpb = self._uncertainty_cpb(h_cpb)                   # [B, C]
        unc_ppb = self._uncertainty_ppb(h_ppb)                   # [B, P]

        unc_weight = unc_cpb.unsqueeze(-1).unsqueeze(-1)               # [B, C, 1, 1]
        feat_sq = feat_diff.pow(2)                                      # [B, C, H, W]
        unc_cpb_map = (feat_sq * unc_weight).sum(dim=1, keepdim=True) / \
                      (feat_sq.sum(dim=1, keepdim=True) + 1e-8)        # [B, 1, H, W] ∈ [0, 1]
        unc_ppb_map = unc_ppb.view(B, 1, self.p_side, self.p_side)
        unc_ppb_map = F.interpolate(unc_ppb_map, size=(H, W), mode='bilinear', align_corners=False)

        epistemic_raw = torch.cat([unc_cpb_map, unc_ppb_map], dim=1)   # [B, 2, H, W]
        epistemic_map = self.uncertainty_mixer(epistemic_raw)           # [B, 1, H, W]

        if self.training and update_prototypes:
            self._bundling_update(
                h_cpb.detach(), h_ppb.detach(), gt_labels,
                confidence_mask=confidence_mask, epistemic_map=epistemic_map.detach()
            )

        return epistemic_map


class UCGF_Module(nn.Module):

    def __init__(self, channels):
        super(UCGF_Module, self).__init__()
        self.spatial_attn = nn.Conv2d(2, 1, 7, padding=3)
        self.sigmoid = nn.Sigmoid()

        self.alpha = nn.Parameter(torch.tensor(-10.0))
        self.beta = nn.Parameter(torch.tensor(-10.0))

    def forward(self, x, log_var_map, epistemic_map=None):

        attenuation_a = torch.sigmoid(-log_var_map)  # [0, 1]，高噪声区域趋近于 0

        gate_a = 1 - torch.sigmoid(self.alpha) * (1 - attenuation_a)
        if epistemic_map is None:
            gate_e = 1.0
        else:
            gate_e = 1 - torch.sigmoid(self.beta) * epistemic_map
        x_weighted = x * gate_a * gate_e

        max_out, _ = torch.max(x_weighted, dim=1, keepdim=True)
        avg_out = torch.mean(x_weighted, dim=1, keepdim=True)
        spatial_w = self.sigmoid(self.spatial_attn(torch.cat([max_out, avg_out], dim=1)))

        return x_weighted * spatial_w + x  # 残差连接

class MultiRelationDiff(nn.Module):

    def __init__(self, feat_channels=768, num_heads=8):
        super(MultiRelationDiff, self).__init__()
        self.feat_channels = feat_channels
        
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=feat_channels, num_heads=num_heads,
            dropout=0.0, batch_first=True
        )
        self.attn_norm = nn.LayerNorm(feat_channels)
        
        self.fuse = nn.Sequential(
            nn.Conv2d(feat_channels * 3, feat_channels, 1, bias=False),
            nn.BatchNorm2d(feat_channels),
            nn.GELU(),
            nn.Conv2d(feat_channels, feat_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(feat_channels),
            nn.GELU(),
        )
    
    def forward(self, l4_A, l4_B):

        B, C, H, W = l4_A.shape
        
        diff_abs = torch.abs(l4_A - l4_B)              # [B, C, H, W]
        
        diff_mul = l4_A * l4_B                          # [B, C, H, W]
        

        a_flat = l4_A.flatten(2).permute(0, 2, 1)       # [B, HW, C]
        b_flat = l4_B.flatten(2).permute(0, 2, 1)       # [B, HW, C]

        attn_out, _ = self.cross_attn(a_flat, b_flat, b_flat)  # [B, HW, C]
        attn_out = self.attn_norm(attn_out)

        diff_cross = (attn_out - a_flat).permute(0, 2, 1).view(B, C, H, W)  # [B, C, H, W]

        feat_diff = self.fuse(torch.cat([diff_abs, diff_mul, diff_cross], dim=1))  # [B, C, H, W]
        
        return feat_diff



class SimpleRelationDiff(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(SimpleRelationDiff, self).__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(in_channels * 3, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, feat_A, feat_B):
        diff_abs = torch.abs(feat_A - feat_B)
        diff_mul = feat_A * feat_B
        diff_signed = feat_A - feat_B
        return self.fuse(torch.cat([diff_abs, diff_mul, diff_signed], dim=1))


class MultiScaleRelationDiff(nn.Module):

    def __init__(self, stage_channels=DEFAULT_OUT_CHANNELS, out_channels=768):
        super(MultiScaleRelationDiff, self).__init__()
        c1, c2, c3, c4 = stage_channels
        self.diff1 = SimpleRelationDiff(in_channels=c1, out_channels=c1)
        self.diff2 = SimpleRelationDiff(in_channels=c2, out_channels=c2)
        self.diff3 = SimpleRelationDiff(in_channels=c3, out_channels=c3)
        # l4 分辨率较低，保留原来的 MultiRelationDiff，以建模非线性跨位置差异。
        self.diff4 = MultiRelationDiff(feat_channels=c4, num_heads=8)

        self.proj1 = nn.Sequential(
            nn.Conv2d(c1, 192, 1, bias=False),
            nn.BatchNorm2d(192),
            nn.GELU(),
        )
        self.proj2 = nn.Sequential(
            nn.Conv2d(c2, 192, 1, bias=False),
            nn.BatchNorm2d(192),
            nn.GELU(),
        )
        self.proj3 = nn.Sequential(
            nn.Conv2d(c3, 192, 1, bias=False),
            nn.BatchNorm2d(192),
            nn.GELU(),
        )
        self.proj4 = nn.Sequential(
            nn.Conv2d(c4, 192, 1, bias=False),
            nn.BatchNorm2d(192),
            nn.GELU(),
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(192 * 4, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, l1_A, l1_B, l2_A, l2_B, l3_A, l3_B, l4_A, l4_B):
        target_size = l1_A.shape[2:]

        d1 = self.proj1(self.diff1(l1_A, l1_B))
        d2 = self.proj2(self.diff2(l2_A, l2_B))
        d3 = self.proj3(self.diff3(l3_A, l3_B))
        d4 = self.proj4(self.diff4(l4_A, l4_B))

        d2 = F.interpolate(d2, size=target_size, mode='bilinear', align_corners=False)
        d3 = F.interpolate(d3, size=target_size, mode='bilinear', align_corners=False)
        d4 = F.interpolate(d4, size=target_size, mode='bilinear', align_corners=False)

        feat_diff = self.fuse(torch.cat([d1, d2, d3, d4], dim=1))
        return feat_diff


class UAMCD(nn.Module):
    def __init__(self, latent_dim, num_classes, backbone='vmamba', pretrained_path=None):
        super(UAMCD, self).__init__()
        channel = 128
        self.backbone_name = backbone
        self.feature_viz_enabled = False
        self.feature_viz_dir = '/mnt/ssd/sdb/Datasets/SARCDvisu'
        self.feature_viz_max_calls = 0
        self._feature_viz_calls = 0
        self._feature_viz_names = None

        self.backbone = build_siamese_backbone(
            backbone=backbone,
            pretrained_path=pretrained_path,
            out_channels=DEFAULT_OUT_CHANNELS,
        )
        c1, c2, c3, c4 = self.backbone.out_channels

        self.conv1 = BasicConv2d(2 * c1, c1, 1)
        self.conv2 = BasicConv2d(2 * c2, c2, 1)
        self.conv3 = BasicConv2d(2 * c3, c3, 1)
        self.conv4 = BasicConv2d(2 * c4, c4, 1)

        self.conv_4 = BasicConv2d(c4, channel, 3, 1, 1)
        self.conv_3 = BasicConv2d(c3, channel, 3, 1, 1)
        self.conv_2 = BasicConv2d(c2, channel, 3, 1, 1)
        self.conv_1 = BasicConv2d(c1, channel, 3, 1, 1)

        self.multi_diff = MultiScaleRelationDiff(stage_channels=self.backbone.out_channels, out_channels=768)

        self.decoder = Refined_Decoder(num_classes)

    def set_feature_viz(self, enabled=False, out_dir=None, max_calls=10):
        self.feature_viz_enabled = bool(enabled)
        self.feature_viz_dir = out_dir
        self.feature_viz_max_calls = int(max_calls)
        self._feature_viz_calls = 0
        self._feature_viz_names = None
        if self.feature_viz_enabled and self.feature_viz_dir is not None:
            os.makedirs(self.feature_viz_dir, exist_ok=True)

    def set_feature_viz_names(self, names):
        self._feature_viz_names = names

    @staticmethod
    def _feat_to_uint8_map(feat):
        x = feat[0].detach().float().mean(dim=0)  # [H, W]
        x = x - x.min()
        denom = x.max() - x.min()
        if denom > 1e-8:
            x = x / denom
        x = (x * 255.0).clamp(0, 255).byte().cpu().numpy()
        return x

    def _save_intermediate_features(self, l1_A, l2_A, l3_A, l4_A, l1_B, l2_B, l3_B, l4_B, feat_diff):
        if (not self.feature_viz_enabled) or self.feature_viz_dir is None:
            return
        if self._feature_viz_names is None:
            return
        if isinstance(self._feature_viz_names, (list, tuple)):
            if len(self._feature_viz_names) == 0:
                return
            prefix = str(self._feature_viz_names[0])
        else:
            prefix = str(self._feature_viz_names)
        prefix = os.path.splitext(os.path.basename(prefix))[0]
        if prefix == "":
            return

        maps = {
            f"{prefix}_l1_A.png": self._feat_to_uint8_map(l1_A),
            f"{prefix}_l2_A.png": self._feat_to_uint8_map(l2_A),
            f"{prefix}_l3_A.png": self._feat_to_uint8_map(l3_A),
            f"{prefix}_l4_A.png": self._feat_to_uint8_map(l4_A),
            f"{prefix}_l1_B.png": self._feat_to_uint8_map(l1_B),
            f"{prefix}_l2_B.png": self._feat_to_uint8_map(l2_B),
            f"{prefix}_l3_B.png": self._feat_to_uint8_map(l3_B),
            f"{prefix}_l4_B.png": self._feat_to_uint8_map(l4_B),
            f"{prefix}_feat_diff.png": self._feat_to_uint8_map(feat_diff),
            f"{prefix}_l1_absdiff.png": self._feat_to_uint8_map(torch.abs(l1_A - l1_B)),
            f"{prefix}_l2_absdiff.png": self._feat_to_uint8_map(torch.abs(l2_A - l2_B)),
            f"{prefix}_l3_absdiff.png": self._feat_to_uint8_map(torch.abs(l3_A - l3_B)),
            f"{prefix}_l4_absdiff.png": self._feat_to_uint8_map(torch.abs(l4_A - l4_B)),
        }
        for name, arr in maps.items():
            Image.fromarray(arr).save(os.path.join(self.feature_viz_dir, name))
        self._feature_viz_calls += 1
        self._feature_viz_names = None

    def Feature_Extraction(self, A, B):
        l1_A, l2_A, l3_A, l4_A = self.backbone(A)
        l1_B, l2_B, l3_B, l4_B = self.backbone(B)

        feat_diff = self.multi_diff(l1_A, l1_B, l2_A, l2_B, l3_A, l3_B, l4_A, l4_B)  # [B, 768, H/4, W/4]
        self._save_intermediate_features(l1_A, l2_A, l3_A, l4_A, l1_B, l2_B, l3_B, l4_B, feat_diff)

        layer_1 = self.conv_1(self.conv1(torch.cat((l1_A, l1_B), dim=1)))
        layer_2 = self.conv_2(self.conv2(torch.cat((l2_A, l2_B), dim=1)))
        layer_3 = self.conv_3(self.conv3(torch.cat((l3_A, l3_B), dim=1)))
        layer_4 = self.conv_4(self.conv4(torch.cat((l4_A, l4_B), dim=1)))

        return layer_1, layer_2, layer_3, layer_4, feat_diff

    def extract_features(self, A, B):
        l1, l2, l3, l4, feat_diff = self.Feature_Extraction(A, B)
        return {
            'l1': l1,
            'l2': l2,
            'l3': l3,
            'l4': l4,
            'feat_diff': feat_diff,
        }

    def decode_with_uncertainty(self, features, epistemic_map=None):
        refined_pred, aleatoric_log_var = self.decoder(
            features['l4'], features['l3'], features['l2'], features['l1'],
            features['feat_diff'], epistemic_map
        )
        return refined_pred, aleatoric_log_var

    def forward(self, A, B, y=None, epistemic_map=None):
        features = self.extract_features(A, B)
        refined_pred, aleatoric_log_var = self.decode_with_uncertainty(
            features, epistemic_map
        )
        return refined_pred, features['feat_diff'], aleatoric_log_var

class Refined_Decoder(nn.Module):
    def __init__(self, num_classes, feat_channels=768):
        super(Refined_Decoder, self).__init__()
        channel = 128
        self.down8 = nn.Upsample(scale_factor=0.125, mode='bilinear')
        self.down4 = nn.Upsample(scale_factor=0.25, mode='bilinear')
        self.down2 = nn.Upsample(scale_factor=0.5, mode='bilinear')
        
        self.ale4 = nn.Sequential(
            nn.Conv2d(feat_channels, channel, 3, 1, 1),
            nn.GroupNorm(8, channel),
            nn.LeakyReLU(0.2),
        )
        self.ale3 = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 1, 1),
            nn.GroupNorm(8, channel),
            nn.LeakyReLU(0.2),
        )
        self.ale2 = nn.Sequential(
            nn.Conv2d(channel, channel // 2, 3, 1, 1),
            nn.GroupNorm(4, channel // 2),
            nn.LeakyReLU(0.2),
        )
        self.ale1 = nn.Sequential(
            nn.Conv2d(channel // 2, channel // 2, 3, 1, 1),
            nn.GroupNorm(4, channel // 2),
            nn.LeakyReLU(0.2),
        )
        self.skip3 = nn.Conv2d(channel, channel, 1)
        self.skip2 = nn.Conv2d(channel, channel, 1)
        self.skip1 = nn.Conv2d(channel, channel // 2, 1)
        self.aleatoric_out = nn.Conv2d(channel // 2, 1, 1)

        self.UCGF4 = UCGF_Module(channel)
        self.UCGF3 = UCGF_Module(channel)
        self.UCGF2 = UCGF_Module(channel)
        self.UCGF1 = UCGF_Module(channel)

        self.Fusion4 = AggUnit(channel)
        self.Fusion3 = AggUnit(channel)
        self.Fusion2 = AggUnit(channel)
        self.Fusion1 = AggUnit(channel)

        self.out_conv = nn.Sequential(
            nn.Conv2d(channel, 64, 3, 1, 1),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(32, num_classes, 1)
        )

    def forward(self, l4, l3, l2, l1, feat_diff, epistemic_map=None):

        feat_diff_l2 = F.interpolate(feat_diff, size=l2.shape[2:], mode='bilinear', align_corners=True)
        a2_seed = self.ale4(feat_diff_l2)  # ~1/8, [B, 128, H_l2, W_l2]
        a3 = F.interpolate(a2_seed, size=l3.shape[2:], mode='bilinear', align_corners=True) + self.skip3(l3)  # ~1/16
        a3 = self.ale3(a3)
        a2 = F.interpolate(a3, size=l2.shape[2:], mode='bilinear', align_corners=True) + self.skip2(l2) + a2_seed  # ~1/8
        a2 = self.ale2(a2)
        a1 = F.interpolate(a2, size=l1.shape[2:], mode='bilinear', align_corners=True) + self.skip1(l1)  # ~1/4
        a1 = self.ale1(a1)
        aleatoric_map = self.aleatoric_out(a1)  # [B, 1, H_l1, W_l1]

        unc_4 = F.interpolate(aleatoric_map, size=l4.shape[2:], mode='bilinear', align_corners=True)
        unc_3 = F.interpolate(aleatoric_map, size=l3.shape[2:], mode='bilinear', align_corners=True)
        unc_2 = F.interpolate(aleatoric_map, size=l2.shape[2:], mode='bilinear', align_corners=True)
        unc_1 = F.interpolate(aleatoric_map, size=l1.shape[2:], mode='bilinear', align_corners=True)

        epi_4 = epi_3 = epi_2 = epi_1 = None
        if epistemic_map is not None:
            epi_4 = F.interpolate(epistemic_map, size=l4.shape[2:], mode='bilinear', align_corners=True)
            epi_3 = F.interpolate(epistemic_map, size=l3.shape[2:], mode='bilinear', align_corners=True)
            epi_2 = F.interpolate(epistemic_map, size=l2.shape[2:], mode='bilinear', align_corners=True)
            epi_1 = F.interpolate(epistemic_map, size=l1.shape[2:], mode='bilinear', align_corners=True)

        l4 = self.UCGF4(l4, unc_4, epi_4)
        l3 = self.UCGF3(l3, unc_3, epi_3)
        l2 = self.UCGF2(l2, unc_2, epi_2)
        l1 = self.UCGF1(l1, unc_1, epi_1)

        Fusion = self.Fusion4(l4)
        Fusion = self.Fusion3(Fusion, l3)
        Fusion = self.Fusion2(Fusion, l2)
        Fusion = self.Fusion1(Fusion, l1)
        return self.out_conv(Fusion), aleatoric_map

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x): return self.relu(self.bn(self.conv(x)))

class ResidualConvUnit(nn.Module):
    def __init__(self, features):
        super(ResidualConvUnit, self).__init__()
        self.c1 = nn.Conv2d(features, features, 3, 1, 1)
        self.c2 = nn.Conv2d(features, features, 3, 1, 1)
        self.relu = nn.ReLU(True)
    def forward(self, x): return self.c2(self.relu(self.c1(self.relu(x)))) + x

class AggUnit(nn.Module):
    def __init__(self, features):
        super(AggUnit, self).__init__()
        self.r1 = ResidualConvUnit(features)
        self.r2 = ResidualConvUnit(features)
    def forward(self, *xs):
        out = xs[0]
        if len(xs) == 2: out += self.r1(xs[1])
        out = self.r2(out)
        return F.interpolate(out, scale_factor=2, mode="bilinear", align_corners=True)
