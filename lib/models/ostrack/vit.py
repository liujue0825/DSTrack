""" Vision Transformer (ViT) in PyTorch
A PyTorch implement of Vision Transformers as described in:
'An Image Is Worth 16 x 16 Words: Transformers for Image Recognition at Scale'
    - https://arxiv.org/abs/2010.11929
`How to train your ViT? Data, Augmentation, and Regularization in Vision Transformers`
    - https://arxiv.org/abs/2106.10270
The official jax code is released and available at https://github.com/google-research/vision_transformer
DeiT model defs and weights from https://github.com/facebookresearch/deit,
paper `DeiT: Data-efficient Image Transformers` - https://arxiv.org/abs/2012.12877
Acknowledgments:
* The paper authors for releasing code and weights, thanks!
* I fixed my class token impl based on Phil Wang's https://github.com/lucidrains/vit-pytorch ... check it out
for some einops/einsum fun
* Simple transformer style inspired by Andrej Karpathy's https://github.com/karpathy/minGPT
* Bert reference code checks against Huggingface Transformers and Tensorflow Bert
Hacked together by / Copyright 2021 Ross Wightman

Modified by Botao Ye
"""
from functools import partial

import math
from re import I
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.helpers import named_apply, adapt_input_conv
from timm.models.layers import Mlp, DropPath, trunc_normal_, lecun_normal_

from lib.models.layers.patch_embed import PatchEmbed
from lib.models.ostrack.base_backbone import BaseBackbone
from lib.models.ostrack.utils import combine_tokens, recover_tokens, token2feature, feature2token


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, return_attention=False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        if return_attention:
            return x, attn
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention=False):
        if return_attention:
            feat, attn = self.attn(self.norm1(x), True)
            x = x + self.drop_path(feat)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x, attn
        else:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x


class GradReverse(torch.autograd.Function):
    """梯度反转层"""

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


class AlphaScheduler:
    """alpha 值调度器"""

    def __init__(self, mode='linear', initial_alpha=1.0, decay_epoch=10, decay_rate=0.1):
        self.mode = mode
        self.initial_alpha = initial_alpha
        self.decay_epoch = decay_epoch
        self.decay_rate = decay_rate

    def get_alpha(self, epoch):
        if self.mode == 'step':
            return self._step_schedule(epoch)
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

    def _step_schedule(self, epoch):
        if epoch < self.decay_epoch:
            return self.initial_alpha
        else:
            return self.initial_alpha * self.decay_rate


class AdaptiveGradReverse(nn.Module):
    def __init__(self, alpha_scheduler):
        super().__init__()
        self.alpha_scheduler = alpha_scheduler
        self.current_alpha = 0.0

    def forward(self, x):
        return GradReverse.apply(x, self.current_alpha)

    def update_alpha(self, epoch):
        """
        实现方案: model.grl.update_alpha(epoch)
        """
        self.current_alpha = self.alpha_scheduler.get_alpha(epoch)


def grad_reverse(x, alpha=1.0):
    """
    在反向传播时，将梯度乘以 -1 或乘以 alpha, 实现梯度的反转
    Args:
        x:
        alpha: 梯度反转系数, 默认值为 1.0, 如果越大, 更侧重与辅助任务, 主任务收敛变慢; 如果越小, 更侧重与主任务, 辅助任务收敛变慢

    Returns:

    """
    return GradReverse.apply(x, alpha)


class ModalityClassifier(nn.Module):
    """域判别器"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, BN=False):
        """

        Args:
            input_dim: 768
            hidden_dim: 768 // 4
            output_dim: 1
            num_layers: 2 / 3
            BN: False
        """
        super().__init__()
        self.num_layers = num_layers
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        h = [hidden_dim] * (num_layers - 1)
        if BN:
            self.layers = nn.ModuleList(nn.Sequential(nn.Linear(n, k), nn.BatchNorm1d(k))
                                        for n, k in zip([input_dim] + h, h + [output_dim]))
        else:
            self.layers = nn.ModuleList(nn.Linear(n, k)
                                        for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        """
        利用 GRL 辅助模态判别
        Args:
            x: (B, C, H, W)

        Returns:
            out: (B, output_dim)
        """
        x = self.gap(x).view(x.size(0), -1)
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class AdapterBlock(nn.Module):
    """模态特定信息提取器"""

    def __init__(self, in_channels=None, expand_ratio=2):
        super().__init__()
        hidden_dim = int(in_channels * expand_ratio)
        self.bottleneckBlock = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 1, bias=False),  # pw
            # nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            nn.ReflectionPad2d(1),  # dw
            nn.Conv2d(hidden_dim, hidden_dim, 3, groups=hidden_dim, bias=False),
            # nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_dim, in_channels, 1, bias=False),  # pw-linear
            # nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        return self.bottleneckBlock(x)


class VisionTransformer(BaseBackbone):
    """ Vision Transformer
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    Includes distillation token & head support for `DeiT: Data-efficient Image Transformers`
        - https://arxiv.org/abs/2012.12877
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init='', grl_index=6):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            weight_init: (str): weight init scheme
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # NOTE: domain discriminator
        self.grl_index = grl_index
        self.classifier = ModalityClassifier(input_dim=embed_dim, hidden_dim=embed_dim // 4, output_dim=1, num_layers=3)
        self.grl = None

        # NOTE: adapter blocks
        adapter_nums = depth - self.grl_index
        self.rgb_adapter = nn.Sequential(*[AdapterBlock(in_channels=embed_dim) for _ in range(adapter_nums)])
        self.nir_adapter = nn.Sequential(*[AdapterBlock(in_channels=embed_dim) for _ in range(adapter_nums)])

        self.init_weights(weight_init)
        self.feat_rgb = []
        self.feat_nir = []
    
    def clear_feat(self):
        # 清空feat集合
        self.feat_rgb = []
        self.feat_nir = []

    def set_grad_reverse(self, grl):
        """
        在初始化 OSTrack 时设置
        """
        self.grl = grl

    def forward_features(self, z, x, template_label=None, search_label=None, infer=True):
        """
        重写 backbone 的 forward 函数
        Args:
            z: (B, 3, H, W)
            x: (B, 3, H, W)
            template_label: (B, 1)
            search_label: (B, 1)
            infer: 训练(False) 或 推理(True)
        Returns:

        """
        B, H, W = x.shape[0], x.shape[2], x.shape[3]

        x = self.patch_embed(x)
        z = self.patch_embed(z)

        if self.add_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            cls_tokens = cls_tokens + self.cls_pos_embed

        z += self.pos_embed_z
        x += self.pos_embed_x

        if self.add_sep_seg:
            x += self.search_segment_pos_embed
            z += self.template_segment_pos_embed

        x = combine_tokens(z, x, mode=self.cat_mode)
        if self.add_cls_token:
            x = torch.cat([cls_tokens, x], dim=1)

        x = self.pos_drop(x)

        lens_z = self.pos_embed_z.shape[1]
        lens_x = self.pos_embed_x.shape[1]

        for i, blk in enumerate(self.blocks):
            if i <= self.grl_index - 1:
                x = blk(x)
                if i == self.grl_index - 1 and self.grl:
                    x_cls = x.clone()
                    feat_cls_z = token2feature(x_cls[:, :lens_z, :])
                    feat_cls_x = token2feature(x_cls[:, lens_z:, :])
                    feat_cls_z = self.grl(feat_cls_z)
                    feat_cls_x = self.grl(feat_cls_x)
                    weight_raw_z = self.classifier(feat_cls_z)
                    weight_raw_x = self.classifier(feat_cls_x)
                    if infer:
                        with open("/DATA/liujue/resource/predict.txt", 'a') as f:
                            f.write(f"{weight_raw_x.sigmoid().view(-1).item()}\n")
                    else:
                        # NOTE: 保存特征
                        mask_nir_x = search_label.bool().view(-1)
                        mask_rgb_x = ~mask_nir_x
                        self.feat_rgb.append(feat_cls_x[mask_rgb_x])
                        self.feat_nir.append(feat_cls_x[mask_nir_x])
            else:
                # 梯度截断
                if i == self.grl_index and self.grl:
                    x = x.detach()
                x = blk(x)
                x_encoder = x
                z_feat = token2feature(x[:, :lens_z, :])
                x_feat = token2feature(x[:, lens_z:, :])  # B, 256, 768 -> B, 768, 16, 16
                z_enhance_feat = torch.zeros_like(z_feat)
                x_enhance_feat = torch.zeros_like(x_feat)
                # 训练阶段: 用模态标签监督两个分支的训练
                if not infer:
                    # 根据模态标签决定数据流向
                    mask_nir_z = template_label.bool().view(-1)
                    mask_rgb_z = ~mask_nir_z
                    mask_nir_x = search_label.bool().view(-1)
                    mask_rgb_x = ~mask_nir_x
                # 推理阶段: 用模态权重 weight 代替模态标签, 使得输入能正确流向对应的模态分支
                else:
                    weight_z = weight_raw_z.sigmoid().view(-1)
                    weight_x = weight_raw_x.sigmoid().view(-1)
                    # 根据 weight 大小决定数据流向
                    mask_rgb_z = weight_z <= 0.5
                    mask_nir_z = weight_z > 0.5
                    mask_rgb_x = weight_x <= 0.5
                    mask_nir_x = weight_x > 0.5
                # 只为需要的样本计算相应的分支
                if mask_rgb_z.any():
                    rgb_branch_z = self.rgb_adapter[i - self.grl_index](z_feat[mask_rgb_z])
                    z_enhance_feat[mask_rgb_z] = rgb_branch_z
                if mask_nir_z.any():
                    nir_branch_z = self.nir_adapter[i - self.grl_index](z_feat[mask_nir_z])
                    z_enhance_feat[mask_nir_z] = nir_branch_z
                if mask_rgb_x.any():
                    rgb_branch_x = self.rgb_adapter[i - self.grl_index](x_feat[mask_rgb_x])
                    x_enhance_feat[mask_rgb_x] = rgb_branch_x
                if mask_nir_x.any():
                    nir_branch_x = self.nir_adapter[i - self.grl_index](x_feat[mask_nir_x])
                    x_enhance_feat[mask_nir_x] = nir_branch_x
                z_enhance_tokens = feature2token(z_enhance_feat)
                x_enhance_tokens = feature2token(x_enhance_feat)
                x_enhance = combine_tokens(z_enhance_tokens, x_enhance_tokens, mode=self.cat_mode)
                x = x_enhance + x_encoder

        x = recover_tokens(x, lens_z, lens_x, mode=self.cat_mode)

        aux_dict = {"attn": None, "weight_z": weight_raw_z, "weight_x": weight_raw_x}
        return self.norm(x), aux_dict

    def forward(self, z, x, template_label=None, search_label=None, infer=True):
        """
        Joint feature extraction and relation modeling for the basic ViT backbone.
        Args:
            z (torch.Tensor): template feature, [B, C, H_z, W_z]
            x (torch.Tensor): search region feature, [B, C, H_x, W_x]
            template_label (torch.Tensor): modality label, [B, 1]
            search_label (torch.Tensor): modality label, [B, 1]
            infer (bool): inference or train mode
        Returns:
           x (torch.Tensor): merged template and search region feature, [B, L_z+L_x, C]
           attn : None
        """
        x, aux_dict = self.forward_features(z, x, template_label, search_label, infer)
        return x, aux_dict

    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'nlhb', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        trunc_normal_(self.pos_embed, std=.02)
        if self.dist_token is not None:
            trunc_normal_(self.dist_token, std=.02)
        if mode.startswith('jax'):
            # leave cls token as zeros to match jax impl
            named_apply(partial(_init_vit_weights, head_bias=head_bias, jax_impl=True), self)
        else:
            trunc_normal_(self.cls_token, std=.02)
            self.apply(_init_vit_weights)

    def _init_weights(self, m):
        # this fn left here for compat with downstream users
        _init_vit_weights(m)

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=''):
        _load_weights(self, checkpoint_path, prefix)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}

    def get_classifier(self):
        if self.dist_token is None:
            return self.head
        else:
            return self.head, self.head_dist

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        if self.num_tokens == 2:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()


def _init_vit_weights(module: nn.Module, name: str = '', head_bias: float = 0., jax_impl: bool = False):
    """ ViT weight initialization
    * When called without n, head_bias, jax_impl args it will behave exactly the same
      as my original init for compatibility with prev hparam / downstream use cases (ie DeiT).
    * When called w/ valid n (module name) and jax_impl=True, will (hopefully) match JAX impl
    """
    if isinstance(module, nn.Linear):
        if name.startswith('head'):
            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, head_bias)
        elif name.startswith('pre_logits'):
            lecun_normal_(module.weight)
            nn.init.zeros_(module.bias)
        else:
            if jax_impl:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    if 'mlp' in name:
                        nn.init.normal_(module.bias, std=1e-6)
                    else:
                        nn.init.zeros_(module.bias)
            else:
                trunc_normal_(module.weight, std=.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    elif jax_impl and isinstance(module, nn.Conv2d):
        # NOTE conv was left to pytorch default in my original init
        lecun_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)


@torch.no_grad()
def _load_weights(model: VisionTransformer, checkpoint_path: str, prefix: str = ''):
    """ Load weights from .npz checkpoints for official Google Brain Flax implementation
    """
    import numpy as np

    def _n2p(w, t=True):
        if w.ndim == 4 and w.shape[0] == w.shape[1] == w.shape[2] == 1:
            w = w.flatten()
        if t:
            if w.ndim == 4:
                w = w.transpose([3, 2, 0, 1])
            elif w.ndim == 3:
                w = w.transpose([2, 0, 1])
            elif w.ndim == 2:
                w = w.transpose([1, 0])
        return torch.from_numpy(w)

    w = np.load(checkpoint_path)
    if not prefix and 'opt/target/embedding/kernel' in w:
        prefix = 'opt/target/'

    if hasattr(model.patch_embed, 'backbone'):
        # hybrid
        backbone = model.patch_embed.backbone
        stem_only = not hasattr(backbone, 'stem')
        stem = backbone if stem_only else backbone.stem
        stem.conv.weight.copy_(adapt_input_conv(stem.conv.weight.shape[1], _n2p(w[f'{prefix}conv_root/kernel'])))
        stem.norm.weight.copy_(_n2p(w[f'{prefix}gn_root/scale']))
        stem.norm.bias.copy_(_n2p(w[f'{prefix}gn_root/bias']))
        if not stem_only:
            for i, stage in enumerate(backbone.stages):
                for j, block in enumerate(stage.blocks):
                    bp = f'{prefix}block{i + 1}/unit{j + 1}/'
                    for r in range(3):
                        getattr(block, f'conv{r + 1}').weight.copy_(_n2p(w[f'{bp}conv{r + 1}/kernel']))
                        getattr(block, f'norm{r + 1}').weight.copy_(_n2p(w[f'{bp}gn{r + 1}/scale']))
                        getattr(block, f'norm{r + 1}').bias.copy_(_n2p(w[f'{bp}gn{r + 1}/bias']))
                    if block.downsample is not None:
                        block.downsample.conv.weight.copy_(_n2p(w[f'{bp}conv_proj/kernel']))
                        block.downsample.norm.weight.copy_(_n2p(w[f'{bp}gn_proj/scale']))
                        block.downsample.norm.bias.copy_(_n2p(w[f'{bp}gn_proj/bias']))
        embed_conv_w = _n2p(w[f'{prefix}embedding/kernel'])
    else:
        embed_conv_w = adapt_input_conv(
            model.patch_embed.proj.weight.shape[1], _n2p(w[f'{prefix}embedding/kernel']))
    model.patch_embed.proj.weight.copy_(embed_conv_w)
    model.patch_embed.proj.bias.copy_(_n2p(w[f'{prefix}embedding/bias']))
    model.cls_token.copy_(_n2p(w[f'{prefix}cls'], t=False))
    pos_embed_w = _n2p(w[f'{prefix}Transformer/posembed_input/pos_embedding'], t=False)
    if pos_embed_w.shape != model.pos_embed.shape:
        pos_embed_w = resize_pos_embed(  # resize pos embedding when different size from pretrained weights
            pos_embed_w, model.pos_embed, getattr(model, 'num_tokens', 1), model.patch_embed.grid_size)
    model.pos_embed.copy_(pos_embed_w)
    model.norm.weight.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/scale']))
    model.norm.bias.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/bias']))
    if isinstance(model.head, nn.Linear) and model.head.bias.shape[0] == w[f'{prefix}head/bias'].shape[-1]:
        model.head.weight.copy_(_n2p(w[f'{prefix}head/kernel']))
        model.head.bias.copy_(_n2p(w[f'{prefix}head/bias']))
    if isinstance(getattr(model.pre_logits, 'fc', None), nn.Linear) and f'{prefix}pre_logits/bias' in w:
        model.pre_logits.fc.weight.copy_(_n2p(w[f'{prefix}pre_logits/kernel']))
        model.pre_logits.fc.bias.copy_(_n2p(w[f'{prefix}pre_logits/bias']))
    for i, block in enumerate(model.blocks.children()):
        block_prefix = f'{prefix}Transformer/encoderblock_{i}/'
        mha_prefix = block_prefix + 'MultiHeadDotProductAttention_1/'
        block.norm1.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/scale']))
        block.norm1.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/bias']))
        block.attn.qkv.weight.copy_(torch.cat([
            _n2p(w[f'{mha_prefix}{n}/kernel'], t=False).flatten(1).T for n in ('query', 'key', 'value')]))
        block.attn.qkv.bias.copy_(torch.cat([
            _n2p(w[f'{mha_prefix}{n}/bias'], t=False).reshape(-1) for n in ('query', 'key', 'value')]))
        block.attn.proj.weight.copy_(_n2p(w[f'{mha_prefix}out/kernel']).flatten(1))
        block.attn.proj.bias.copy_(_n2p(w[f'{mha_prefix}out/bias']))
        for r in range(2):
            getattr(block.mlp, f'fc{r + 1}').weight.copy_(_n2p(w[f'{block_prefix}MlpBlock_3/Dense_{r}/kernel']))
            getattr(block.mlp, f'fc{r + 1}').bias.copy_(_n2p(w[f'{block_prefix}MlpBlock_3/Dense_{r}/bias']))
        block.norm2.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_2/scale']))
        block.norm2.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_2/bias']))


def resize_pos_embed(posemb, posemb_new, num_tokens=1, gs_new=()):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    print('Resized position embedding: %s to %s', posemb.shape, posemb_new.shape)
    ntok_new = posemb_new.shape[1]
    if num_tokens:
        posemb_tok, posemb_grid = posemb[:, :num_tokens], posemb[0, num_tokens:]
        ntok_new -= num_tokens
    else:
        posemb_tok, posemb_grid = posemb[:, :0], posemb[0]
    gs_old = int(math.sqrt(len(posemb_grid)))
    if not len(gs_new):  # backwards compatibility
        gs_new = [int(math.sqrt(ntok_new))] * 2
    assert len(gs_new) >= 2
    print('Position embedding grid-size from %s to %s', [gs_old, gs_old], gs_new)
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=gs_new, mode='bilinear')
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new[0] * gs_new[1], -1)
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    return posemb


def checkpoint_filter_fn(state_dict, model):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    if 'model' in state_dict:
        # For deit models
        state_dict = state_dict['model']
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k and len(v.shape) < 4:
            # For old models that I trained prior to conv based patchification
            O, I, H, W = model.patch_embed.proj.weight.shape
            v = v.reshape(O, -1, H, W)
        elif k == 'pos_embed' and v.shape != model.pos_embed.shape:
            # To resize pos embedding when using model at different size from pretrained weights
            v = resize_pos_embed(
                v, model.pos_embed, getattr(model, 'num_tokens', 1), model.patch_embed.grid_size)
        out_dict[k] = v
    return out_dict


def _create_vision_transformer(variant, pretrained=False, default_cfg=None, **kwargs):
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')

    model = VisionTransformer(**kwargs)

    if pretrained:
        if 'npz' in pretrained:
            model.load_pretrained(pretrained, prefix='')
        else:
            checkpoint = torch.load(pretrained, map_location="cpu")
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint["model"], strict=False)
            print('Load pretrained model from: ' + pretrained)

    return model


def vit_base_patch16_224(pretrained=False, **kwargs):
    """
    ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer('vit_base_patch16_224_in21k', pretrained=pretrained, **model_kwargs)
    return model
