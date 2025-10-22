"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import copy
from collections import OrderedDict

import torch 
import torch.nn as nn 
import torch.nn.functional as F 

from .utils import get_activation

from ...core import register


__all__ = ['HybridEncoder']

class ConvNormLayer(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, padding=None, bias=False, act=None):
        super().__init__()
        self.conv = nn.Conv2d(
            ch_in, 
            ch_out, 
            kernel_size, 
            stride, 
            padding=(kernel_size-1)//2 if padding is None else padding, 
            bias=bias)
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act) 

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class RepVggBlock(nn.Module):
    def __init__(self, ch_in, ch_out, act='relu'):
        super().__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.conv1 = ConvNormLayer(ch_in, ch_out, 3, 1, padding=1, act=None)
        self.conv2 = ConvNormLayer(ch_in, ch_out, 1, 1, padding=0, act=None)
        self.act = nn.Identity() if act is None else get_activation(act) 

    def forward(self, x):
        if hasattr(self, 'conv'):
            y = self.conv(x)
        else:
            y = self.conv1(x) + self.conv2(x)

        return self.act(y)

    def convert_to_deploy(self):
        if not hasattr(self, 'conv'):
            self.conv = nn.Conv2d(self.ch_in, self.ch_out, 3, 1, padding=1)

        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv.weight.data = kernel
        self.conv.bias.data = bias 

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1), bias3x3 + bias1x1

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return F.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch: ConvNormLayer):
        if branch is None:
            return 0, 0
        kernel = branch.conv.weight
        running_mean = branch.norm.running_mean
        running_var = branch.norm.running_var
        gamma = branch.norm.weight
        beta = branch.norm.bias
        eps = branch.norm.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std


class CSPRepLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_blocks=3,
                 expansion=1.0,
                 bias=None,
                 act="silu"):
        super(CSPRepLayer, self).__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvNormLayer(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.conv2 = ConvNormLayer(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.bottlenecks = nn.Sequential(*[
            RepVggBlock(hidden_channels, hidden_channels, act=act) for _ in range(num_blocks)
        ])
        if hidden_channels != out_channels:
            self.conv3 = ConvNormLayer(hidden_channels, out_channels, 1, 1, bias=bias, act=act)
        else:
            self.conv3 = nn.Identity()

    def forward(self, x):
        x_1 = self.conv1(x)
        x_1 = self.bottlenecks(x_1)
        x_2 = self.conv2(x)
        return self.conv3(x_1 + x_2)


# transformer
class TransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation="relu",
                 normalize_before=False):
        super().__init__()
        self.normalize_before = normalize_before

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout, batch_first=True)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = get_activation(activation) 

    @staticmethod
    def with_pos_embed(tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, src, src_mask=None, pos_embed=None) -> torch.Tensor:
        residual = src
        if self.normalize_before:
            src = self.norm1(src)
        q = k = self.with_pos_embed(src, pos_embed)
        src, _ = self.self_attn(q, k, value=src, attn_mask=src_mask)

        src = residual + self.dropout1(src)
        if not self.normalize_before:
            src = self.norm1(src)

        residual = src
        if self.normalize_before:
            src = self.norm2(src)
        src = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = residual + self.dropout2(src)
        if not self.normalize_before:
            src = self.norm2(src)
        return src

class VimBlock(nn.Module):
    """
    Algo.1 기반의 B_bar-directional SSM 블록 (참조 구현)
    입력/출력: (B, M, D)
    파라미터:
      - D: 입력/출력 임베딩 차원
      - E: 내부 확장 차원 (보통 D 또는 2D)
      - N: SSM 상태 차원 (논문 기본 16)
      - k: local mixing Conv1d kernel size (보통 3)
    """
    def __init__(self, D: int, E: int, N: int = 16, k: int = 3, activation="silu"):
        super().__init__()
        self.D, self.E, self.N = D, E, N

        # 1) Pre-norm & projection (D -> E)
        self.norm = nn.LayerNorm(D)
        self.lin_x = nn.Linear(D, E, bias=True)  # value-like
        self.lin_z = nn.Linear(D, E, bias=True)  # gating vector

        # 2) Local mixing (per direction)
        padding = k // 2
        self.conv_fwd = nn.Conv1d(E, E, kernel_size=k, padding=padding, bias=True)
        self.conv_bwd = nn.Conv1d(E, E, kernel_size=k, padding=padding, bias=True)

        # 3) SSM parameter generators (per direction)
        # B, C, Δ는 x'에서 생성 (data-dependent)
        #   B: (E -> N), C: (E -> N), Δ: (E -> E)  ← 가벼운/무거운 변형 중 여기선 E->E 채택
        self.B_fwd = nn.Linear(E, N, bias=True)
        self.C_fwd = nn.Linear(E, N, bias=True)
        self.Delta_fwd = nn.Linear(E, E, bias=True)

        self.B_bwd = nn.Linear(E, N, bias=True)
        self.C_bwd = nn.Linear(E, N, bias=True)
        self.Delta_bwd = nn.Linear(E, E, bias=True)

        # 학습 가능한 기준 A, Δ 오프셋(방향별)
        # A_base: (E, N) — Δ로 스케일되어 A_i = Δ_i[..., None] * A_base
        self.A_base_fwd = nn.Parameter(torch.randn(E, N) * 0.01)
        self.A_base_bwd = nn.Parameter(torch.randn(E, N) * 0.01)
        self.Delta_bias_fwd = nn.Parameter(torch.zeros(E))
        self.Delta_bias_bwd = nn.Parameter(torch.zeros(E))

        # 4) Output projection (E -> D) + residual
        self.lin_T = nn.Linear(E, D, bias=True)

        self.activation = get_activation(activation)

    @staticmethod
    def with_pos_embed(tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed

    @staticmethod
    def _scan_ssm(xp,  # (B, M, E)
                  A_base, B_lin, C_lin, Delta_lin, Delta_bias,
                  direction: str = "forward"):
        """
        한 방향 SSM 스캔.
        xp: local mixing 및 SiLU 후의 x'  (B, M, E)
        A_base: (E, N) learnable base
        B_lin, C_lin, Delta_lin: nn.Linear(E->N/E)
        Delta_bias: (E,)
        반환: y ∈ (B, M, E)
        """
        B, M, E = xp.shape
        N = A_base.shape[1]
        # data-dependent parameters
        B_param = B_lin(xp)            # (B, M, N)
        C_param = C_lin(xp)            # (B, M, N)
        Delta = Delta_lin(xp) + Delta_bias  # (B, M, E)
        Delta = F.softplus(Delta)      # Δ >= 0

        # 시퀀스 방향 설정 (backward일 땐 뒤집었다가 결과 되돌림)
        if direction == "backward":
            xp = torch.flip(xp, dims=[1])
            B_param = torch.flip(B_param, dims=[1])
            C_param = torch.flip(C_param, dims=[1])
            Delta = torch.flip(Delta, dims=[1])

        # 상태 초기화: h ∈ (B, E, N), y 모음: (B, M, E)
        h = xp.new_zeros(B, E, N)
        ys = []

        # 토큰 순차 갱신
        for i in range(M):
            # A_i, B_i 만들기 (Δ로 스케일)
            # A_i: (B, E, N), B_i: (B, E, N)
            Ai = Delta[:, i, :].unsqueeze(-1) * A_base  # (B,E,1)*(E,N) 브로드캐스트 허용을 위해 expand
            Ai = Ai.expand(B, E, N)  # (B,E,N)
            Bi = Delta[:, i, :].unsqueeze(-1) * B_param[:, i, :].unsqueeze(1)  # (B,E,1)*(B,1,N) -> (B,E,N)

            # 상태 갱신: h = A_i ⊙ h + B_i ⊙ x'_i[:, :, None]
            xei = xp[:, i, :].unsqueeze(-1)  # (B,E,1)
            h = Ai * h + Bi * xei

            # 출력: y_i = (h ⊗ C_i)  —— N 축 수축
            Ci = C_param[:, i, :].unsqueeze(1)  # (B,1,N)
            yi = (h * Ci).sum(dim=-1)           # (B,E)
            ys.append(yi)

        y = torch.stack(ys, dim=1)  # (B,M,E)

        # backward면 다시 원래 순서로 뒤집기
        if direction == "backward":
            y = torch.flip(y, dims=[1])

        return y  # (B, M, E)

    def forward(self, t, src_mask=None, pos_embed=None):
        """
        x: (B, M, D)
        return: (B, M, D)
        """
        B, M, D = t.shape
        assert D == self.D
        t = self.with_pos_embed(t, pos_embed)
        residual = t

        # 1) Pre-norm & project
        x_n = self.norm(t)                 # (B,M,D)
        x_proj = self.lin_x(x_n)           # (B,M,E)
        z = self.lin_z(x_n)                # (B,M,E) — gating용

        # 2) Local mixing (Conv1d는 (B,E,M) 입력)
        xp_f = self.activation(self.conv_fwd(x_proj.transpose(1, 2))).transpose(1, 2)  # (B,M,E)
        xp_b = self.activation(self.conv_bwd(x_proj.transpose(1, 2))).transpose(1, 2)  # (B,M,E)

        # 3) Bidirectional SSM scan
        y_f = self._scan_ssm(
            xp_f, self.A_base_fwd, self.B_fwd, self.C_fwd, self.Delta_fwd, self.Delta_bias_fwd, "forward"
        )  # (B,M,E)
        y_b = self._scan_ssm(
            xp_b, self.A_base_bwd, self.B_bwd, self.C_bwd, self.Delta_bwd, self.Delta_bias_bwd, "backward"
        )  # (B,M,E)

        # 4) Gating & merge (+ Linear_T, residual)
        g = self.activation(z)                  # (B,M,E)
        y = (y_f + y_b) * g                # (B,M,E)
        out = self.lin_T(y) + residual     # (B,M,D) + residual
        return out


class Encoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, src_mask=None, pos_embed=None) -> torch.Tensor:
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=src_mask, pos_embed=pos_embed)

        if self.norm is not None:
            output = self.norm(output)

        return output

class SubPixelUpsample(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2, act='silu'):
        super(SubPixelUpsample, self).__init__()
        r = scale_factor
        
        # 1. Convolution 단계: 채널을 r^2 배로 확장 (여기서 r=2, 즉 4배)
        # out_channels이 SubPixelShuffle의 출력 채널이므로, Conv의 출력 채널은 out_channels * r^2
        self.conv = nn.Conv2d(
            in_channels, 
            out_channels * (r ** 2), 
            kernel_size=3, 
            padding=1, 
            bias=False
        )
        self.norm = nn.BatchNorm2d(out_channels * (r ** 2))
        
        # 2. Pixel Shuffle 단계: 채널을 공간 해상도로 재배치
        self.pixel_shuffle = nn.PixelShuffle(r)
        
        self.final_norm = nn.BatchNorm2d(out_channels) # PixelShuffle 후 채널 수
        self.act = get_activation(act)

    def forward(self, x):
        # Conv -> Norm -> Act (선택적)
        x = self.conv(x)
        x = self.act(self.norm(x)) # 채널 확장된 상태에서 Norm/Act 적용
        
        # Pixel Shuffle (채널 축소, 공간 확장)
        x = self.pixel_shuffle(x)
        
        # 최종 정규화
        return self.act(self.final_norm(x))

@register()
class HybridEncoder(nn.Module):
    __share__ = ['eval_spatial_size', ]

    def __init__(self,
                 in_channels=[512, 1024, 2048],
                 feat_strides=[8, 16, 32],
                 hidden_dim=256,
                 nhead=8,
                 dim_feedforward = 1024,
                 dropout=0.0,
                 enc_act='gelu',
                 encoder_type='VimEncoder',
                 use_encoder_idx=[2],
                 num_encoder_layers=3,
                 pe_temperature=10000,
                 expansion=1.0,
                 depth_mult=1.0,
                 upsampling_type='sub_pixel',
                 act='silu',
                 eval_spatial_size=None, 
                 version='v2'):
        super().__init__()
        self.in_channels = in_channels
        self.feat_strides = feat_strides
        self.hidden_dim = hidden_dim
        self.use_encoder_idx = use_encoder_idx
        self.num_encoder_layers = num_encoder_layers
        self.pe_temperature = pe_temperature
        self.eval_spatial_size = eval_spatial_size        
        self.out_channels = [hidden_dim for _ in range(len(in_channels))]
        self.out_strides = feat_strides
        
        # channel projection
        self.input_proj = nn.ModuleList()
        for in_channel in in_channels:
            if version == 'v1':
                proj = nn.Sequential(
                    nn.Conv2d(in_channel, hidden_dim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(hidden_dim))
            elif version == 'v2':
                proj = nn.Sequential(OrderedDict([
                    ('conv', nn.Conv2d(in_channel, hidden_dim, kernel_size=1, bias=False)),
                    ('norm', nn.BatchNorm2d(hidden_dim))
                ]))
            else:
                raise AttributeError()
                
            self.input_proj.append(proj)

        # encoder
        if encoder_type == 'VimEncoder':
            print("vi,")
            encoder_layer = VimBlock(
                D=hidden_dim,
                E=dim_feedforward,
                activation=enc_act
            )
        else:
            print("tra")
            encoder_layer = TransformerEncoderLayer(
                hidden_dim, 
                nhead=nhead,
                dim_feedforward=dim_feedforward, 
                dropout=dropout,
                activation=enc_act)

        self.encoder = nn.ModuleList([
            Encoder(copy.deepcopy(encoder_layer), num_encoder_layers) for _ in range(len(use_encoder_idx))
        ])

        # top-down fpn
        self.lateral_convs = nn.ModuleList()
        self.fpn_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1, 0, -1):
            self.lateral_convs.append(ConvNormLayer(hidden_dim, hidden_dim, 1, 1, act=act))
            self.fpn_blocks.append(
                CSPRepLayer(hidden_dim * 2, hidden_dim, round(3 * depth_mult), act=act, expansion=expansion)
            )

        # bottom-up pan
        self.downsample_convs = nn.ModuleList()
        self.pan_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1):
            self.downsample_convs.append(
                ConvNormLayer(hidden_dim, hidden_dim, 3, 2, act=act)
            )
            self.pan_blocks.append(
                CSPRepLayer(hidden_dim * 2, hidden_dim, round(3 * depth_mult), act=act, expansion=expansion)
            )

        if upsampling_type == 'sub_pixel':
            print("sp")
            self.upsampling = SubPixelUpsample(hidden_dim, hidden_dim)
        else:
            print("else")
            self.upsampling = nn.Upsample(scale_factor=2, mode='nearest')


        self._reset_parameters()

    def _reset_parameters(self):
        if self.eval_spatial_size:
            for idx in self.use_encoder_idx:
                stride = self.feat_strides[idx]
                pos_embed = self.build_2d_sincos_position_embedding(
                    self.eval_spatial_size[1] // stride, self.eval_spatial_size[0] // stride,
                    self.hidden_dim, self.pe_temperature)
                setattr(self, f'pos_embed{idx}', pos_embed)
                # self.register_buffer(f'pos_embed{idx}', pos_embed)

    @staticmethod
    def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000.):
        """
        """
        grid_w = torch.arange(int(w), dtype=torch.float32)
        grid_h = torch.arange(int(h), dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='ij')
        assert embed_dim % 4 == 0, \
            'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature ** omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]

        return torch.concat([out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()], dim=1)[None, :, :]

    def forward(self, feats):
        assert len(feats) == len(self.in_channels)
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        
        # encoder
        if self.num_encoder_layers > 0:
            for i, enc_ind in enumerate(self.use_encoder_idx):
                h, w = proj_feats[enc_ind].shape[2:]
                # flatten [B, C, H, W] to [B, HxW, C]
                src_flatten = proj_feats[enc_ind].flatten(2).permute(0, 2, 1)
                if self.training or self.eval_spatial_size is None:
                    pos_embed = self.build_2d_sincos_position_embedding(
                        w, h, self.hidden_dim, self.pe_temperature).to(src_flatten.device)
                else:
                    pos_embed = getattr(self, f'pos_embed{enc_ind}', None).to(src_flatten.device)

                memory :torch.Tensor = self.encoder[i](src_flatten, pos_embed=pos_embed)
                proj_feats[enc_ind] = memory.permute(0, 2, 1).reshape(-1, self.hidden_dim, h, w).contiguous()

        # broadcasting and fusion
        inner_outs = [proj_feats[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_heigh = inner_outs[0]
            feat_low = proj_feats[idx - 1]
            feat_heigh = self.lateral_convs[len(self.in_channels) - 1 - idx](feat_heigh)
            inner_outs[0] = feat_heigh
            upsample_feat = self.upsampling(feat_heigh)
            inner_out = self.fpn_blocks[len(self.in_channels)-1-idx](torch.concat([upsample_feat, feat_low], dim=1))
            inner_outs.insert(0, inner_out)

        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_height = inner_outs[idx + 1]
            downsample_feat = self.downsample_convs[idx](feat_low)
            out = self.pan_blocks[idx](torch.concat([downsample_feat, feat_height], dim=1))
            outs.append(out)

        return outs
