# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch
from torch import nn, Tensor

from sam2_train.modeling.sam.transformer import RoPEAttention

from sam2_train.modeling.sam2_utils import get_activation_fn, get_clones

from torchvision.utils import save_image
import os
from PIL import Image
import itertools
visualization_counter = itertools.count()
visualization_counter2 = itertools.count()

def visualize_memory(memory, save_dir="visualizations", batch_idx=0):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os

    # # Memory 數值分佈
    # memory = memory.detach().contiguous()
    # flattened_memory = memory.view(-1).cpu().numpy()
    # os.makedirs(save_dir, exist_ok=True)

    # # 可視化 Patch 特徵
    # sample_memory = memory[0].cpu().numpy()  # 假設只看 batch 0
    # sns.heatmap(sample_memory[:100, :], cmap="viridis")
    # plt.title("Patch Feature Heatmap")
    # plt.xlabel("Feature Dimension")
    # plt.ylabel("Patch Index")
    # # plt.show()
    # heatmap_path = os.path.join(save_dir, f"patch_heatmap_batch{batch_idx}.png")
    # plt.savefig(heatmap_path)
    # plt.close()

def reshape_tgt_to_image(tgt):
    # 假設 `tgt` 為 [B, 4096, 256]
    B, N, D = tgt.shape

    # 確認序列長度是否為 4096 並可映射為 64x64
    assert N == 64 * 64, "N must be 4096 (64x64)"

    # 降維到 3 通道
    tgt = tgt.permute(0, 2, 1)  # [B, 256, 4096]
    tgt = tgt.view(B, D, 64, 64)  # [B, 256, 64, 64]

    # 選取前 3 通道
    tgt_image = tgt[:, :3, :, :]  # [B, 3, 64, 64]
    return tgt_image

def save_tensor_as_rgb_image(tensor, save_path):
    # 如果張量是 [C, H, W]，轉換為 [H, W, C]
    if tensor.dim() == 3 and tensor.size(0) == 3:
        tensor = tensor.permute(1, 2, 0)  # [H, W, C]
    
    # 如果範圍不是 [0, 1]，進行標準化
    if tensor.max() > 1.0:
        tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
    
    # 將範圍轉換到 [0, 255]
    tensor = (tensor * 255).byte()
    
    # 將張量轉為 numpy 格式
    np_image = tensor.cpu().numpy()
    
    # 使用 PIL 保存為圖片
    img = Image.fromarray(np_image, mode='RGB')
    img.save(save_path)

def save_tensor_as_image(tensor, save_path):
    """
    保存形狀為 [2, 4096, 256] 的張量作為 RGB 圖像。
    """
    # 確保是 3 維張量 [N, 4096, 256]
    assert tensor.dim() == 3 and tensor.size(1) == 4096 and tensor.size(2) >= 3, \
        "Tensor shape must be [N, 4096, 256] and at least 3 channels."
    
    # 取第一個 batch
    tensor = tensor[0]  # shape [4096, 256]

    # 將 4096 解構為 [64, 64]
    h, w = 64, 64
    tensor = tensor.view(h, w, -1)  # shape [64, 64, 256]
    
    # 提取前 3 個通道作為 RGB 圖像
    tensor = tensor[:, :, :3]  # shape [64, 64, 3]
    
    # 標準化到範圍 [0, 1]
    if tensor.max() > 1.0:
        tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
    
    # 轉換到 [0, 255]
    tensor = (tensor * 255).byte()
    
    # 將張量轉為 numpy 格式
    np_image = tensor.cpu().numpy()
    
    # 保存圖片
    img = Image.fromarray(np_image, mode='RGB')
    img.save(save_path)

def save_feature_map(tensor, layer_idx, batch_idx):
    from torchvision.utils import save_image
    import os
    save_dir="visualizations"
    os.makedirs(save_dir, exist_ok=True)
    # 確保 tensor 在 CPU 上並轉換為 [0, 1] 範圍
    tensor = tensor.detach().cpu()
    normalized_tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min() + 1e-8)

    # 確保輸出是可視化的形式 (e.g., batch 為第一維)
    if tensor.ndim == 3:  # [Batch, H, W]
        for i in range(tensor.shape[0]):  # 遍歷 batch
            save_image(
                normalized_tensor[i],
                f"{save_dir}/layer_{layer_idx}_batch_{batch_idx}_img_{i}.png"
            )
    elif tensor.ndim == 4:  # [Batch, Channels, H, W]
        for i in range(tensor.shape[0]):  # 遍歷 batch
            save_image(
                normalized_tensor[i, 0],  # 僅保存第一個通道
                f"{save_dir}/layer_{layer_idx}_batch_{batch_idx}_img_{i}.png"
            )

class MemoryAttentionLayer(nn.Module):

    def __init__(
        self,
        activation: str,
        cross_attention: nn.Module,
        d_model: int,
        dim_feedforward: int,
        dropout: float,
        pos_enc_at_attn: bool,
        pos_enc_at_cross_attn_keys: bool,
        pos_enc_at_cross_attn_queries: bool,
        self_attention: nn.Module,
    ):
        super().__init__()
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        self.dropout_value = dropout
        self.self_attn = self_attention
        self.cross_attn_image = cross_attention

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation_str = activation
        self.activation = get_activation_fn(activation)

        # Where to add pos enc
        self.pos_enc_at_attn = pos_enc_at_attn
        self.pos_enc_at_cross_attn_queries = pos_enc_at_cross_attn_queries
        self.pos_enc_at_cross_attn_keys = pos_enc_at_cross_attn_keys

    def _forward_sa(self, tgt, query_pos):
        # Self-Attention
        tgt2 = self.norm1(tgt)
        q = k = tgt2 + query_pos if self.pos_enc_at_attn else tgt2
        tgt2 = self.self_attn(q, k, v=tgt2)
        tgt = tgt + self.dropout1(tgt2)

        # # print(tgt.shape)
        # batch_idx = next(visualization_counter)
        # layer_idx = next(visualization_counter2)
        # save_path = f"tgtImage/tgt_batch{batch_idx}_layer{layer_idx}.png"
        # # save_tensor_as_rgb_image(tgt_image, save_path)
        # save_tensor_as_image(tgt, save_path)
        # # save_tensor_as_image(tgt, batch_idx, layer_idx)
        return tgt

    def _forward_ca(self, tgt, memory, query_pos, pos, num_k_exclude_rope=0, batch_idx=0):
        kwds = {}
        if num_k_exclude_rope > 0:
            assert isinstance(self.cross_attn_image, RoPEAttention)
            kwds = {"num_k_exclude_rope": num_k_exclude_rope}

        # batch_idx = next(visualization_counter)
        # layer_idx = next(visualization_counter2)
        # save_feature_map(memory, layer_idx, batch_idx)
        # visualize_memory(memory, save_dir="visualizations", batch_idx=batch_idx)

        # Cross-Attention
        tgt2 = self.norm2(tgt)
        tgt2 = self.cross_attn_image(
            q=tgt2 + query_pos if self.pos_enc_at_cross_attn_queries else tgt2,
            k=memory + pos if self.pos_enc_at_cross_attn_keys else memory,
            v=memory,
            **kwds,
        )
        tgt = tgt + self.dropout2(tgt2)
        return tgt

    def forward(
        self,
        tgt,
        memory,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
        num_k_exclude_rope: int = 0,
    ) -> torch.Tensor:

        # Self-Attn, Cross-Attn
        tgt = self._forward_sa(tgt, query_pos)
        tgt = self._forward_ca(tgt, memory, query_pos, pos, num_k_exclude_rope)
        # MLP
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt


class MemoryAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        pos_enc_at_input: bool,
        layer: nn.Module,
        num_layers: int,
        batch_first: bool = True,  # Do layers expect batch first input?
    ):
        super().__init__()
        self.d_model = d_model
        self.layers = get_clones(layer, num_layers)
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(d_model)
        self.pos_enc_at_input = pos_enc_at_input
        self.batch_first = batch_first

    def forward(
        self,
        curr: torch.Tensor,  # self-attention inputs
        memory: torch.Tensor,  # cross-attention inputs
        curr_pos: Optional[Tensor] = None,  # pos_enc for self-attention inputs
        memory_pos: Optional[Tensor] = None,  # pos_enc for cross-attention inputs
        num_obj_ptr_tokens: int = 0,  # number of object pointer *tokens*
    ):
        if isinstance(curr, list):
            assert isinstance(curr_pos, list)
            assert len(curr) == len(curr_pos) == 1
            curr, curr_pos = (
                curr[0],
                curr_pos[0],
            )

        assert (
            curr.shape[1] == memory.shape[1]
        ), "Batch size must be the same for curr and memory"

        output = curr
        if self.pos_enc_at_input and curr_pos is not None:
            output = output + 0.1 * curr_pos

        if self.batch_first:
            # Convert to batch first
            output = output.transpose(0, 1)
            curr_pos = curr_pos.transpose(0, 1)
            memory = memory.transpose(0, 1)
            memory_pos = memory_pos.transpose(0, 1)

        for layer in self.layers:
            kwds = {}
            if isinstance(layer.cross_attn_image, RoPEAttention):
                kwds = {"num_k_exclude_rope": num_obj_ptr_tokens}

            output = layer(
                tgt=output,
                memory=memory,
                pos=memory_pos,
                query_pos=curr_pos,
                **kwds,
            )
        normed_output = self.norm(output)

        if self.batch_first:
            # Convert back to seq first
            normed_output = normed_output.transpose(0, 1)
            curr_pos = curr_pos.transpose(0, 1)

        return normed_output
