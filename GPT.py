# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

"""Full definition of a decoder-only transformer-based language model, all of it in this single file.
 
Based on the nanoGPT implementation: https://github.com/karpathy/nanoGPT and
https://github.com/EleutherAI/gpt-neox/tree/main/megatron/model.
"""

import math
from typing import Any, Optional, Tuple, Union
import warnings

import torch
import torch.nn as nn
from typing_extensions import Self

from litgpt.config import Config

from transformers.utils import(
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
)

# if is_flash_attn_2_available():
#     from flash_attn import flash_attn_func, flash_attn_varlen_func
#     from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input

logger = logging.get_logger(__name__)

class GPT(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        assert config.padded_vocab_size is not None
        self.config = config

        self.lm_head = nn.Linear(config.n_embd, config.padded_vocab_size, bias=config.lm_head_bias)
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.padded_vocab_size, config.n_embd),
                h=nn.ModuleList(Block(config) for _ in range(config.n_layer)),
                ln_f=config.norm_class(config.n_embd, eps=config.norm_eps),
            )
        )
        self.max_seq_length = self.config.block_size
        self.mask_cache: Optional[torch.Tensor] = None

    @property
    def max_seq_length(self) -> int:
        return self._max_seq_length

    @max_seq_length.setter
    def max_seq_length(self, value: int) -> None:
        """
        When doing inference, the sequences used might be shorter than the model's context length.
        This allows setting a smaller number to avoid allocating unused memory
        """
        if value > self.config.block_size:
            raise ValueError(f"Cannot attend to {value}, block size is only {self.config.block_size}")
        self._max_seq_length = value
        if not hasattr(self, "cos"):
            # first call
            cos, sin = self.rope_cache()
            self.register_buffer("cos", cos, persistent=False)
            self.register_buffer("sin", sin, persistent=False)
        # override
        elif value != self.cos.size(0):
            self.cos, self.sin = self.rope_cache(device=self.cos.device)
        # the mask and kv cache size will get updated on `set_kv_cache`. we cannot update it here because we don't know
        # if the kv cache is expected

    def reset_parameters(self) -> None:
        # Trigger resetting the rope-cache
        self.cos, self.sin = self.rope_cache(device=self.cos.device)

    def _init_weights(self, module: nn.Module) -> None:
        """Meant to be used with `gpt.apply(gpt._init_weights)`."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, input_pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        T = idx.size(1)
        if self.max_seq_length < T:
            raise ValueError(f"Cannot forward sequence of length {T}, max seq length is only {self.max_seq_length}.")

        if input_pos is not None:  # use the kv cache
            cos = self.cos.index_select(0, input_pos)
            sin = self.sin.index_select(0, input_pos)
            if self.mask_cache is None:
                raise TypeError("You need to call `gpt.set_kv_cache()`")
            mask = self.mask_cache.index_select(2, input_pos)
        else:
            cos = self.cos[:T]
            sin = self.sin[:T]
            mask = None

        x = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        if self.config.scale_embeddings:
            x = x * (self.config.n_embd**0.5)

        for block in self.transformer.h:
            x = block(x, cos, sin, mask, input_pos)
        x = self.transformer.ln_f(x)
        return self.lm_head(x)  # (b, t, vocab_size)

    @classmethod
    def from_name(cls, name: str, **kwargs: Any) -> Self:
        return cls(Config.from_name(name, **kwargs))

    def rope_cache(self, device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        return build_rope_cache(
            seq_len=self.max_seq_length,
            n_elem=self.config.rope_n_elem,
            device=device,
            condense_ratio=self.config.rope_condense_ratio,
            base=self.config.rope_base,
        )

    def set_kv_cache(
        self,
        batch_size: int,
        rope_cache_length: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        if rope_cache_length is None:
            rope_cache_length = self.cos.size(-1)
        max_seq_length = self.max_seq_length

        # initialize the kv cache for all blocks
        for block in self.transformer.h:
            block.attn.kv_cache = block.attn.build_kv_cache(
                batch_size, max_seq_length, rope_cache_length, device, dtype
            )

        if self.mask_cache is None or self.mask_cache.size(3) != max_seq_length:
            # passing `attn_mask` to SDPA disables the flash implementation. since we only need the mask
            # for the kv-cache support (only during inference), we only create it in that situation
            self.mask_cache = build_mask_cache(max_seq_length, device)

    def clear_kv_cache(self) -> None:
        self.mask_cache = None
        for block in self.transformer.h:
            block.attn.kv_cache = None


class Block(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        if not config.parallel_residual and config.shared_attention_norm:
            raise NotImplementedError(
                "No checkpoint amongst the ones we support uses this configuration"
                " (non-parallel residual and shared attention norm)."
            )

        self.norm_1 = config.norm_class(config.n_embd, eps=config.norm_eps)
        self.attn = CausalSelfAttention(config)
        self.norm_2 = None if config.shared_attention_norm else config.norm_class(config.n_embd, eps=config.norm_eps)
        self.mlp = config.mlp_class(config)

        self.config = config

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Non-parallel residual       Parallel residual
           ┌─ x                     ┌─ x ────────────┐             Note: if `shared_attention_norm` is True,
           │  ↓                     │  ↓             ↓                   the output from `norm_1` is reused
           │  norm_1                │  norm_1  ───►  norm_2
           │  ↓                     │  ↓             ↓
           │  attn                  │  attn          mlp
           │  ↓                     │  ↓             │
        ┌─ └► +                     └► + ◄───────────┘
        │     norm_2
        │     ↓
        │     mlp
        │     ↓
        └───► +
        """

        x_normed = self.norm_1(x)
        attention_output = self.attn(x_normed, cos, sin, mask, input_pos)

        if self.config.parallel_residual:
            x_normed = x_normed if self.config.shared_attention_norm else self.norm_2(x)
            x = self.mlp(x_normed) + attention_output + x
        else:
            x = attention_output + x
            x = self.mlp(self.norm_2(x)) + x
        return x


class CausalSelfAttention(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        shape = (config.n_head + 2 * config.n_query_groups) * config.head_size
        # key, query, value projections for all heads, but in a batch
        self.attn = nn.Linear(config.n_embd, shape, bias=config.bias)
        # output projection
        # if `head_size` is explicitly specified in the config, `n_emd` might not be equal to `head_size * n_head`
        self.proj = nn.Linear(config.head_size * config.n_head, config.n_embd, bias=config.bias)
        # disabled by default
        self.kv_cache: Optional[KVCache] = None

        self.config = config

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        qkv = self.attn(x)

        # assemble into a number of query groups to support MHA, MQA and GQA together (see `config.n_query_groups`)
        q_per_kv = self.config.n_head // self.config.n_query_groups
        total_qkv = q_per_kv + 2  # each group has 1+ queries, 1 key, and 1 value
        qkv = qkv.view(B, T, self.config.n_query_groups, total_qkv, self.config.head_size)
        qkv = qkv.permute(0, 2, 3, 1, 4)  # (B, n_query_groups, total_qkv, T, hs)

        # split batched computation into three
        q, k, v = qkv.split((q_per_kv, 1, 1), dim=2)

        # maybe repeat k and v if for the non multi-head attention cases
        # training: flash attention requires it
        # inference: multi-query would require a full kv cache so avoid it to limit its memory usage
        if self.config.n_query_groups != self.config.n_head and (input_pos is None or self.config.n_query_groups != 1):
            k = k.expand(B, self.config.n_query_groups, q_per_kv, T, self.config.head_size)
            v = v.expand(B, self.config.n_query_groups, q_per_kv, T, self.config.head_size)

        q = q.reshape(B, -1, T, self.config.head_size)  # (B, nh_q, T, hs)
        k = k.reshape(B, -1, T, self.config.head_size)  # (B, nh_k, T, hs)
        v = v.reshape(B, -1, T, self.config.head_size)  # (B, nh_v, T, hs)

        q_roped = apply_rope(q[..., : self.config.rope_n_elem], cos, sin)
        k_roped = apply_rope(k[..., : self.config.rope_n_elem], cos, sin)
        q = torch.cat((q_roped, q[..., self.config.rope_n_elem :]), dim=-1)
        k = torch.cat((k_roped, k[..., self.config.rope_n_elem :]), dim=-1)

        if input_pos is not None:
            if not isinstance(self.kv_cache, KVCache):
                raise TypeError("You need to call `gpt.set_kv_cache()`")
            k, v = self.kv_cache(input_pos, k, v)

        y = self.scaled_dot_product_attention(q, k, v, mask)

        y = y.reshape(B, T, self.config.head_size * self.config.n_head)  # re-assemble all head outputs side by side

        # output projection
        return self.proj(y)

    def scaled_dot_product_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        scale = 1.0 / math.sqrt(self.config.head_size)
        y = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=0.0, scale=scale, is_causal=mask is None
        )
        return y.transpose(1, 2)

    def build_kv_cache(
        self,
        batch_size: int,
        max_seq_length: int,
        rope_cache_length: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "KVCache":
        heads = 1 if self.config.n_query_groups == 1 else self.config.n_head
        v_shape = (batch_size, heads, max_seq_length, self.config.head_size)
        if rope_cache_length is None:
            if self.config.rotary_percentage != 1.0:
                raise TypeError("Please pass the `rope_cache_length=gpt.cos.size(-1)` value")
            k_shape = v_shape
        else:
            k_shape = (
                batch_size,
                heads,
                max_seq_length,
                rope_cache_length + self.config.head_size - self.config.rope_n_elem,
            )
        return KVCache(k_shape, v_shape, device=device, dtype=dtype)


class GptNeoxMLP(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.fc = nn.Linear(config.n_embd, config.intermediate_size, bias=config.bias)
        self.proj = nn.Linear(config.intermediate_size, config.n_embd, bias=config.bias)

        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = torch.nn.functional.gelu(x, approximate=self.config.gelu_approximate)
        return self.proj(x)


class LLaMAMLP(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.fc_1 = nn.Linear(config.n_embd, config.intermediate_size, bias=config.bias)
        self.fc_2 = nn.Linear(config.n_embd, config.intermediate_size, bias=config.bias)
        self.proj = nn.Linear(config.intermediate_size, config.n_embd, bias=config.bias)

        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fc_1 = self.fc_1(x)
        x_fc_2 = self.fc_2(x)
        x = torch.nn.functional.silu(x_fc_1) * x_fc_2
        return self.proj(x)


class GemmaMLP(LLaMAMLP):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fc_1 = self.fc_1(x)
        x_fc_2 = self.fc_2(x)
        x = torch.nn.functional.gelu(x_fc_1, approximate=self.config.gelu_approximate) * x_fc_2
        return self.proj(x)


class LLaMAMoE(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.gate = nn.Linear(config.n_embd, config.n_expert, bias=False)
        self.experts = nn.ModuleList(LLaMAMLP(config) for _ in range(config.n_expert))

        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Derived from: https://github.com/mistralai/mistral-src/blob/b46d6/moe_one_file_ref.py#L203-L219
        See also figure 1 in https://arxiv.org/abs/2211.15841
        """
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)
        x = x.view(-1, C)  # (B*T, C)
        router = self.gate(x)  # (B*T, n_expert)
        probs, indices = torch.topk(router, self.config.n_expert_per_token)  # (B*T, n_expert_per_token)
        probs = probs.softmax(dim=1, dtype=torch.float).to(dtype=x.dtype)
        masks = indices.unsqueeze(-1) == torch.arange(self.config.n_expert, device=x.device)
        masks = masks.permute(2, 0, 1)  # (n_expert, B*T, n_expert_per_token)
        y = torch.zeros_like(x)  # (B*T, C)
        for mask, expert in zip(masks, self.experts):
            token_idx, expert_idx = torch.where(mask)
            y[token_idx] += probs[token_idx, expert_idx, None] * expert(x[token_idx])
        return y.view(B, T, C)


def build_rope_cache(
    seq_len: int, n_elem: int, device: Optional[torch.device] = None, base: int = 10000, condense_ratio: int = 1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Enhanced Transformer with Rotary Position Embedding.

    Derived from: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/
    transformers/rope/__init__.py. MIT License:
    https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/license.
    """
    # $\Theta = {\theta_i = 10000^{\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$
    theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, device=device).float() / n_elem))

    # Create position indexes `[0, 1, ..., seq_len - 1]`
    seq_idx = torch.arange(seq_len, device=device) / condense_ratio

    # Calculate the product of position index and $\theta_i$
    idx_theta = torch.outer(seq_idx, theta).repeat(1, 2)

    return torch.cos(idx_theta), torch.sin(idx_theta)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    head_size = x.size(-1)
    x1 = x[..., : head_size // 2]  # (B, nh, T, hs/2)
    x2 = x[..., head_size // 2 :]  # (B, nh, T, hs/2)
    rotated = torch.cat((-x2, x1), dim=-1)  # (B, nh, T, hs)
    roped = (x * cos) + (rotated * sin)
    return roped.to(dtype=x.dtype)


class KVCache(nn.Module):
    def __init__(
        self,
        k_shape: Tuple[int, int, int, int],
        v_shape: Tuple[int, int, int, int],
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.register_buffer("k", torch.zeros(k_shape, device=device, dtype=dtype), persistent=False)
        self.register_buffer("v", torch.zeros(v_shape, device=device, dtype=dtype), persistent=False)

    def forward(self, input_pos: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # move the buffer to the activation dtype for when AMP is used
        self.k = self.k.to(k.dtype)
        self.v = self.v.to(v.dtype)
        # update the cache
        k = self.k.index_copy_(2, input_pos, k)
        v = self.v.index_copy_(2, input_pos, v)
        return k, v

    def reset_parameters(self) -> None:
        torch.nn.init.zeros_(self.k)
        torch.nn.init.zeros_(self.v)


def build_mask_cache(max_seq_length: int, device: Optional[torch.device] = None) -> torch.Tensor:
    ones = torch.ones((max_seq_length, max_seq_length), device=device, dtype=torch.bool)
    return torch.tril(ones).unsqueeze(0).unsqueeze(0)


class RMSNorm(torch.nn.Module):
    """Root Mean Square Layer Normalization.

    Derived from https://github.com/bzhangGo/rmsnorm/blob/master/rmsnorm_torch.py. BSD 3-Clause License:
    https://github.com/bzhangGo/rmsnorm/blob/master/LICENSE.
    """

    def __init__(self, size: int, dim: int = -1, eps: float = 1e-6, add_unit_offset: bool = False) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(size))
        self.eps = eps
        self.dim = dim
        self.add_unit_offset = add_unit_offset

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        # NOTE: the original RMSNorm paper implementation is not equivalent
        norm_x = torch.mean(x * x, dim=self.dim, keepdim=True)
        x_normed = x * torch.rsqrt(norm_x + self.eps)
        x_normed = x_normed.to(dtype=dtype)
        if self.add_unit_offset:
            # Gemma model requires a unit offset
            # https://github.com/google/gemma_pytorch/blob/main/gemma/model.py#L176
            return x_normed * (1 + self.weight)
        return x_normed * self.weight

    def reset_parameters(self) -> None:
        torch.nn.init.ones_(self.weight)


# class GPT2FlashAttention2(CausalSelfAttention):
#     """
#     GPT2 flash attention module. This module inherits from `GPT2Attention` as the weights of the module stays
#     untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
#     flash attention and deal with padding tokens in case the input contains any of them.
#     """

#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#         # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
#         # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
#         # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
#         self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

#     def forward(
#         self,
#         hidden_states: Optional[Tuple[torch.FloatTensor]],
#         layer_past: Optional[Tuple[torch.Tensor]] = None,
#         attention_mask: Optional[torch.FloatTensor] = None,
#         head_mask: Optional[torch.FloatTensor] = None,
#         encoder_hidden_states: Optional[torch.Tensor] = None,
#         encoder_attention_mask: Optional[torch.FloatTensor] = None,
#         use_cache: Optional[bool] = False,
#         output_attentions: Optional[bool] = False,
#     ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
#         bsz, _, _ = hidden_states.size()
#         if encoder_hidden_states is not None:
#             if not hasattr(self, "q_attn"):
#                 raise ValueError(
#                     "If class is used as cross attention, the weights `q_attn` have to be defined. "
#                     "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
#                 )

#             query = self.q_attn(hidden_states)
#             key, value = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
#             attention_mask = encoder_attention_mask
#         else:
#             query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)

#         query = self._split_heads(query, self.num_heads, self.head_dim)
#         key = self._split_heads(key, self.num_heads, self.head_dim)
#         value = self._split_heads(value, self.num_heads, self.head_dim)

#         if layer_past is not None:
#             past_key = layer_past[0]
#             past_value = layer_past[1]
#             key = torch.cat((past_key, key), dim=-2)
#             value = torch.cat((past_value, value), dim=-2)

#         present = None
#         if use_cache is True:
#             present = (key, value)

#         query_length = query.shape[2]
#         tgt_len = key.shape[2]

#         # Flash attention requires the input to have the shape
#         # batch_size x seq_length x head_dim x hidden_dim
#         query = query.transpose(1, 2).view(bsz, query_length, self.num_heads, self.head_dim)
#         key = key.transpose(1, 2).view(bsz, tgt_len, self.num_heads, self.head_dim)
#         value = value.transpose(1, 2).view(bsz, tgt_len, self.num_heads, self.head_dim)

#         attn_dropout = self.attn_dropout.p if self.training else 0.0

#         # In PEFT, usually we cast the layer norms in float32 for training stability reasons
#         # therefore the input hidden states gets silently casted in float32. Hence, we need
#         # cast them back in the correct dtype just to be sure everything works as expected.
#         # This might slowdown training & inference so it is recommended to not cast the LayerNorms
#         # in fp32. (LlamaRMSNorm handles it correctly)

#         if query.dtype == torch.float32:
#             if torch.is_autocast_enabled():
#                 target_dtype = torch.get_autocast_gpu_dtype()
#             # Handle the case where the model is quantized
#             elif hasattr(self.config, "_pre_quantization_dtype"):
#                 target_dtype = self.config._pre_quantization_dtype
#             else:
#                 target_dtype = self.c_proj.weight.dtype

#             logger.warning_once(
#                 f"The input hidden states seems to be silently casted in float32, this might be related to"
#                 f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
#                 f" {target_dtype}."
#             )

#             query = query.to(target_dtype)
#             key = key.to(target_dtype)
#             value = value.to(target_dtype)

#         attn_output = self._flash_attention_forward(
#             query, key, value, attention_mask, query_length, dropout=attn_dropout
#         )

#         attn_weights_reshaped = attn_output.reshape(bsz, query_length, self.num_heads * self.head_dim)
#         attn_output = self.c_proj(attn_weights_reshaped)
#         attn_output = self.resid_dropout(attn_output)

#         outputs = (attn_output, present)
#         if output_attentions:
#             outputs += (attn_weights_reshaped,)

#         return outputs

#     # Copied from transformers.models.llama.modeling_llama.LlamaFlashAttention2._flash_attention_forward
#     def _flash_attention_forward(
#         self, query_states, key_states, value_states, attention_mask, query_length, dropout=0.0, softmax_scale=None
#     ):
#         """
#         Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
#         first unpad the input, then computes the attention scores and pad the final attention scores.

#         Args:
#             query_states (`torch.Tensor`):
#                 Input query states to be passed to Flash Attention API
#             key_states (`torch.Tensor`):
#                 Input key states to be passed to Flash Attention API
#             value_states (`torch.Tensor`):
#                 Input value states to be passed to Flash Attention API
#             attention_mask (`torch.Tensor`):
#                 The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
#                 position of padding tokens and 1 for the position of non-padding tokens.
#             dropout (`float`):
#                 Attention dropout
#             softmax_scale (`float`, *optional*):
#                 The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
#         """
#         if not self._flash_attn_uses_top_left_mask:
#             causal = self.is_causal
#         else:
#             # TODO: Remove the `query_length != 1` check once Flash Attention for RoCm is bumped to 2.1. For details, please see the comment in LlamaFlashAttention2 __init__.
#             causal = self.is_causal and query_length != 1

#         # Contains at least one padding token in the sequence
#         if attention_mask is not None:
#             batch_size = query_states.shape[0]
#             query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
#                 query_states, key_states, value_states, attention_mask, query_length
#             )

#             cu_seqlens_q, cu_seqlens_k = cu_seq_lens
#             max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

#             attn_output_unpad = flash_attn_varlen_func(
#                 query_states,
#                 key_states,
#                 value_states,
#                 cu_seqlens_q=cu_seqlens_q,
#                 cu_seqlens_k=cu_seqlens_k,
#                 max_seqlen_q=max_seqlen_in_batch_q,
#                 max_seqlen_k=max_seqlen_in_batch_k,
#                 dropout_p=dropout,
#                 softmax_scale=softmax_scale,
#                 causal=causal,
#             )

#             attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
#         else:
#             attn_output = flash_attn_func(
#                 query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, causal=causal
#             )

#         return attn_output

#     # Copied from transformers.models.llama.modeling_llama.LlamaFlashAttention2._upad_input
#     def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
#         indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
#         batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

#         key_layer = index_first_axis(
#             key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
#         )
#         value_layer = index_first_axis(
#             value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
#         )
#         if query_length == kv_seq_len:
#             query_layer = index_first_axis(
#                 query_layer.reshape(batch_size * kv_seq_len, self.num_heads, head_dim), indices_k
#             )
#             cu_seqlens_q = cu_seqlens_k
#             max_seqlen_in_batch_q = max_seqlen_in_batch_k
#             indices_q = indices_k
#         elif query_length == 1:
#             max_seqlen_in_batch_q = 1
#             cu_seqlens_q = torch.arange(
#                 batch_size + 1, dtype=torch.int32, device=query_layer.device
#             )  # There is a memcpy here, that is very bad.
#             indices_q = cu_seqlens_q[:-1]
#             query_layer = query_layer.squeeze(1)
#         else:
#             # The -q_len: slice assumes left padding.
#             attention_mask = attention_mask[:, -query_length:]
#             query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

#         return (
#             query_layer,
#             key_layer,
#             value_layer,
#             indices_q,
#             (cu_seqlens_q, cu_seqlens_k),
#             (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
#         )

