# coding=utf-8
# Copyright 2018 The OpenAI Team Authors and HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch OpenAI GPT-2 model."""

import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from packaging import version
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.models.gpt2.modeling_gpt2 import load_tf_weights_in_gpt2, GPT2LMHeadModel, GPT2MLP, GPT2Attention, GPT2Block, GPT2Model 

from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from transformers.modeling_utils import PreTrainedModel, SequenceSummary
from transformers.pytorch_utils import Conv1D, find_pruneable_heads_and_indices, prune_conv1d_layer
from transformers.utils import (
    ModelOutput,
    logging,
)
from transformers.utils.model_parallel_utils import assert_device_map, get_device_map
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions


if version.parse(torch.__version__) >= version.parse("1.6"):
    is_amp_available = True
    from torch.cuda.amp import autocast
else:
    is_amp_available = False

    
class ThisGPT2Config(GPT2Config):
    model_type = "this_gpt2"

    def __init__(
        self,
        cross_attention_reduce_factor = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.cross_attention_reduce_factor = cross_attention_reduce_factor

class ThisGPT2Attention(GPT2Attention): #交叉注意力层
    def __init__(self, config, is_cross_attention=False, layer_idx=None):
        super().__init__(config, is_cross_attention, layer_idx)

        #print("this gpt2")

        #print("self.is_cross_attention = is_cross_attention", self.is_cross_attention, is_cross_attention)
        
        self.cross_attention_reduce_factor = config.cross_attention_reduce_factor
        
        if self.is_cross_attention:
            self.c_attn = Conv1D(int(2 / self.cross_attention_reduce_factor * self.embed_dim), 
                                                                                  self.embed_dim) 
            self.q_attn = Conv1D(int(self.embed_dim / self.cross_attention_reduce_factor), self.embed_dim)
            self.c_proj = Conv1D(self.embed_dim, int(self.embed_dim / self.cross_attention_reduce_factor))
        else:
            self.c_attn = Conv1D(3 * self.embed_dim, self.embed_dim)
            self.c_proj = Conv1D(self.embed_dim, self.embed_dim)

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
        if encoder_hidden_states is not None:
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
                )
            split_size = int(self.split_size / self.cross_attention_reduce_factor)
            head_dim = int(self.head_dim / self.cross_attention_reduce_factor)

            query = self.q_attn(hidden_states)
            key, value = self.c_attn(encoder_hidden_states).split(split_size, dim=2)
            attention_mask = encoder_attention_mask

            query = self._split_heads(query, self.num_heads, head_dim)
            key = self._split_heads(key, self.num_heads, head_dim)
            value = self._split_heads(value, self.num_heads, head_dim)
        else:
            query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
        
            query = self._split_heads(query, self.num_heads, self.head_dim)
            key = self._split_heads(key, self.num_heads, self.head_dim)
            value = self._split_heads(value, self.num_heads, self.head_dim)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        if self.reorder_and_upcast_attn:
            attn_output, attn_weights = self._upcast_and_reordered_attn(query, key, value, attention_mask, head_mask)
        else:
            attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)
        
        attn_output = self._merge_heads(attn_output, self.num_heads, int(self.head_dim / self.cross_attention_reduce_factor))
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)

    def _split_heads(self, x, num_heads, head_dim):
        # x shape: [batch_size, seq_length, hidden_dim]
        new_shape = x.size()[:-1] + (num_heads, head_dim)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)  # [B, num_heads, seq_len, head_dim]
    
    def _merge_heads(self, x, num_heads, head_dim):
        # x shape: [batch_size, num_heads, seq_length, head_dim]
        x = x.permute(0, 2, 1, 3).contiguous()  # [B, seq_len, num_heads, head_dim]
        new_shape = x.size()[:-2] + (num_heads * head_dim,)
        return x.view(*new_shape)

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
    # Scaled dot-product attention
        attn_weights = torch.matmul(query, key.transpose(-1, -2))
        attn_weights = attn_weights / math.sqrt(value.size(-1))

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask  # attention_mask should be broadcastable

        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)
        return attn_output, attn_weights



class ThisGPT2Block(GPT2Block):
    def __init__(self, config, layer_idx=None):
        super().__init__(config, layer_idx)
        hidden_size = config.hidden_size

        if config.add_cross_attention:
            self.crossattention = ThisGPT2Attention(config, is_cross_attention=True, layer_idx=layer_idx)
            self.ln_cross_attn = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

class ThisGPT2Model(GPT2Model):

    def __init__(self, config):
        super().__init__(config)
        self.h = nn.ModuleList([ThisGPT2Block(config, layer_idx=i) for i in range(config.num_hidden_layers)])


class ThisGPT2LMHeadModel(GPT2LMHeadModel):
    config_class = ThisGPT2Config
    
    def __init__(self, config):
        super().__init__(config)
        self.transformer = ThisGPT2Model(config)
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        encoder_hidden_states=None,
        pixel_values=None,  # ✅ 图像特征
        **kwargs
        ):
    # 传入 transformer 时加入 cross-attn 的图像信息
        print(f'encoder hidden states:{encoder_hidden_states}')
        if encoder_hidden_states is None and pixel_values is not None:
            encoder_outputs = self.vision_encoder(pixel_values=pixel_values)
            encoder_hidden_states = encoder_outputs.last_hidden_state
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            **kwargs
    )

        hidden_states = outputs[0]
        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            hidden_states=hidden_states,
            attentions=None,
            cross_attentions=None
            )


