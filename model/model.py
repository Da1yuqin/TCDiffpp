from typing import Any, Callable, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torch import Tensor
from torch.nn import functional as F, ModuleList, Linear, Module

from model.rotary_embedding_torch import RotaryEmbedding
from model.utils import PositionalEncoding, SinusoidalPosEmb, prob_mask_like
import random


class ConcatSquashLinear(nn.Module):
    """Conditional linear layer with context modulation.
    
    Combines input features with contextual information through:
    output = Linear(x) * σ(gate(ctx)) + bias(ctx)
    
    Args:
        dim_in: Input feature dimension
        dim_out: Output feature dimension
        dim_ctx: Context feature dimension
    """
    
    def __init__(self, dim_in: int, dim_out: int, dim_ctx: int):
        super().__init__()
        self.main_layer = Linear(dim_in, dim_out)
        self.context_bias = Linear(dim_ctx, dim_out, bias=False)  # Context-to-bias projection
        self.context_gate = Linear(dim_ctx, dim_out)  # Context-to-gate projection

    def forward(self, ctx: Tensor, x: Tensor) -> Tensor:
        """Forward pass with context modulation.
        
        Args:
            ctx: Context tensor [..., dim_ctx]
            x: Input tensor [..., dim_in]
            
        Returns:
            Modulated output tensor [..., dim_out]
        """
        # Compute gating signal from context
        gate = torch.sigmoid(self.context_gate(ctx))
        
        # Compute additive bias from context
        bias = self.context_bias(ctx)
        
        # Apply conditional modulation
        return self.main_layer(x) * gate + bias


# Temporal + Identity Positional Embedding Encoder
class Tem_ID_Encoder(nn.Module):
    """
    Temporal + Identity Positional Encoder

    Adds both temporal positional encodings and identity-specific encodings
    to input features, enabling the model to distinguish time steps and individuals.

    Args:
        latent_dim (int): Dimensionality of the embeddings.
        dropout (float): Dropout rate applied after encoding.
        max_t_len (int): Maximum sequence length for positional encoding.
        max_id_num (int): Maximum number of unique identities for identity encoding.
    """
    def __init__(self, latent_dim, dropout=0.1, max_t_len=500, max_id_num=20):
        super(Tem_ID_Encoder, self).__init__()
        self.latent_dim = latent_dim
        self.dropout = nn.Dropout(p=dropout)

        # Precompute sinusoidal positional encodings for temporal dimension
        pe = self.build_pos_enc(max_t_len)
        self.register_buffer('pe', pe)

        # Precompute sinusoidal encodings for identities
        ie = self.build_id_enc(max_id_num)
        self.register_buffer('ie', ie)

    def build_pos_enc(self, max_len):
        """Construct sinusoidal positional encodings for temporal dimension."""
        pe = torch.zeros(max_len, self.latent_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.latent_dim, 2).float() * (-np.log(10000.0) / self.latent_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # Shape: (1, max_len, latent_dim)
        return pe

    def build_id_enc(self, max_len): 
        """Construct sinusoidal encodings for identity dimension (individuals)."""
        ie = torch.zeros(max_len, self.latent_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.latent_dim, 2).float() * (-np.log(10000.0) / self.latent_dim))
        ie[:, 0::2] = torch.sin(position * div_term)
        ie[:, 1::2] = torch.cos(position * div_term) 
        ie = ie.unsqueeze(0) # Shape: (1, max_len, latent_dim)
        return ie

    def get_pos_enc(self, x, seq_len=150, t_offset=0): 
        """
            Retrieve and repeat positional encodings to match the sequence length.

            - Typical sequence length is 150 frames.
            - Repeats encodings across individuals, ensuring consistent
            temporal encoding for each individual's sequence.

            Args:
                x (Tensor): Input tensor of shape (bs, total_seq_len, latent_dim)
                seq_len (int): Temporal length per individual.
                t_offset (int): Offset to shift the positional encoding.

            Returns:
                Tensor of shape (1, total_seq_len, latent_dim)
        """
        pe = self.pe[:, t_offset :seq_len+ t_offset, :] # (1,seq_len, latent_dim)
        pe = pe.repeat(1, x.shape[1]//seq_len, 1) # (1, seq_len * dancer_num, latent_dim)
        return pe 

    def get_id_enc(self, seq_len, i_offset, id_enc_shuffle):
        """
        Retrieve and repeat identity encodings for individuals.

        - Randomly assigns unique encodings to individuals.
        - Repeats each individual's encoding across its entire time sequence.

        Args:
            seq_len (int): Temporal length per individual.
            i_offset (int): Identity offset (currently unused).
            id_enc_shuffle (List[int]): Random indices mapping identities to encodings.

        Returns:
            Tensor of shape (1, total_seq_len, latent_dim)
        """
        ie = self.ie[:, id_enc_shuffle] # (1, dancer_num, latent_dim)
        ie = ie.repeat_interleave(seq_len, dim=1) # (1, dancer_num * seq_len, latent_dim)
        return ie  # (1, seq_len, latent_dim)

    def forward(self, x, seq_len=150, t_offset=0, i_offset=0): 
        """
        Apply temporal and identity encodings to input features.

        Args:
            x (Tensor): Input tensor of shape (bs, total_seq_len, latent_dim)
                        where total_seq_len = num_individuals * seq_len
            seq_len (int): Temporal length per individual.
            t_offset (int): Temporal offset for positional encoding.
            i_offset (int): Identity offset (currently unused).

        Returns:
            Tensor of shape (bs, total_seq_len, latent_dim)
        """
        dancer_num = 5 # Fixed number of possible dancer identities 
        num_dancers  = x.shape[1]//seq_len 

        # Generate random indices to assign unique identity encodings
        index = list(np.arange(0, dancer_num)) 
        id_enc_shuffle = random.sample(index, num_dancers) 

        # Retrieve encodings
        pos_enc = self.get_pos_enc(x, seq_len, t_offset) # repeat seq_len
        id_enc = self.get_id_enc(seq_len, i_offset, id_enc_shuffle) # repeat num_dancers

        # Add encodings to input
        x = x + pos_enc + id_enc  
        return self.dropout(x)

class DenseFiLM(nn.Module): 
    """Feature-wise linear modulation (FiLM) generator."""

    def __init__(self, embed_channels):
        super().__init__()
        self.embed_channels = embed_channels
        self.block = nn.Sequential(
            nn.Mish(), nn.Linear(embed_channels, embed_channels * 2)
        )

    def forward(self, position): 
        pos_encoding = self.block(position)
        pos_encoding = rearrange(pos_encoding, "b c -> b 1 c")
        scale_shift = pos_encoding.chunk(2, dim=-1) 
        return scale_shift


def featurewise_affine(x, scale_shift):
    scale, shift = scale_shift
    return (scale + 1) * x + shift


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = True,
        device=None,
        dtype=None,
        rotary=None,
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first
        )
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = activation

        self.rotary = rotary
        self.use_rotary = rotary is not None

    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
            x = self.norm2(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(
        self, x: Tensor, attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]
    ) -> Tensor:
        qk = self.rotary.rotate_queries_or_keys(x) if self.use_rotary else x 
        x = self.self_attn( # torch.nn
            qk,
            qk,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class FiLMTransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward=2048,
        dropout=0.1,
        activation=F.relu,
        layer_norm_eps=1e-5,
        batch_first=False,
        norm_first=True,
        device=None,
        dtype=None,
        rotary=None,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first
        )
        self.multihead_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first
        )
        # Feedforward
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = activation

        self.film1 = DenseFiLM(d_model)
        self.film2 = DenseFiLM(d_model)
        self.film3 = DenseFiLM(d_model)

        self.rotary = rotary
        self.use_rotary = rotary is not None

    # x, cond, t
    def forward(
        self,
        tgt, # x
        memory, # cond
        t,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
    ):
        x = tgt 
        # bs, sd, c = tgt.shape # (b, sd*dn, c(151))

        if self.norm_first: # True
            # self-attention -> film -> residual
            x_1 = self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask) # self-attention 
            x = x + featurewise_affine(x_1, self.film1(t)) 
            # cross-attention -> film -> residual
            x_2 = self._mha_block( # cross_attention
                self.norm2(x), memory, memory_mask, memory_key_padding_mask 
            )
            x = x + featurewise_affine(x_2, self.film2(t)) 
            # feedforward -> film -> residual
            x_3 = self._ff_block(self.norm3(x))
            x = x + featurewise_affine(x_3, self.film3(t)) 
        else:
            x = self.norm1(
                x
                + featurewise_affine(
                    self._sa_block(x, tgt_mask, tgt_key_padding_mask), self.film1(t)
                )
            )
            x = self.norm2(
                x
                + featurewise_affine(
                    self._mha_block(x, memory, memory_mask, memory_key_padding_mask),
                    self.film2(t),
                )
            )
            x = self.norm3(x + featurewise_affine(self._ff_block(x), self.film3(t)))
        return x

    # self-attention block
    # qkv
    def _sa_block(self, x, attn_mask, key_padding_mask):
        qk = self.rotary.rotate_queries_or_keys(x) if self.use_rotary else x
        x = self.self_attn(
            qk,
            qk,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        return self.dropout1(x)

    # multihead attention block
    # qkv
    def _mha_block(self, x, mem, attn_mask, key_padding_mask):
        q = self.rotary.rotate_queries_or_keys(x) if self.use_rotary else x # embedding
        k = self.rotary.rotate_queries_or_keys(mem) if self.use_rotary else mem
        x = self.multihead_attn(
            q,
            k,
            mem,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        return self.dropout2(x)

    # feed forward block
    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x)))) 
        return self.dropout3(x)



class FiLMMambaDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward=2048,
        dropout=0.1,
        activation=F.relu,
        layer_norm_eps=1e-5,
        batch_first=False,
        norm_first=True,
        device=None,
        dtype=None,
        rotary=None,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first
        )
        from mamba_addon import CrossMamba
        self.mamba = CrossMamba(d_model = d_model)
        self.multihead_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = activation

        self.film1 = DenseFiLM(d_model)
        self.film2 = DenseFiLM(d_model)
        self.film3 = DenseFiLM(d_model)

        self.rotary = rotary
        self.use_rotary = rotary is not None

    # x, cond, t
    def forward(
        self,
        tgt, 
        memory, 
        t,
        tgt_mask=None, 
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
    ):
        x = tgt 

        if self.norm_first: 
            x_1 = self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask) # self-attention 
            x = x + featurewise_affine(x_1, self.film1(t)) 

            x_2 = self.mamba(self.norm2(x),memory,t)
            x = x + featurewise_affine(x_2, self.film2(t)) 
            # feedforward -> film -> residual
            x_3 = self._ff_block(self.norm3(x)) 
            x = x + featurewise_affine(x_3, self.film3(t)) 
        else:
            x = self.norm1(
                x
                + featurewise_affine(
                    self._sa_block(x, tgt_mask, tgt_key_padding_mask), self.film1(t)
                )
            )
            x = self.norm2(
                x
                + featurewise_affine(
                    self._mha_block(x, memory, memory_mask, memory_key_padding_mask),
                    self.film2(t),
                )
            )
            x = self.norm3(x + featurewise_affine(self._ff_block(x), self.film3(t)))
        return x

    # self-attention block
    # qkv
    def _sa_block(self, x, attn_mask, key_padding_mask):
        qk = self.rotary.rotate_queries_or_keys(x) if self.use_rotary else x
        x = self.self_attn(
            qk,
            qk,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        return self.dropout1(x)

    # multihead attention block
    # qkv
    def _mha_block(self, x, mem, attn_mask, key_padding_mask):
        q = self.rotary.rotate_queries_or_keys(x) if self.use_rotary else x # embedding
        k = self.rotary.rotate_queries_or_keys(mem) if self.use_rotary else mem
        x = self.multihead_attn(
            q,
            k,
            mem,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        return self.dropout2(x)

    # feed forward block
    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x)))) 
        return self.dropout3(x)


class DecoderLayerStack(nn.Module):
    def __init__(self, stack):
        super().__init__()
        self.stack = stack

    def forward(self, x, cond, t):
        for layer in self.stack:
            x = layer(x, cond, t)
        return x


class DanceDecoder(nn.Module):
    def __init__(
        self,
        nfeats: int,
        seq_len: int = 150,  # 5 seconds, 30 fps
        latent_dim: int = 256,
        ff_size: int = 1024,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        cond_feature_dim: int = 4800,
        activation: Callable[[Tensor], Tensor] = F.gelu,
        use_rotary=True,
        required_dancer_num = 3,
        use_ssm = True,
        **kwargs
    ) -> None:

        super().__init__()

        output_feats = nfeats

        # positional embeddings
        self.required_dancer_num = required_dancer_num
        self.latent_dim = latent_dim
        self.rotary = None
        self.abs_pos_encoding = nn.Identity()
        # if rotary, replace absolute embedding with a rotary embedding instance (absolute becomes an identity)
        if use_rotary:
            self.rotary = RotaryEmbedding(dim=latent_dim)
        else:
            self.abs_pos_encoding = PositionalEncoding(
                latent_dim, dropout, batch_first=True
            )

        # time embedding processing
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(latent_dim),  # learned?
            nn.Linear(latent_dim, latent_dim * 4),
            nn.Mish(),
        )

        self.to_time_cond = nn.Sequential(nn.Linear(latent_dim * 4, latent_dim),)

        self.to_time_tokens = nn.Sequential(
            nn.Linear(latent_dim * 4, latent_dim * 2),  # 2 time tokens
            Rearrange("b (r d) -> b r d", r=2),
        )

        # null embeddings for guidance dropout
        self.null_cond_embed = nn.Parameter(torch.randn(1, seq_len, latent_dim))
        self.null_cond_hidden = nn.Parameter(torch.randn(1, latent_dim))

        self.norm_cond = nn.LayerNorm(latent_dim)

        # input projection
        self.input_projection = nn.Linear(nfeats, latent_dim)
        self.cond_encoder = nn.Sequential()
        for _ in range(2):
            self.cond_encoder.append(
                TransformerEncoderLayer(
                    d_model=latent_dim,
                    nhead=num_heads,
                    dim_feedforward=ff_size,
                    dropout=dropout,
                    activation=activation,
                    batch_first=True,
                    rotary=self.rotary,
                )
            )
        # conditional projection
        # self.cond_projection = nn.Linear(cond_feature_dim, latent_dim)
        self.cond_projection = nn.Sequential(  # music-condition only, currently
            nn.Linear(cond_feature_dim * 2, cond_feature_dim),
            nn.ReLU(),
            nn.Linear(cond_feature_dim, latent_dim),
        )

        self.non_attn_cond_projection = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )
        # decoder
        decoderstack = nn.ModuleList([])

        if use_ssm:
            for _ in range(num_layers):
                decoderstack.append(
                    FiLMMambaDecoderLayer( 
                        latent_dim,
                        num_heads,
                        dim_feedforward=ff_size,
                        dropout=dropout,
                        activation=activation,
                        batch_first=True,
                        rotary=self.rotary,
                    )
                )
        else:
            for _ in range(num_layers):
                decoderstack.append(
                    FiLMTransformerDecoderLayer( 
                        latent_dim,
                        num_heads,
                        dim_feedforward=ff_size,
                        dropout=dropout,
                        activation=activation,
                        batch_first=True,
                        rotary=self.rotary,
                    )
                )

        self.seqTransDecoder = DecoderLayerStack(decoderstack) 
        
        self.final_layer = nn.Linear(latent_dim, output_feats)

        self.embeddings = Tem_ID_Encoder(latent_dim=latent_dim, dropout=dropout, #  temporal encodings + identity encodings
                                        max_id_num=10) 
        # relative embedding
        self.relative_projection_layer = nn.Sequential( 
            nn.Linear(latent_dim * required_dancer_num, latent_dim * 2),
            nn.ReLU(),
            nn.Linear(latent_dim * 2, latent_dim * 2),
            nn.ReLU(),
            nn.Linear(latent_dim * 2, latent_dim * required_dancer_num)
        )
        
        # Position swapping embeddings
        self.swap_mode_embedding = nn.Sequential(
            nn.Linear(required_dancer_num*2, latent_dim), 
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim), 
            nn.GELU(),
        )

        # Trajectory modulation network
        self.traj_Modulation = nn.ModuleList([
            ConcatSquashLinear(nfeats, 128, 2),
            ConcatSquashLinear(128, 128, 2),
            ConcatSquashLinear(128, nfeats, 2),
        ])
        self.act = F.leaky_relu
        

    def guided_forward(self, x, cond_embed, times, guidance_weight):
        unc = self.forward(x, cond_embed, times, cond_drop_prob=1) 
        conditioned = self.forward(x, cond_embed, times, cond_drop_prob=0) 

        return unc + (conditioned - unc) * guidance_weight

    def _handle_swap_mode(self, cond_tokens: Tensor, t_tokens: Tensor,
                                 final_x_y_axes_order: Tensor, batch_size: int) -> Tensor:
        """Handle dancer position swapping."""
        if final_x_y_axes_order is None:
            final_x_y_axes_order = torch.arange(self.required_dancer_num).repeat(2)\
                        .unsqueeze(0).repeat(batch_size, 1)
        swap_emb = self.swap_mode_embedding(final_x_y_axes_order.to(t_tokens.dtype).to(t_tokens.device))
        return torch.cat([
            swap_emb.view(batch_size, 1, -1),
            cond_tokens,
            t_tokens
        ], dim=-2)

    def forward( 
        self, x: Tensor, cond_embed: Tensor, times: Tensor, cond_drop_prob: float = 0.0,
        final_x_y_axes_order: Optional[Tensor] = None
    ):
        batch_size, device = x.shape[0], x.device

        x = x.reshape(batch_size, -1, 151) 

        # project to latent space
        x = self.input_projection(x) 
        # add the positional embeddings of the input sequence to provide temporal information
        x = self.relative_projection_layer(x.reshape(batch_size,150,-1)).reshape(batch_size,-1,self.latent_dim)
        x = self.embeddings(x) # DPE

        # create music conditional embedding with conditional dropout
        keep_mask = prob_mask_like((batch_size,), 1 - cond_drop_prob, device=device) # torch.Size([2])
        keep_mask_embed = rearrange(keep_mask, "b -> b 1 1") # torch.Size([2, 1, 1])
        keep_mask_hidden = rearrange(keep_mask, "b -> b 1") # torch.Size([2, 1])

        c_bs,cond_seq_len,_ = cond_embed.shape
        if cond_seq_len % 2 == 1: 
            cond_embed = cond_embed[:,:-1,:].reshape(c_bs, cond_seq_len//2,-1)
        else:
            cond_embed = cond_embed.reshape(c_bs, cond_seq_len//2,-1)

        cond_tokens = self.cond_projection(cond_embed.float()) 
        # encode tokens
        cond_tokens = self.abs_pos_encoding(cond_tokens) 
        cond_tokens = self.cond_encoder(cond_tokens) 

        null_cond_embed = self.null_cond_embed.to(cond_tokens.dtype) 
        cond_tokens = torch.where(keep_mask_embed, cond_tokens, null_cond_embed) 

        mean_pooled_cond_tokens = cond_tokens.mean(dim=-2) 
        cond_hidden = self.non_attn_cond_projection(mean_pooled_cond_tokens) 

        # create the diffusion timestep embedding, add the extra music projection
        t_hidden = self.time_mlp(times) 

        # project to attention and FiLM conditioning
        t = self.to_time_cond(t_hidden) 
        t_tokens = self.to_time_tokens(t_hidden) 

        # FiLM conditioning 
        null_cond_hidden = self.null_cond_hidden.to(t.dtype) 
        cond_hidden = torch.where(keep_mask_hidden, cond_hidden, null_cond_hidden) 
        t += cond_hidden 

        

        # cross-attention conditioning 
        # Position swap handling
        c = self._handle_swap_mode(
            cond_tokens, t_tokens, final_x_y_axes_order, batch_size
        )
        cond_tokens = self.norm_cond(c) # layernorm 

        # Pass through the transformer decoder 
        # attending to the conditional embedding
        output = self.seqTransDecoder(x, cond_tokens, t) 

        output = self.final_layer(output)

        # # Trajectory modulation and footwork adjustment
        return self._apply_trajectory_modulation(output, batch_size)

    def _apply_trajectory_modulation(self, output: Tensor, batch_size: int) -> Tensor:
        """Apply trajectory modulation to generated motion."""
        # Reshape for trajectory processing
        output = output.view(batch_size, 150, -1, 151)
        
        # Calculate trajectory offsets
        traj_offset = output[:, 1:, :, [4,5]] - output[:, :-1, :, [4,5]]
        traj_offset = F.pad(traj_offset, (0,0,0,0,1,0), "constant", 0)

        # Apply modulation network
        modulated = output.clone()
        for i, layer in enumerate(self.traj_Modulation):
            modulated = layer(ctx=traj_offset, x=modulated)
            if i < len(self.traj_Modulation) - 1:
                modulated = self.act(modulated)

        # Blend modulated body parts
        return self._blend_modulated_output(output, modulated)

    def _blend_modulated_output(self, original: Tensor, modulated: Tensor) -> Tensor:
        """Blend modulated body parts with original output."""
        # Body part indices to modulate
        FOOTWORK_JOINTS = [1,2,3,4,5,7,8,10,11]
        
        # Extract and modify body components
        orig_body = original[..., 7:].view(*original.shape[:-1], 24, 6)
        mod_body = modulated[..., 7:].view(*modulated.shape[:-1], 24, 6)
        
        # Create blending mask
        mask = torch.zeros_like(orig_body, dtype=torch.bool)
        mask[:, :, :, FOOTWORK_JOINTS] = True
        
        # Apply blending
        blended_body = torch.where(mask, mod_body, orig_body)
        blended_body = blended_body.view(*original.shape[:-1], -1)
        
        # Combine with original root positions
        final = torch.cat([original[..., :7], blended_body], dim=-1)
        return final.view(original.shape[0], -1, 151)
