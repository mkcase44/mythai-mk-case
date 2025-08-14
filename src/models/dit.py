import math
import torch
import torch.nn as nn
from diffusers.models.attention import BasicTransformerBlock


class SinusoidalTimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding for Transformer models."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer models."""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return x
    
class DiffusionTransformer(nn.Module):
    def __init__(self, 
                 d_model=256, 
                 nhead=8, 
                 num_encoder_layers=6,
                 dim_feedforward=1024,
                 input_feats=5,
                 output_feats=5,
                 pen_state_feats=3,
                 max_seq_length=250,
                 pen_condition=True):
        super().__init__()

        self.pen_condition = pen_condition

        self.input_projection = nn.Linear(input_feats, d_model)
        self.timestep_embedding = SinusoidalTimestepEmbedding(d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_seq_length)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        self.output_projection = nn.Linear(d_model, output_feats)

        if not self.pen_condition:
            self.pen_state_projection = nn.Linear(d_model, pen_state_feats)

    def forward(self, sample, timestep, **kwargs):
        padding_mask = (sample.abs().sum(dim=-1) == 0)

        x_proj = self.input_projection(sample)
        t_emb = self.timestep_embedding(timestep)
        x_with_pos = self.pos_encoder(x_proj)
        x_t = x_with_pos + t_emb.unsqueeze(1)

        transformer_output = self.transformer_encoder(
            x_t,
            src_key_padding_mask=padding_mask
            )
        
        predicted_noise = self.output_projection(transformer_output)
        if not self.pen_condition:
            pen_state_out = self.pen_state_projection(transformer_output)
        else:
            pen_state_out = None

        return predicted_noise, pen_state_out

class DiffusersBlockCrossAttentionTransformer(nn.Module):
    def __init__(self,
                 input_feats=5,
                 output_feats=5,
                 pen_state_feats=2,
                 hidden_dim=128,
                 num_layers=4,
                 num_heads=8,
                 pen_condition=True):
        super().__init__()
        
        self.pen_condition = pen_condition

        self.input_proj = nn.Linear(input_feats, hidden_dim)

        self.blocks = nn.ModuleList([
            BasicTransformerBlock(hidden_dim, num_heads, hidden_dim*4, activation_fn="gelu", attention_bias=False, only_cross_attention=False)
            for _ in range(num_layers)
        ])

        self.output_projection = nn.Linear(hidden_dim, output_feats)

        if not self.pen_condition:
            self.pen_state_projection = nn.Linear(hidden_dim, pen_state_feats)

    def forward(self, sample, timestep, context, **kwargs):
        """
        sample: (B, T, hidden_dim)
        timestep: (B,)
        context: (B, H_len, hidden_dim)
        context_mask: (B, H_len)
        """
        hidden_states = self.input_proj(sample)

        for block in self.blocks:
            hidden_states = block(
                hidden_states, 
                timestep=timestep,
                encoder_hidden_states=context,
                )

        predicted_noise = self.output_projection(hidden_states)

        if not self.pen_condition:
            pen_state_out = self.pen_state_projection(hidden_states)
        else:
            pen_state_out = None

        return predicted_noise, pen_state_out
