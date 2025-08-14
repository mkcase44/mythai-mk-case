import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, dim)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class StrokeHistoryEncoder(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=128, num_layers=2, num_heads=4, dropout=0.1):
        super().__init__()
        # Positional embedding girişte uygulanacak
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_emb = PositionalEncoding(hidden_dim)

        self.lstm = nn.LSTM(hidden_dim, hidden_dim // 2, num_layers,
                            batch_first=True, bidirectional=True, dropout=dropout)

        self.lstm_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads=num_heads, batch_first=True, dropout=dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, stroke_history, stroke_mask):
        """
        stroke_history: (B, H_len, input_dim)
        stroke_mask:    (B, H_len)  (1 for valid, 0 for padding)
        """
        lengths = stroke_mask.sum(dim=1).cpu()

        x = self.input_proj(stroke_history)
        x = self.pos_emb(x)

        # LSTM
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True,
                                                  total_length=stroke_history.size(1))
        out = self.lstm_proj(out)
        out = out * stroke_mask.unsqueeze(-1)

        out = self.norm1(out)
        out = self.dropout(out)

        # Self-attention
        attn_out, _ = self.attn(out, out, out, key_padding_mask=(stroke_mask == 0))
        out = out + self.dropout(attn_out)  # residual
        out = self.norm2(out)

        return out

