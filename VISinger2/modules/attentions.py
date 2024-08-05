import copy
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import modules.commons as commons


class LayerNorm(nn.Module):
  def __init__(self, channels, eps=1e-5):
    super().__init__()
    self.channels = channels
    self.eps = eps

    self.gamma = nn.Parameter(torch.ones(channels))
    self.beta = nn.Parameter(torch.zeros(channels))

  def forward(self, x):
    x = x.transpose(1, -1)
    x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
    return x.transpose(1, -1)


class Encoder(nn.Module):
  def __init__(self, hidden_channels, filter_channels, n_heads, n_layers, kernel_size=1, p_dropout=0., window_size=4, **kwargs):
    super().__init__()
    self.hidden_channels = hidden_channels
    self.filter_channels = filter_channels
    self.n_heads = n_heads
    self.n_layers = n_layers
    self.kernel_size = kernel_size
    self.p_dropout = p_dropout
    self.window_size = window_size

    self.drop = nn.Dropout(p_dropout)
    self.attn_layers = nn.ModuleList()
    self.norm_layers_1 = nn.ModuleList()
    self.ffn_layers = nn.ModuleList()
    self.norm_layers_2 = nn.ModuleList()
    for i in range(self.n_layers):
      self.attn_layers.append(MultiHeadAttention(hidden_channels, hidden_channels, n_heads, p_dropout=p_dropout, window_size=window_size))
      self.norm_layers_1.append(LayerNorm(hidden_channels))
      self.ffn_layers.append(FFN(hidden_channels, hidden_channels, filter_channels, kernel_size, p_dropout=p_dropout))
      self.norm_layers_2.append(LayerNorm(hidden_channels))

  def forward(self, x, x_mask):
    attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
    x = x * x_mask
    for i in range(self.n_layers):
      y = self.attn_layers[i](x, x, attn_mask)
      y = self.drop(y)
      x = self.norm_layers_1[i](x + y)

      y = self.ffn_layers[i](x, x_mask)
      y = self.drop(y)
      x = self.norm_layers_2[i](x + y)
    x = x * x_mask
    return x

class Decoder(nn.Module):
  def __init__(self, hidden_channels, filter_channels, n_heads, n_layers, kernel_size=1, p_dropout=0., proximal_bias=False, proximal_init=True, **kwargs):
    super().__init__()
    self.hidden_channels = hidden_channels
    self.filter_channels = filter_channels
    self.n_heads = n_heads
    self.n_layers = n_layers
    self.kernel_size = kernel_size
    self.p_dropout = p_dropout
    self.proximal_bias = proximal_bias
    self.proximal_init = proximal_init

    self.drop = nn.Dropout(p_dropout)
    self.self_attn_layers = nn.ModuleList()
    self.norm_layers_0 = nn.ModuleList()
    self.encdec_attn_layers = nn.ModuleList()
    self.norm_layers_1 = nn.ModuleList()
    self.ffn_layers = nn.ModuleList()
    self.norm_layers_2 = nn.ModuleList()
    for i in range(self.n_layers):
      self.self_attn_layers.append(MultiHeadAttention(hidden_channels, hidden_channels, n_heads, p_dropout=p_dropout, proximal_bias=proximal_bias, proximal_init=proximal_init))
      self.norm_layers_0.append(LayerNorm(hidden_channels))
      self.encdec_attn_layers.append(MultiHeadAttention(hidden_channels, hidden_channels, n_heads, p_dropout=p_dropout))
      self.norm_layers_1.append(LayerNorm(hidden_channels))
      self.ffn_layers.append(FFN(hidden_channels, hidden_channels, filter_channels, kernel_size, p_dropout=p_dropout, causal=True))
      self.norm_layers_2.append(LayerNorm(hidden_channels))

  def forward(self, x, x_mask, h, h_mask):
    """
    x: decoder input
    h: encoder output
    """
    self_attn_mask = commons.subsequent_mask(x_mask.size(2)).to(device=x.device, dtype=x.dtype)
    encdec_attn_mask = h_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
    x = x * x_mask
    for i in range(self.n_layers):
      y = self.self_attn_layers[i](x, x, self_attn_mask)
      y = self.drop(y)
      x = self.norm_layers_0[i](x + y)

      y = self.encdec_attn_layers[i](x, h, encdec_attn_mask)
      y = self.drop(y)
      x = self.norm_layers_1[i](x + y)
      
      y = self.ffn_layers[i](x, x_mask)
      y = self.drop(y)
      x = self.norm_layers_2[i](x + y)
    x = x * x_mask
    return x

class FFT(nn.Module):
  def __init__(self, hidden_channels, filter_channels, n_heads, n_layers=1, kernel_size=1, p_dropout=0., proximal_bias=False, proximal_init=True, **kwargs):
    super().__init__()
    self.hidden_channels = hidden_channels
    self.filter_channels = filter_channels
    self.n_heads = n_heads
    self.n_layers = n_layers
    self.kernel_size = kernel_size
    self.p_dropout = p_dropout
    self.proximal_bias = proximal_bias
    self.proximal_init = proximal_init

    self.drop = nn.Dropout(p_dropout)
    self.self_attn_layers = nn.ModuleList()
    self.norm_layers_0 = nn.ModuleList()
    self.ffn_layers = nn.ModuleList()
    self.norm_layers_1 = nn.ModuleList()
    for i in range(self.n_layers):
      self.self_attn_layers.append(MultiHeadAttention(hidden_channels, hidden_channels, n_heads, p_dropout=p_dropout, proximal_bias=proximal_bias, proximal_init=proximal_init))
      self.norm_layers_0.append(LayerNorm(hidden_channels))
      self.ffn_layers.append(FFN(hidden_channels, hidden_channels, filter_channels, kernel_size, p_dropout=p_dropout, causal=True))
      self.norm_layers_1.append(LayerNorm(hidden_channels))

  def forward(self, x, x_mask):
    """
    x: decoder input
    h: encoder output
    """
    self_attn_mask = commons.subsequent_mask(x_mask.size(2)).to(device=x.device, dtype=x.dtype)
    x = x * x_mask
    for i in range(self.n_layers):
      y = self.self_attn_layers[i](x, x, self_attn_mask)
      y = self.drop(y)
      x = self.norm_layers_0[i](x + y)
      
      y = self.ffn_layers[i](x, x_mask)
      y = self.drop(y)
      x = self.norm_layers_1[i](x + y) 
    x = x * x_mask
    return x


class FFNs(nn.Module):
  def __init__(self, hidden_channels, filter_channels, n_heads, n_layers=1, kernel_size=1, p_dropout=0., proximal_bias=False, proximal_init=True, **kwargs):
    super().__init__()
    self.hidden_channels = hidden_channels
    self.filter_channels = filter_channels
    self.n_heads = n_heads
    self.n_layers = n_layers
    self.kernel_size = kernel_size
    self.p_dropout = p_dropout
    self.proximal_bias = proximal_bias
    self.proximal_init = proximal_init

    self.drop = nn.Dropout(p_dropout)
    #self.self_attn_layers = nn.ModuleList()
    #self.norm_layers_0 = nn.ModuleList()
    self.ffn_layers = nn.ModuleList()
    self.norm_layers_1 = nn.ModuleList()
    for i in range(self.n_layers):
      #self.self_attn_layers.append(MultiHeadAttention(hidden_channels, hidden_channels, n_heads, p_dropout=p_dropout, proximal_bias=proximal_bias, proximal_init=proximal_init))
      #self.norm_layers_0.append(LayerNorm(hidden_channels))
      self.ffn_layers.append(FFN(hidden_channels, hidden_channels, filter_channels, kernel_size, p_dropout=p_dropout, causal=True))
      self.norm_layers_1.append(LayerNorm(hidden_channels))

  def forward(self, x, x_mask):
    """
    x: decoder input
    h: encoder output
    """
    self_attn_mask = commons.subsequent_mask(x_mask.size(2)).to(device=x.device, dtype=x.dtype)
    x = x * x_mask
    for i in range(self.n_layers):
      #y = self.self_attn_layers[i](x, x, self_attn_mask)
      #y = self.drop(y)
      #x = self.norm_layers_0[i](x + y)
      
      y = self.ffn_layers[i](x, x_mask)
      y = self.drop(y)
      x = self.norm_layers_1[i](x + y) 
    x = x * x_mask
    return x

class MultiHeadAttention(nn.Module):
  def __init__(self, channels, out_channels, n_heads, p_dropout=0., window_size=None, heads_share=True, block_length=None, proximal_bias=False, proximal_init=False):
    super().__init__()
    assert channels % n_heads == 0

    self.channels = channels
    self.out_channels = out_channels
    self.n_heads = n_heads
    self.p_dropout = p_dropout
    self.window_size = window_size
    self.heads_share = heads_share
    self.block_length = block_length
    self.proximal_bias = proximal_bias
    self.proximal_init = proximal_init
    self.attn = None

    self.k_channels = channels // n_heads
    self.conv_q = nn.Conv1d(channels, channels, 1)
    self.conv_k = nn.Conv1d(channels, channels, 1)
    self.conv_v = nn.Conv1d(channels, channels, 1)
    self.conv_o = nn.Conv1d(channels, out_channels, 1)
    self.drop = nn.Dropout(p_dropout)

    if window_size is not None:
      n_heads_rel = 1 if heads_share else n_heads
      rel_stddev = self.k_channels**-0.5
      self.emb_rel_k = nn.Parameter(torch.randn(n_heads_rel, window_size * 2 + 1, self.k_channels) * rel_stddev)
      self.emb_rel_v = nn.Parameter(torch.randn(n_heads_rel, window_size * 2 + 1, self.k_channels) * rel_stddev)

    nn.init.xavier_uniform_(self.conv_q.weight)
    nn.init.xavier_uniform_(self.conv_k.weight)
    nn.init.xavier_uniform_(self.conv_v.weight)
    if proximal_init:
      with torch.no_grad():
        self.conv_k.weight.copy_(self.conv_q.weight)
        self.conv_k.bias.copy_(self.conv_q.bias)
      
  def forward(self, x, c, attn_mask=None):
    q = self.conv_q(x)
    k = self.conv_k(c)
    v = self.conv_v(c)
    
    x, self.attn = self.attention(q, k, v, mask=attn_mask)

    x = self.conv_o(x)
    return x

  def attention(self, query, key, value, mask=None):
    # reshape [b, d, t] -> [b, n_h, t, d_k]
    b, d, t_s, t_t = (*key.size(), query.size(2))
    query = query.view(b, self.n_heads, self.k_channels, t_t).transpose(2, 3)
    key = key.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)
    value = value.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)

    scores = torch.matmul(query / math.sqrt(self.k_channels), key.transpose(-2, -1))
    if self.window_size is not None:
      assert t_s == t_t, "Relative attention is only available for self-attention."
      key_relative_embeddings = self._get_relative_embeddings(self.emb_rel_k, t_s)
      rel_logits = self._matmul_with_relative_keys(query /math.sqrt(self.k_channels), key_relative_embeddings)
      scores_local = self._relative_position_to_absolute_position(rel_logits)
      scores = scores + scores_local
    if self.proximal_bias:
      assert t_s == t_t, "Proximal bias is only available for self-attention."
      scores = scores + self._attention_bias_proximal(t_s).to(device=scores.device, dtype=scores.dtype)
    if mask is not None:
      scores = scores.masked_fill(mask == 0, -1e4)
      if self.block_length is not None:
        assert t_s == t_t, "Local attention is only available for self-attention."
        block_mask = torch.ones_like(scores).triu(-self.block_length).tril(self.block_length)
        scores = scores.masked_fill(block_mask == 0, -1e4)
    p_attn = F.softmax(scores, dim=-1) # [b, n_h, t_t, t_s]
    p_attn = self.drop(p_attn)
    output = torch.matmul(p_attn, value)
    if self.window_size is not None:
      relative_weights = self._absolute_position_to_relative_position(p_attn)
      value_relative_embeddings = self._get_relative_embeddings(self.emb_rel_v, t_s)
      output = output + self._matmul_with_relative_values(relative_weights, value_relative_embeddings)
    output = output.transpose(2, 3).contiguous().view(b, d, t_t) # [b, n_h, t_t, d_k] -> [b, d, t_t]
    return output, p_attn

  def _matmul_with_relative_values(self, x, y):
    """
    x: [b, h, l, m]
    y: [h or 1, m, d]
    ret: [b, h, l, d]
    """
    ret = torch.matmul(x, y.unsqueeze(0))
    return ret

  def _matmul_with_relative_keys(self, x, y):
    """
    x: [b, h, l, d]
    y: [h or 1, m, d]
    ret: [b, h, l, m]
    """
    ret = torch.matmul(x, y.unsqueeze(0).transpose(-2, -1))
    return ret

  def _get_relative_embeddings(self, relative_embeddings, length):
    max_relative_position = 2 * self.window_size + 1
    # Pad first before slice to avoid using cond ops.
    pad_length = max(length - (self.window_size + 1), 0)
    slice_start_position = max((self.window_size + 1) - length, 0)
    slice_end_position = slice_start_position + 2 * length - 1
    if pad_length > 0:
      padded_relative_embeddings = F.pad(
          relative_embeddings,
          commons.convert_pad_shape([[0, 0], [pad_length, pad_length], [0, 0]]))
    else:
      padded_relative_embeddings = relative_embeddings
    used_relative_embeddings = padded_relative_embeddings[:,slice_start_position:slice_end_position]
    return used_relative_embeddings

  def _relative_position_to_absolute_position(self, x):
    """
    x: [b, h, l, 2*l-1]
    ret: [b, h, l, l]
    """
    batch, heads, length, _ = x.size()
    # Concat columns of pad to shift from relative to absolute indexing.
    x = F.pad(x, commons.convert_pad_shape([[0,0],[0,0],[0,0],[0,1]]))

    # Concat extra elements so to add up to shape (len+1, 2*len-1).
    x_flat = x.view([batch, heads, length * 2 * length])
    x_flat = F.pad(x_flat, commons.convert_pad_shape([[0,0],[0,0],[0,length-1]]))

    # Reshape and slice out the padded elements.
    x_final = x_flat.view([batch, heads, length+1, 2*length-1])[:, :, :length, length-1:]
    return x_final

  def _absolute_position_to_relative_position(self, x):
    """
    x: [b, h, l, l]
    ret: [b, h, l, 2*l-1]
    """
    batch, heads, length, _ = x.size()
    # padd along column
    x = F.pad(x, commons.convert_pad_shape([[0, 0], [0, 0], [0, 0], [0, length-1]]))
    x_flat = x.view([batch, heads, length**2 + length*(length -1)])
    # add 0's in the beginning that will skew the elements after reshape
    x_flat = F.pad(x_flat, commons.convert_pad_shape([[0, 0], [0, 0], [length, 0]]))
    x_final = x_flat.view([batch, heads, length, 2*length])[:,:,:,1:]
    return x_final

  def _attention_bias_proximal(self, length):
    """Bias for self-attention to encourage attention to close positions.
    Args:
      length: an integer scalar.
    Returns:
      a Tensor with shape [1, 1, length, length]
    """
    r = torch.arange(length, dtype=torch.float32)
    diff = torch.unsqueeze(r, 0) - torch.unsqueeze(r, 1)
    return torch.unsqueeze(torch.unsqueeze(-torch.log1p(torch.abs(diff)), 0), 0)


class FFN(nn.Module):
  def __init__(self, in_channels, out_channels, filter_channels, kernel_size, p_dropout=0., activation=None, causal=False):
    super().__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.filter_channels = filter_channels
    self.kernel_size = kernel_size
    self.p_dropout = p_dropout
    self.activation = activation
    self.causal = causal

    if causal:
      self.padding = self._causal_padding
    else:
      self.padding = self._same_padding

    self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size)
    self.conv_2 = nn.Conv1d(filter_channels, out_channels, kernel_size)
    self.drop = nn.Dropout(p_dropout)

  def forward(self, x, x_mask):
    x = self.conv_1(self.padding(x * x_mask))
    if self.activation == "gelu":
      x = x * torch.sigmoid(1.702 * x)
    else:
      x = torch.relu(x)
    x = self.drop(x)
    x = self.conv_2(self.padding(x * x_mask))
    return x * x_mask
  
  def _causal_padding(self, x):
    if self.kernel_size == 1:
      return x
    pad_l = self.kernel_size - 1
    pad_r = 0
    padding = [[0, 0], [0, 0], [pad_l, pad_r]]
    x = F.pad(x, commons.convert_pad_shape(padding))
    return x

  def _same_padding(self, x):
    if self.kernel_size == 1:
      return x
    pad_l = (self.kernel_size - 1) // 2
    pad_r = self.kernel_size // 2
    padding = [[0, 0], [0, 0], [pad_l, pad_r]]
    x = F.pad(x, commons.convert_pad_shape(padding))
    return x

# import copy
# import math
# import numpy as np
# import torch
# from torch import nn
# from torch.nn import functional as F

# import modules.commons as commons


# class LayerNorm(nn.Module):
#   def __init__(self, channels, eps=1e-5):
#     super().__init__()
#     self.channels = channels
#     self.eps = eps

#     self.gamma = nn.Parameter(torch.ones(channels))
#     self.beta = nn.Parameter(torch.zeros(channels))

#   def forward(self, x):
#     x = x.transpose(1, -1)
#     x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
#     return x.transpose(1, -1)


# class Encoder(nn.Module):
#   def __init__(self, hidden_channels, filter_channels, n_heads, n_layers, kernel_size=1, p_dropout=0., window_size=4, **kwargs):
#     super().__init__()
#     self.hidden_channels = hidden_channels
#     self.filter_channels = filter_channels
#     self.n_heads = n_heads
#     self.n_layers = n_layers
#     self.kernel_size = kernel_size
#     self.p_dropout = p_dropout
#     self.window_size = window_size

#     self.drop = nn.Dropout(p_dropout)
#     self.attn_layers = nn.ModuleList()
#     self.norm_layers_1 = nn.ModuleList()
#     self.ffn_layers = nn.ModuleList()
#     self.norm_layers_2 = nn.ModuleList()
#     for i in range(self.n_layers):
#       self.attn_layers.append(MultiHeadAttention(hidden_channels, hidden_channels, n_heads, p_dropout=p_dropout, window_size=window_size))
#       self.norm_layers_1.append(LayerNorm(hidden_channels))
#       self.ffn_layers.append(FFN(hidden_channels, hidden_channels, filter_channels, kernel_size, p_dropout=p_dropout))
#       self.norm_layers_2.append(LayerNorm(hidden_channels))

#   def forward(self, x, x_mask):
#     attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
#     x = x * x_mask
#     for i in range(self.n_layers):
#       y = self.attn_layers[i](x, x, attn_mask)
#       y = self.drop(y)
#       x = self.norm_layers_1[i](x + y)

#       y = self.ffn_layers[i](x, x_mask)
#       y = self.drop(y)
#       x = self.norm_layers_2[i](x + y)
#     x = x * x_mask
#     return x

# class Decoder(nn.Module):
#   def __init__(self, hidden_channels, filter_channels, n_heads, n_layers, kernel_size=1, p_dropout=0., proximal_bias=False, proximal_init=True, **kwargs):
#     super().__init__()
#     self.hidden_channels = hidden_channels
#     self.filter_channels = filter_channels
#     self.n_heads = n_heads
#     self.n_layers = n_layers
#     self.kernel_size = kernel_size
#     self.p_dropout = p_dropout
#     self.proximal_bias = proximal_bias
#     self.proximal_init = proximal_init

#     self.drop = nn.Dropout(p_dropout)
#     self.self_attn_layers = nn.ModuleList()
#     self.norm_layers_0 = nn.ModuleList()
#     self.encdec_attn_layers = nn.ModuleList()
#     self.norm_layers_1 = nn.ModuleList()
#     self.ffn_layers = nn.ModuleList()
#     self.norm_layers_2 = nn.ModuleList()
#     for i in range(self.n_layers):
#       self.self_attn_layers.append(MultiHeadAttention(hidden_channels, hidden_channels, n_heads, p_dropout=p_dropout, proximal_bias=proximal_bias, proximal_init=proximal_init))
#       self.norm_layers_0.append(LayerNorm(hidden_channels))
#       self.encdec_attn_layers.append(MultiHeadAttention(hidden_channels, hidden_channels, n_heads, p_dropout=p_dropout))
#       self.norm_layers_1.append(LayerNorm(hidden_channels))
#       self.ffn_layers.append(FFN(hidden_channels, hidden_channels, filter_channels, kernel_size, p_dropout=p_dropout, causal=True))
#       self.norm_layers_2.append(LayerNorm(hidden_channels))

#   def forward(self, x, x_mask, h, h_mask):
#     """
#     x: decoder input
#     h: encoder output
#     """
#     self_attn_mask = commons.subsequent_mask(x_mask.size(2)).to(device=x.device, dtype=x.dtype)
#     encdec_attn_mask = h_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
#     x = x * x_mask
#     for i in range(self.n_layers):
#       y = self.self_attn_layers[i](x, x, self_attn_mask)
#       y = self.drop(y)
#       x = self.norm_layers_0[i](x + y)

#       y = self.encdec_attn_layers[i](x, h, encdec_attn_mask)
#       y = self.drop(y)
#       x = self.norm_layers_1[i](x + y)
      
#       y = self.ffn_layers[i](x, x_mask)
#       y = self.drop(y)
#       x = self.norm_layers_2[i](x + y)
#     x = x * x_mask
#     return x

# class ConformerBlock(nn.Module):
#   def __init__(self, hidden_channels, filter_channels, n_heads, kernel_size=1, p_dropout=0.):
#     super().__init__()
#     self.hidden_channels = hidden_channels
#     self.filter_channels = filter_channels
#     self.n_heads = n_heads
#     self.kernel_size = kernel_size
#     self.p_dropout = p_dropout

#     self.layer_norm1 = LayerNorm(hidden_channels)
#     self.attention = MultiHeadAttention(hidden_channels, hidden_channels, n_heads, p_dropout=p_dropout)
#     self.layer_norm2 = LayerNorm(hidden_channels)
#     self.ffn1 = FFN(hidden_channels, hidden_channels, filter_channels, kernel_size, p_dropout=p_dropout)
#     self.conv_module = nn.Sequential(
#       nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding=kernel_size//2),
#       nn.GLU(dim=1),
#       nn.Conv1d(hidden_channels, hidden_channels, 1)
#     )
#     self.layer_norm3 = LayerNorm(hidden_channels)
#     self.ffn2 = FFN(hidden_channels, hidden_channels, filter_channels, kernel_size, p_dropout=p_dropout)
#     self.drop = nn.Dropout(p_dropout)

#   def forward(self, x, x_mask):
#     residual = x
#     x = self.layer_norm1(x)
#     x = self.attention(x, x, x_mask.unsqueeze(2) * x_mask.unsqueeze(-1))
#     x = self.drop(x) + residual

#     residual = x
#     x = self.layer_norm2(x)
#     x = self.ffn1(x, x_mask)
#     x = self.drop(x) + residual

#     residual = x
#     x = self.conv_module(x.transpose(1, 2)).transpose(1, 2)
#     x = self.drop(x) + residual

#     residual = x
#     x = self.layer_norm3(x)
#     x = self.ffn2(x, x_mask)
#     x = self.drop(x) + residual

#     return x * x_mask

# class FFT(nn.Module):
# #Conformer
#   def __init__(self, hidden_channels, filter_channels, n_heads, n_layers=1, kernel_size=1, p_dropout=0.):
#     super().__init__()
#     self.hidden_channels = hidden_channels
#     self.filter_channels = filter_channels
#     self.n_heads = n_heads
#     self.n_layers = n_layers
#     self.kernel_size = kernel_size
#     self.p_dropout = p_dropout

#     self.layers = nn.ModuleList()
#     for _ in range(self.n_layers):
#       self.layers.append(ConformerBlock(hidden_channels, filter_channels, n_heads, kernel_size, p_dropout))

#   def forward(self, x, x_mask):
#     for layer in self.layers:
#       x = layer(x, x_mask)
#     return x

# class FFNs(nn.Module):
#   def __init__(self, hidden_channels, filter_channels, n_heads, n_layers=1, kernel_size=1, p_dropout=0., proximal_bias=False, proximal_init=True, **kwargs):
#     super().__init__()
#     self.hidden_channels = hidden_channels
#     self.filter_channels = filter_channels
#     self.n_heads = n_heads
#     self.n_layers = n_layers
#     self.kernel_size = kernel_size
#     self.p_dropout = p_dropout
#     self.proximal_bias = proximal_bias
#     self.proximal_init = proximal_init

#     self.drop = nn.Dropout(p_dropout)
#     self.ffn_layers = nn.ModuleList()
#     self.norm_layers_1 = nn.ModuleList()
#     for i in range(self.n_layers):
#       self.ffn_layers.append(FFN(hidden_channels, hidden_channels, filter_channels, kernel_size, p_dropout=p_dropout, causal=True))
#       self.norm_layers_1.append(LayerNorm(hidden_channels))

#   def forward(self, x, x_mask):
#     self_attn_mask = commons.subsequent_mask(x_mask.size(2)).to(device=x.device, dtype=x.dtype)
#     x = x * x_mask
#     for i in range(self.n_layers):
#       y = self.ffn_layers[i](x, x_mask)
#       y = self.drop(y)
#       x = self.norm_layers_1[i](x + y) 
#     x = x * x_mask
#     return x

# class MultiHeadAttention(nn.Module):
#   def __init__(self, channels, out_channels, n_heads, p_dropout=0., window_size=None, heads_share=True, block_length=None, proximal_bias=False, proximal_init=False):
#     super().__init__()
#     assert channels % n_heads == 0
#     self.channels = channels
#     self.out_channels = out_channels
#     self.n_heads = n_heads
#     self.p_dropout = p_dropout
#     self.window_size = window_size
#     self.heads_share = heads_share
#     self.block_length = block_length
#     self.proximal_bias = proximal_bias
#     self.proximal_init = proximal_init

#     self.k_channels = channels // n_heads
#     self.conv_rel_pos_enc = None
#     if window_size is not None:
#       self.conv_rel_pos_enc = nn.Conv2d(n_heads, n_heads, (1, 2*window_size+1), padding=(0, window_size), bias=False)

#     self.conv_q = nn.Conv1d(channels, channels, 1)
#     self.conv_k = nn.Conv1d(channels, channels, 1)
#     self.conv_v = nn.Conv1d(channels, channels, 1)
#     self.conv_o = nn.Conv1d(channels, out_channels, 1)

#     if proximal_bias:
#       self.bias = torch.tril(torch.ones(block_length, block_length)).view(1, 1, block_length, block_length)
#     else:
#       self.bias = None

#     if proximal_init:
#       self.conv_k.weight.data.copy_(torch.eye(channels).unsqueeze(-1))
#       self.conv_q.weight.data.copy_(torch.eye(channels).unsqueeze(-1))
#       self.conv_v.weight.data.copy_(torch.eye(channels).unsqueeze(-1))

#     self.drop = nn.Dropout(p_dropout)

#   def forward(self, x, c, attn_mask=None):
#     q = self.conv_q(x)
#     k = self.conv_k(c)
#     v = self.conv_v(c)

#     q = q.view(x.size(0), self.n_heads, self.k_channels, x.size(2)).transpose(2, 3)
#     k = k.view(x.size(0), self.n_heads, self.k_channels, c.size(2)).transpose(2, 3)
#     v = v.view(x.size(0), self.n_heads, self.k_channels, c.size(2)).transpose(2, 3)

#     scores = torch.matmul(q / math.sqrt(self.k_channels), k.transpose(-2, -1))

#     if self.conv_rel_pos_enc is not None:
#       pos_enc = self.conv_rel_pos_enc(k.unsqueeze(-2) - q.unsqueeze(-3))
#       scores = scores + pos_enc

#     if self.bias is not None:
#       scores = scores + self.bias[:, :, :scores.size(2), :scores.size(3)]

#     if attn_mask is not None:
#       scores = scores.masked_fill(attn_mask == 0, -1e9)
#     p_attn = F.softmax(scores, dim=-1)
#     p_attn = self.drop(p_attn)

#     output = torch.matmul(p_attn, v)
#     output = output.transpose(2, 3).contiguous().view(x.size(0), self.channels, x.size(2))
#     output = self.conv_o(output)
#     return output

# class FFN(nn.Module):
#   def __init__(self, in_channels, out_channels, filter_channels, kernel_size, p_dropout=0., causal=False):
#     super().__init__()
#     self.in_channels = in_channels
#     self.out_channels = out_channels
#     self.filter_channels = filter_channels
#     self.kernel_size = kernel_size
#     self.p_dropout = p_dropout
#     self.causal = causal

#     self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size//2)
#     self.conv_2 = nn.Conv1d(filter_channels, out_channels, kernel_size, padding=kernel_size//2)
#     if causal:
#       self.conv_1 = nn.utils.weight_norm(self.conv_1)
#       self.conv_2 = nn.utils.weight_norm(self.conv_2)
#     self.drop = nn.Dropout(p_dropout)

#   def forward(self, x, x_mask):
#     x = x * x_mask
#     x = self.conv_1(x.transpose(1, 2)).transpose(1, 2)
#     x = F.gelu(x)
#     x = self.conv_2(x.transpose(1, 2)).transpose(1, 2)
#     x = self.drop(x)
#     x = x * x_mask
#     return x
