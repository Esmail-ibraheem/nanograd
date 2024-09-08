import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.helpers import fetch

from tinygrad import Tensor

class TransformerBlock:
  def __init__(self, embed_dim, num_heads, ff_dim, prenorm=False, act=lambda x: x.relu(), dropout=0.1):
    assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

    self.num_heads = num_heads
    self.head_size = embed_dim // num_heads
    self.prenorm, self.act = prenorm, act
    self.dropout = dropout

    self.query = (Tensor.scaled_uniform(embed_dim, embed_dim), Tensor.zeros(embed_dim))
    self.key = (Tensor.scaled_uniform(embed_dim, embed_dim), Tensor.zeros(embed_dim))
    self.value = (Tensor.scaled_uniform(embed_dim, embed_dim), Tensor.zeros(embed_dim))

    self.out = (Tensor.scaled_uniform(embed_dim, embed_dim), Tensor.zeros(embed_dim))

    self.ff1 = (Tensor.scaled_uniform(embed_dim, ff_dim), Tensor.zeros(ff_dim))
    self.ff2 = (Tensor.scaled_uniform(ff_dim, embed_dim), Tensor.zeros(embed_dim))

    self.ln1 = (Tensor.ones(embed_dim), Tensor.zeros(embed_dim))
    self.ln2 = (Tensor.ones(embed_dim), Tensor.zeros(embed_dim))

  def attn(self, x):
    
    query, key, value = [x.linear(*y).reshape(shape=(x.shape[0], -1, self.num_heads, self.head_size)).transpose(1,2) for y in [self.query, self.key, self.value]]
    attention = Tensor.scaled_dot_product_attention(query, key, value).transpose(1,2)
    return attention.reshape(shape=(x.shape[0], -1, self.num_heads * self.head_size)).linear(*self.out)

  def __call__(self, x):
    if self.prenorm:
      x = x + self.attn(x.layernorm().linear(*self.ln1)).dropout(self.dropout)
      x = x + self.act(x.layernorm().linear(*self.ln2).linear(*self.ff1)).linear(*self.ff2).dropout(self.dropout)
    else:
      x = x + self.attn(x).dropout(self.dropout)
      x = x.layernorm().linear(*self.ln1)
      x = x + self.act(x.linear(*self.ff1)).linear(*self.ff2).dropout(self.dropout)
      x = x.layernorm().linear(*self.ln2)
    return x

class Transformer:
  def __init__(self, syms, maxlen, layers, embed_dim, num_heads, ff_dim):
    self.maxlen, self.syms = maxlen, syms
    self.embed = Tensor.scaled_uniform(maxlen+syms, embed_dim, requires_grad=False)
    self.tbs = [TransformerBlock(embed_dim, num_heads, ff_dim) for _ in range(layers)]
    self.final = Tensor.scaled_uniform(embed_dim, syms)

  def forward(self, x):
    bs = x.shape[0]

    maxlen_eye = Tensor.eye(x.shape[1])
    maxlen_eye = maxlen_eye.unsqueeze(0).expand([bs, *maxlen_eye.shape])

    onehot_feat = x.one_hot(self.syms)

    onehot = maxlen_eye.cat(onehot_feat, dim=2).flatten(end_dim=1)

    x = onehot.dot(self.embed).reshape((bs, x.shape[1], -1))
    x = x.sequential(self.tbs)
    x = x.reshape((-1, x.shape[-1])).dot(self.final).log_softmax()
    return x.reshape((bs, -1, x.shape[-1]))


class ViT:
  def __init__(self, layers=12, embed_dim=192, num_heads=3):
    self.embedding = (Tensor.uniform(embed_dim, 3, 16, 16), Tensor.zeros(embed_dim))
    self.embed_dim = embed_dim
    self.cls = Tensor.ones(1, 1, embed_dim)
    self.pos_embedding = Tensor.ones(1, 197, embed_dim)
    self.tbs = [
      TransformerBlock(embed_dim=embed_dim, num_heads=num_heads, ff_dim=embed_dim*4,
        prenorm=True, act=lambda x: x.gelu())
      for i in range(layers)]
    self.encoder_norm = (Tensor.uniform(embed_dim), Tensor.zeros(embed_dim))
    self.head = (Tensor.uniform(embed_dim, 1000), Tensor.zeros(1000))

  def patch_embed(self, x):
    x = x.conv2d(*self.embedding, stride=16)
    x = x.reshape(shape=(x.shape[0], x.shape[1], -1)).permute(order=(0,2,1))
    return x

  def forward(self, x):
    ce = self.cls.add(Tensor.zeros(x.shape[0],1,1))
    pe = self.patch_embed(x)
    x = ce.cat(pe, dim=1)
    x = x.add(self.pos_embedding).sequential(self.tbs)
    x = x.layernorm().linear(*self.encoder_norm)
    return x[:, 0].linear(*self.head)

  def load_from_pretrained(m):
    
    if m.embed_dim == 192:
      url = "https://storage.googleapis.com/vit_models/augreg/Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz"
    elif m.embed_dim == 768:
      url = "https://storage.googleapis.com/vit_models/augreg/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz"
    else:
      raise Exception("no pretrained weights for configuration")
    dat = np.load(fetch(url))

    
    

    m.embedding[0].assign(np.transpose(dat['embedding/kernel'], (3,2,0,1)))
    m.embedding[1].assign(dat['embedding/bias'])

    m.cls.assign(dat['cls'])

    m.head[0].assign(dat['head/kernel'])
    m.head[1].assign(dat['head/bias'])

    m.pos_embedding.assign(dat['Transformer/posembed_input/pos_embedding'])
    m.encoder_norm[0].assign(dat['Transformer/encoder_norm/scale'])
    m.encoder_norm[1].assign(dat['Transformer/encoder_norm/bias'])

    for i in range(12):
      m.tbs[i].query[0].assign(dat[f'Transformer/encoderblock_{i}/MultiHeadDotProductAttention_1/query/kernel'].reshape(m.embed_dim, m.embed_dim))
      m.tbs[i].query[1].assign(dat[f'Transformer/encoderblock_{i}/MultiHeadDotProductAttention_1/query/bias'].reshape(m.embed_dim))
      m.tbs[i].key[0].assign(dat[f'Transformer/encoderblock_{i}/MultiHeadDotProductAttention_1/key/kernel'].reshape(m.embed_dim, m.embed_dim))
      m.tbs[i].key[1].assign(dat[f'Transformer/encoderblock_{i}/MultiHeadDotProductAttention_1/key/bias'].reshape(m.embed_dim))
      m.tbs[i].value[0].assign(dat[f'Transformer/encoderblock_{i}/MultiHeadDotProductAttention_1/value/kernel'].reshape(m.embed_dim, m.embed_dim))
      m.tbs[i].value[1].assign(dat[f'Transformer/encoderblock_{i}/MultiHeadDotProductAttention_1/value/bias'].reshape(m.embed_dim))
      m.tbs[i].out[0].assign(dat[f'Transformer/encoderblock_{i}/MultiHeadDotProductAttention_1/out/kernel'].reshape(m.embed_dim, m.embed_dim))
      m.tbs[i].out[1].assign(dat[f'Transformer/encoderblock_{i}/MultiHeadDotProductAttention_1/out/bias'].reshape(m.embed_dim))
      m.tbs[i].ff1[0].assign(dat[f'Transformer/encoderblock_{i}/MlpBlock_3/Dense_0/kernel'])
      m.tbs[i].ff1[1].assign(dat[f'Transformer/encoderblock_{i}/MlpBlock_3/Dense_0/bias'])
      m.tbs[i].ff2[0].assign(dat[f'Transformer/encoderblock_{i}/MlpBlock_3/Dense_1/kernel'])
      m.tbs[i].ff2[1].assign(dat[f'Transformer/encoderblock_{i}/MlpBlock_3/Dense_1/bias'])
      m.tbs[i].ln1[0].assign(dat[f'Transformer/encoderblock_{i}/LayerNorm_0/scale'])
      m.tbs[i].ln1[1].assign(dat[f'Transformer/encoderblock_{i}/LayerNorm_0/bias'])
      m.tbs[i].ln2[0].assign(dat[f'Transformer/encoderblock_{i}/LayerNorm_2/scale'])
      m.tbs[i].ln2[1].assign(dat[f'Transformer/encoderblock_{i}/LayerNorm_2/bias'])
