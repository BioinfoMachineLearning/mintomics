import torch
from torch import nn
from einops import rearrange, repeat, pack, unpack
# more details refer to the original paper: https://openreview.net/pdf?id=YicbFdNTTy


class FeedForward(nn.Module):   # Feedforward block in encoder
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):  # Multihead attention block in encoder
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):  # The main body, i.e., Encoder
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)   # layernorm is the first step in Attention and FeedForward function
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class GenViT(nn.Module):  # The whole framework of GenViT
    def __init__(self, *, seq_len, in_dim, regression, num_classes, dim, depth, heads, mlp_dim, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()

        #enlarge the embeding dimension so that multihead attention can capture more input information.
        self.gen_embedding = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, dim),
            nn.LayerNorm(dim),
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, dim))
        self.dropout = nn.Dropout(emb_dropout)

        # ViT Body: encoder
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        # MLP after encoder to do regression or classification and each layer can be modified as you wannt
        self.mlp_head = None
        if regression:
            self.mlp_head = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, 1),
                nn.ReLU()
            )
        else:
            self.mlp_head = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, num_classes),
                nn.Softmax()
            )

    def forward(self, series):
        x = self.gen_embedding(series)
        b, n, _ = x.shape   # n = seq_len;
        # x += self.pos_embedding[:, :(n)] # we don't need postional encoding anymore
        x = self.dropout(x)
        x = self.transformer(x)
        x = self.mlp_head(x)
        return x

if __name__ == '__main__':

    Nsample = 2
    NumberGen = 5
    seq_len = NumberGen
    v = GenViT(
        seq_len = NumberGen,
        in_dim = 7,  # input feature
        regression = True,  # if regression true do regression other do classification
        num_classes = 3,
        dim = 1024,   # Last dimension of output tensor after linear transformation
        depth = 6,   # Number of Transformer blocks.
        heads = 8,      # Number of heads in Multi-head Attention layer
        mlp_dim = 2048,     # Dimension of the MLP (FeedForward) layer
        dropout = 0.0,      # Dropout rate.
        emb_dropout = 0.0       # Embedding dropout rate.
    )

    genID = torch.randn(Nsample, NumberGen, 7)
    logits = v(genID) # (Nsample, NumberGen, 1) for regression, (Nsample, NumberGen, 3) for classification
    print(genID.shape)
    print(genID)
    print(logits.shape)
    print(logits)

