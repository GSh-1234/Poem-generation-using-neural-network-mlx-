import math
import mlx.core as mx
import mlx.nn as nn


class Multiheadattention(nn.Module):
    def __init__(self, embed_num, num_head):
        super().__init__()
        self.embed_num = embed_num
        self.num_head = num_head

        self.head_dim = embed_num // num_head

        self.W_Q = nn.Linear(embed_num, embed_num, bias=False)
        self.W_K = nn.Linear(embed_num, embed_num, bias=False)
        self.W_V = nn.Linear(embed_num, embed_num, bias=False)

        self.W_o = nn.Linear(embed_num, embed_num, bias=False)

    def __call__(self, x):
        B, S, D = x.shape

        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)

        Q = Q.reshape(B, S, self.num_head, self.head_dim)
        K = K.reshape(B, S, self.num_head, self.head_dim)
        V = V.reshape(B, S, self.num_head, self.head_dim)

        Q = Q.swapaxes(1, 2)
        K = K.swapaxes(1, 2)
        V = V.swapaxes(1, 2)

        score = (Q @ (K.swapaxes(-1, -2))) // (math.sqrt(self.head_dim))
        indices = mx.arange(S)
        mask = indices[:, None] < indices[None, :]
        score = mx.where(mask, mx.array(-1e9), score)

        attention = mx.softmax(score, axis=-1)

        context = attention @ V
        context = context.swapaxes(1, 2)
        context = context.reshape(B, S, D)

        return self.W_o(context)


class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model=128, num_head=4):
        super().__init__()
        self.d_model = d_model
        self.num_head = num_head

        self.layer1 = nn.Linear(d_model, d_model * 4, bias=False)
        self.layer2 = nn.Linear(d_model * 4, d_model, bias=False)

    def __call__(self, x):
        x = self.layer1(x)
        x = nn.relu(x)
        x = self.layer2(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, d_model=128, num_head=4):
        super().__init__()

        self.ln1 = nn.LayerNorm(d_model)
        self.attention = Multiheadattention(d_model, num_head)

        self.ln2 = nn.LayerNorm(d_model)
        self.fnn = FeedForwardNetwork(d_model, num_head)

    def __call__(self, x):
        x1 = self.ln1(x)
        attention_x1 = self.attention(x1)
        x = x + attention_x1

        x2 = self.ln2(x)
        ffn_x2 = self.fnn(x2)
        x = x + ffn_x2

        return x


class LLM_Poet(nn.Module):
    def __init__(
        self, vocab_size, d_model=128, num_head=4, num_layer=6, block_size=256
    ):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, d_model)
        self.pos_embeddings = nn.Embedding(block_size, d_model)

        self.layers = [TransformerBlock(d_model, num_head) for _ in range(num_layer)]

        self.final_norm = nn.LayerNorm(d_model)
        self.final_output = nn.Linear(d_model, vocab_size, bias=False)

    def __call__(self, x):
        B, S = x.shape
        tok_emb = self.embeddings(x)
        positions = mx.arange(S)
        pos_emb = self.pos_embeddings(positions)
        x = tok_emb + pos_emb

        for layer in self.layers:
            x = layer(x)

        x = self.final_norm(x)
        logit = self.final_output(x)

        return logit
