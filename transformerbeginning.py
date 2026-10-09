import mlx.core as mx
import mlx.nn as nn
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class feedforwardnetwork(nn.Module):
    def __init__(self, d_model):
        # exapnd the dimension by 4 times

        self.Layer1 = nn.Linear(d_model, d_model * 4)
        # Project it down to orginal
        self.layer2 = nn.Linear(d_model * 4, d_model)

    def __call__(self, x):
        x = self.linear1(x)
        x = nn.relu(x)
        x = self.linear2(x)
        return x


class multiheadattention(nn.Module):
    def __init__(self, embed_dim, num_head):
        self.embed_dim = embed_dim
        self.num_head = num_head
        self.head_dim = embed_dim // num_head

        self.W_Q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_K = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_V = nn.Linear(embed_dim, embed_dim, bias=False)

        self.W_o = nn.Linear(embed_dim, embed_dim, bias=False)

    def __call__(self, x):
        B, S, D = x.shape

        Q = self.W_Q(x)
        K = self.W_k(x)
        V = self.W_V(x)

        Q = Q.reshape(B, S, self.num_head, self.head_dim)
        K = K.reshape(B, S, self.num_head, self.head_dim)
        V = V.reshape(B, S, self.num_head, self.head_dim)

        Q = Q.swapaxes(1, 2)
        K = K.swapaxes(1, 2)
        V = V.swapaxes(1, 2)

        scores = (Q @ (K.swapaxes(-1, -2))) // (math.sqrt(self.head_dim))
        attention = mx.softmax(scores, axis=-1)

        context = attention @ V
        context = context.swapaxes(1, 2)
        context = context.reshape(B, S, D)

        return self.W_o(context)


class TransformerBlock(nn.Module):
    def __init__(self, d_model=128, num_head=4):
        super().__init__()

        self.ln1 = nn.LayerNorm(d_model)
        self.attention = multiheadattention(d_model, num_head)

        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = feedforwardnetwork(d_model)

    def __call__(self, x):

        layer1_x = self.ln1(x)
        attention_output = self.attention(layer1_x)

        x = x + attention_output

        layer2_x = self.ln2(x)
        ffnoutput = self.ffn(layer2_x)

        x = x + ffnoutput

        return x


class LLM(nn.Module):
    def __init__(self, vocab_size, d_model=128, num_head=4, num_layer=6):
        super().__init__()

        self.embeddings = nn.Embedding(vocab_size, d_model)

        self.layers = [TransformerBlock(d_model, num_head) for _ in range(num_layer)]

        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(
            d_model, vocab_size, bias=False
        )  # predict the next word

    def __call__(self, x):
        x = self.embeddings(x)

        for layer in self.layers:
            x = layer(x)

        x = self.final_norm(x)
        # predict the next word
        logit = self.lm_head(x)
        return logit
