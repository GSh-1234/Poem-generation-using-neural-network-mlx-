import mlx.core as mx
import mlx.nn as nn
import math


class selfattention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.W_Q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_K = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_V = nn.Linear(embed_dim, embed_dim, bias=False)

    def __call__(self, x):

        # x shape is (batch_size,sequence_length,embed_dim)

        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)

        # step 2 dot product of q and kt

        kt = K.swapaxes(-1, -2)

        scores = Q @ kt

        # Step 3: The Scale
        # Divide by the square root of the embedding dimension to prevent explosion
        scale = math.sqrt(self.embed_dim)
        scaled_scores = scores / scale

        # step 4 softmax

        attention_weights = nn.softmax(scaled_scores, axis=-1)

        # Step 5: The Extraction
        # Multiply the percentage weights by the Values.
        # (batch, seq, seq) @ (batch, seq, dim) -> (batch, seq, dim)
        output = attention_weights @ V

        return output
