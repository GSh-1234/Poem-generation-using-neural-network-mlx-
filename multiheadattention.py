import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

"""class multihead(nn.Module):

    def __init__(self,embed_dim,num_head):
        self.embed_dim=embed_dim
        self.num_head=num_head

        assert embed_dim % num_head == 0 
        self.head_dim = embed_dim // num_head

        self.W_Q=nn.Linear(embed_dim,embed_dim,bias=False)
        self.W_K=nn.Linear(embed_dim,embed_dim,bias=False)
        self.W_V=nn.Linear(embed_dim,embed_dim,bias=False)

        self.W_o=nn.Linear(embed_dim,embed_dim,bias=False)

    
    def __call__(self,x):
        B,S,D=x.shape

        Q=self.W_Q(x)
        K=self.W_K(x)
        V=self.W_V(x)

        Q=Q.reshape(B,S,self.num_head,self.head_dim)
        K=K.reshape(B,S,self.num_head,self.head_dim)
        V=V.reshape(B,S,self.head_dim,self.num_head)

        Q=Q.swapaxes(1,2)
        K=K.swapaxes(1,2)
        V=V.swapaxes(1,2)

        scores=(Q@(K.swapaxes(-1,-2)) // math.sqrt(self.head_dim))
        attention=mx.softmax(scores,axis=-1)

        context=attention@V
        context=context.swapaxes(1,2)
        context=context.reshape(B,S,D)

        return self.W_o(context)
    """


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
        K = self.W_K(x)
        V = self.W_V(x)

        Q = Q.reshape(B, S, self.num_head, self.head_dim)
        K = K.reshape(B, S, self.num_head, self.head_dim)
        V = V.reshape(B, S, self.num_head, self.head_dim)

        Q = Q.swapaxes(1, 2)
        K = K.swapaxes(1, 2)
        V = V.swapaxes(1, 2)

        scores = (Q @ K.swapaxes(-1, -2)) // (math.sqrt(self.head_dim))

        attention = mx.softmax(scores, axis=-1)

        context = attention @ V
        context = context.swapaxes(1, 2)
        context = context.reshape(B, S, D)

        return self.W_o(context)
