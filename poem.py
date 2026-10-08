import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from llm_simple import LLM_Poet

with open("poems.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])

data = mx.array(encode(text), dtype=mx.uint32)

n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

block_size = 64
batch_size = 4


def get_batch(split="train"):
    d = train_data if split == "train" else val_data
    ix = np.random.randint(0, len(d) - block_size - 1, batch_size)
    x = mx.stack([d[i : i + block_size] for i in ix])
    y = mx.stack([d[i + 1 : i + block_size + 1] for i in ix])
    return x, y


model = LLM_Poet(
    vocab_size=vocab_size, d_model=64, num_head=2, num_layer=2, block_size=block_size
)
optimizer = optim.AdamW(learning_rate=1e-3, weight_decay=0.01)


def lossfn(model, x, y):
    logits = model(x)
    B, S, V = logits.shape
    logits = logits.reshape(B * S, V)
    y = y.reshape(B * S)
    return mx.mean(nn.losses.cross_entropy(logits, y))


loss_and_grad_fn = nn.value_and_grad(model, lossfn)


def step(model, optimizer, x, y):
    loss, grads = loss_and_grad_fn(model, x, y)
    optimizer.update(model, grads)
    return loss


print("Training character-level model...")
mx.eval(model.parameters(), optimizer.state)

for iter in range(1000):
    xb, yb = get_batch("train")
    loss = step(model, optimizer, xb, yb)
    mx.eval(model.parameters(), optimizer.state)
    if iter % 100 == 0:
        print(f"Iteration {iter:4d} | Loss: {loss.item():.4f}")

model.save_weights("test_poet_weights.safetensors")
print("Weights written to test_poet_weights.safetensors")
