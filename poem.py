import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from llm_simple import LLM_Poet
from BPEtokensier import BPETokenizer

with open("poems.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# Train the tokenizer to exactly 500 tokens
tokenizer = BPETokenizer()
tokenizer.train(raw_text, target_vocab_size=500)
vocab_size = len(tokenizer.vocab)

# Split poems by double newline and encode them
individual_poems = raw_text.split("\n\n")
encoded_data = []

for poem in individual_poems:
    if poem.strip():  # Skip empty strings
        # Encode the poem and inject the <|endoftext|> token at the end
        encoded_data.extend(tokenizer.encode(poem))
        encoded_data.append(tokenizer.special_token_id)

data = mx.array(encoded_data, dtype=mx.uint32)
print(f"BPE Tokenization Complete. Total tokens in dataset: {len(data)}")
print(f"Vocabulary Size: {vocab_size}")

n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

block_size = 256
batch_size = 4


def get_batch(split="train"):
    d = train_data if split == "train" else val_data

    # Ensure we don't pick an index out of bounds on a tiny dataset
    max_idx = max(1, len(d) - block_size)
    ix = np.random.randint(0, max_idx, batch_size)

    x = mx.stack([d[i : i + block_size] for i in ix])
    y = mx.stack([d[i + 1 : i + block_size + 1] for i in ix])
    return x, y


model = LLM_Poet(
    vocab_size=vocab_size, d_model=64, num_head=2, num_layer=2, block_size=256
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


# 4. The Training Loop
print("Training started...")
mx.eval(model.parameters(), optimizer.state)

# 1000 iterations is usually enough to overfit a tiny text file
for iter in range(10000):
    xb, yb = get_batch("train")
    loss = step(model, optimizer, xb, yb)

    mx.eval(model.parameters(), optimizer.state)

    if iter % 100 == 0:
        print(f"Iteration {iter:4d} | Loss: {loss.item():.4f}")

# 5. Save the final weights
model.save_weights("test_poet_weights.safetensors")
print("Test complete! Weights written to test_poet_weights.safetensors")
