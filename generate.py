import mlx.core as mx
from llm_simple import LLM_Poet

with open("poems.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])

block_size = 64
model = LLM_Poet(
    vocab_size=vocab_size, d_model=64, num_head=2, num_layer=2, block_size=block_size
)
model.load_weights("test_poet_weights.safetensors")
print("Weights loaded successfully!")


def generate_poem_stream(model, starting_text, max_new_tokens=250):
    context = mx.array([encode(starting_text)], dtype=mx.uint32)
    print(starting_text, end="", flush=True)

    for _ in range(max_new_tokens):
        context_cond = context[:, -block_size:]
        logits = model(context_cond)[:, -1, :]
        next_token_id = mx.argmax(logits, axis=-1, keepdims=True)
        print(decode([next_token_id.item()]), end="", flush=True)
        context = mx.concatenate((context, next_token_id), axis=1)

    print("\n")


generate_poem_stream(model, starting_text="Some say the world", max_new_tokens=250)
