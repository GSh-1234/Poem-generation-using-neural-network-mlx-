import mlx.core as mx
from llm_simple import LLM_Poet
from BPEtokensier import BPETokenizer

with open("poems.txt", "r", encoding="utf-8") as f:
    text = f.read()

tokenizer = BPETokenizer()
# IMPORTANT: target_vocab_size must exactly match what you used in train_poems.py!
tokenizer.train(text, target_vocab_size=500)
vocab_size = len(tokenizer.vocab)

# Initialize model (Ensure block_size matches your training script!)
model = LLM_Poet(
    vocab_size=vocab_size, d_model=64, num_head=2, num_layer=2, block_size=256
)
model.load_weights("test_poet_weights.safetensors")

# 2. Instantiate the model with the exact same tiny hyperparameters
model = LLM_Poet(vocab_size=vocab_size, d_model=64, num_head=2, num_layer=2)

# 3. Inject the trained weights
model.load_weights("test_poet_weights.safetensors")
print("Weights loaded successfully!")


# ==========================================
# 3. STREAMING GENERATION LOOP
# ==========================================
def generate_poem_stream(model, starting_text, max_new_tokens=200):
    # Encode prompt using BPE
    encoded_prompt = tokenizer.encode(starting_text)
    context = mx.array([encoded_prompt], dtype=mx.uint32)
    block_size = 256

    print(starting_text, end="", flush=True)

    for _ in range(max_new_tokens):
        # Crop context to block_size
        context_cond = context[:, -block_size:]

        # Get predictions
        logits = model(context_cond)
        logits = logits[:, -1, :]

        # Greedy Decoding (Always pick the most probable next token)
        next_token_id = mx.argmax(logits, axis=-1, keepdims=True)

        # Stop generating if the model outputs <|endoftext|>
        if next_token_id.item() == tokenizer.special_token_id:
            break

        # Decode the single token back to string
        next_text = tokenizer.decode([next_token_id.item()])

        # Stream to terminal
        print(next_text, end="", flush=True)

        # Append to context for the next loop
        context = mx.concatenate((context, next_token_id), axis=1)

    print("\n")


print("\n--- STREAMING BPE GENERATION ---")
generate_poem_stream(model, starting_text="Some say the world ", max_new_tokens=250)
