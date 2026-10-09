import json


class BPETokenizer:
    def __init__(self):
        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        self.merges = {}
        self.special_token = "<|endoftext|>"
        self.special_token_id = None

    def get_stats(self, ids):
        counts = {}
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    def merge(self, ids, pair, idx):
        new_ids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                new_ids.append(idx)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids

    def train(self, text, target_vocab_size):
        tokens = list(text.encode("utf-8"))
        num_merges = target_vocab_size - 256

        for i in range(num_merges):
            stats = self.get_stats(tokens)
            if not stats:
                break
            top_pair = max(stats, key=stats.get)
            new_idx = 256 + i
            tokens = self.merge(tokens, top_pair, new_idx)
            self.merges[top_pair] = new_idx
            self.vocab[new_idx] = self.vocab[top_pair[0]] + self.vocab[top_pair[1]]

        self.special_token_id = len(self.vocab)
        self.vocab[self.special_token_id] = self.special_token.encode("utf-8")

    def encode(self, text):
        tokens = list(text.encode("utf-8"))
        while len(tokens) >= 2:
            stats = self.get_stats(tokens)
            pair = min(stats.keys(), key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            idx = self.merges[pair]
            tokens = self.merge(tokens, pair, idx)
        return tokens

    def decode(self, ids):
        tokens = b"".join(self.vocab.get(idx, b"") for idx in ids)
        return tokens.decode("utf-8", errors="replace")
