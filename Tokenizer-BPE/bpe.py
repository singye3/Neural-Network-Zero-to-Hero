#!/usr/bin/env python
"""
mini BPE tokenizer implementation
Based on the minbpe  by Karapathy
"""

import unicodedata
import regex as re

# -----------------------------------------------------------------------------
# Helper functions

def get_stats(token_ids, pair_counts=None):
    """
    Given a list of token IDs, return a dictionary counting consecutive pairs.
    Example: [1, 2, 3, 1, 2] -> {(1, 2): 2, (2, 3): 1, (3, 1): 1}
    Optionally update an existing dictionary of pair counts.
    """
    pair_counts = {} if pair_counts is None else pair_counts
    for first, second in zip(token_ids, token_ids[1:]):
        pair = (first, second)
        pair_counts[pair] = pair_counts.get(pair, 0) + 1
    return pair_counts

def merge(ids, pair, idx):
    """
    In the list of integers (ids), replace every consecutive occurrence
    of the given pair with the new token id.
    Example: ids=[1, 2, 3, 1, 2], pair=(1, 2), idx=4 -> [4, 3, 4]
    """
    new_ids = []
    i = 0
    while i < len(ids):
        if ids[i] == pair[0] and i < len(ids) - 1 and ids[i+1] == pair[1]:
            new_ids.append(idx)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids

def replace_control_characters(s: str) -> str:
    """
    Replace control characters in s with their Unicode escape sequences.
    """
    chars = []
    for ch in s:
        if unicodedata.category(ch)[0] != "C":
            chars.append(ch)
        else:
            chars.append(f"\\u{ord(ch):04x}")
    return "".join(chars)

def render_token(t: bytes) -> str:
    """
    Decodes a byte token into a string and escapes control characters.
    """
    s = t.decode('utf-8', errors='replace')
    return replace_control_characters(s)

# -----------------------------------------------------------------------------
# Base Tokenizer class

class Tokenizer:
    """Base class for Tokenizers with common save/load functionality."""
    def __init__(self):
        # By default, the vocabulary starts with all 256 byte tokens.
        self.merges = {}           # Mapping: (int, int) -> int (for merged tokens)
        self.pattern = ""          # Optional splitting pattern (used in RegexTokenizer)
        self.special_tokens = {}   # Mapping: special token (str) -> int
        self.vocab = self._build_vocab()  # Mapping: int -> bytes

    def train(self, text, vocab_size, verbose=False):
        raise NotImplementedError("Subclasses must implement train()")

    def encode(self, text):
        raise NotImplementedError("Subclasses must implement encode()")

    def decode(self, ids):
        raise NotImplementedError("Subclasses must implement decode()")

    def _build_vocab(self):
        # Start with 256 byte tokens and add merged tokens
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode("utf-8")
        return vocab

    def save(self, file_prefix):
        """
        Save the tokenizer model and a human-readable vocabulary.
        This creates:
          - {file_prefix}.model : used for loading later.
          - {file_prefix}.vocab : a pretty-printed version.
        """
        model_file = file_prefix + ".model"
        with open(model_file, 'w', encoding="utf-8") as f:
            f.write("minbpe v1\n")
            f.write(f"{self.pattern}\n")
            f.write(f"{len(self.special_tokens)}\n")
            for special, idx in self.special_tokens.items():
                f.write(f"{special} {idx}\n")
            for idx1, idx2 in self.merges:
                f.write(f"{idx1} {idx2}\n")
        vocab_file = file_prefix + ".vocab"
        inverted_merges = {idx: pair for pair, idx in self.merges.items()}
        with open(vocab_file, "w", encoding="utf-8") as f:
            for idx, token in self.vocab.items():
                s = render_token(token)
                if idx in inverted_merges:
                    idx0, idx1 = inverted_merges[idx]
                    s0 = render_token(self.vocab[idx0])
                    s1 = render_token(self.vocab[idx1])
                    f.write(f"[{s0}][{s1}] -> [{s}] {idx}\n")
                else:
                    f.write(f"[{s}] {idx}\n")

    def load(self, model_file):
        """
        Load a tokenizer model from a .model file.
        """
        assert model_file.endswith(".model")
        merges = {}
        special_tokens = {}
        idx = 256
        with open(model_file, 'r', encoding="utf-8") as f:
            version = f.readline().strip()
            assert version == "minbpe v1", "Unknown model version!"
            self.pattern = f.readline().strip()
            num_special = int(f.readline().strip())
            for _ in range(num_special):
                special, special_idx = f.readline().strip().split()
                special_tokens[special] = int(special_idx)
            for line in f:
                idx1, idx2 = map(int, line.split())
                merges[(idx1, idx2)] = idx
                idx += 1
        self.merges = merges
        self.special_tokens = special_tokens
        self.vocab = self._build_vocab()

# -----------------------------------------------------------------------------
# BasicTokenizer: Minimal byte-level BPE tokenizer

class BasicTokenizer(Tokenizer):
    def __init__(self):
        super().__init__()

    def train(self, text, vocab_size, verbose=False):
        """
        Train the tokenizer using a simple byte-level BPE approach.
        """
        assert vocab_size >= 256, "vocab_size must be at least 256"
        num_merges = vocab_size - 256
        text_bytes = text.encode("utf-8")
        ids = list(text_bytes)
        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for i in range(num_merges):
            stats = get_stats(ids)
            pair = max(stats, key=stats.get)
            idx = 256 + i
            ids = merge(ids, pair, idx)
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences")
        self.merges = merges
        self.vocab = vocab

    def encode(self, text):
        """
        Encode a string into a list of token ids.
        """
        text_bytes = text.encode("utf-8")
        ids = list(text_bytes)
        while len(ids) >= 2:
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids

    def decode(self, ids):
        """
        Decode a list of token ids back into a string.
        """
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        return text_bytes.decode("utf-8", errors="replace")

# -----------------------------------------------------------------------------
# RegexTokenizer: BPE tokenizer with regex splitting and special token support

# Default patterns (from GPT variants)
GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

class RegexTokenizer(Tokenizer):
    def __init__(self, pattern=None):
        super().__init__()
        self.pattern = GPT4_SPLIT_PATTERN if pattern is None else pattern
        self.compiled_pattern = re.compile(self.pattern)
        self.special_tokens = {}          # Mapping: special token (str) -> int
        self.inverse_special_tokens = {}  # Mapping: id -> special token (str)

    def train(self, text, vocab_size, verbose=False):
        """
        Train the tokenizer using a regex-based splitting approach.
        """
        assert vocab_size >= 256, "vocab_size must be at least 256"
        num_merges = vocab_size - 256
        # Split the text into chunks based on the regex pattern
        text_chunks = re.findall(self.compiled_pattern, text)
        # Convert each chunk to a list of byte values
        ids_list = [list(chunk.encode("utf-8")) for chunk in text_chunks]
        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for i in range(num_merges):
            stats = {}
            for ids in ids_list:
                get_stats(ids, stats)
            if not stats:
                break  # No more pairs to merge
            pair = max(stats, key=stats.get)
            idx = 256 + i
            ids_list = [merge(ids, pair, idx) for ids in ids_list]
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences")
        self.merges = merges
        self.vocab = vocab

    def register_special_tokens(self, special_tokens):
        """
        Register special tokens. special_tokens is a dict mapping token string to id.
        Example: {"<|endoftext|>": 100257}
        """
        self.special_tokens = special_tokens
        self.inverse_special_tokens = {v: k for k, v in special_tokens.items()}

    def _encode_chunk(self, text_bytes):
        """
        Encode a single chunk of text (as bytes) into token ids.
        """
        ids = list(text_bytes)
        while len(ids) >= 2:
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids

    def encode_ordinary(self, text):
        """
        Encode text that does not include any special tokens.
        """
        text_chunks = re.findall(self.compiled_pattern, text)
        ids = []
        for chunk in text_chunks:
            chunk_bytes = chunk.encode("utf-8")
            chunk_ids = self._encode_chunk(chunk_bytes)
            ids.extend(chunk_ids)
        return ids

    def encode(self, text, allowed_special="none_raise"):
        """
        Encode text into token ids while handling special tokens.

        allowed_special can be:
          - "all": allow all special tokens.
          - "none": ignore special tokens.
          - "none_raise": raise an error if any special token is encountered.
          - A custom set of special token strings.
        """
        if allowed_special == "all":
            special = self.special_tokens
        elif allowed_special == "none":
            special = {}
        elif allowed_special == "none_raise":
            special = {}
            for token in self.special_tokens:
                if token in text:
                    raise ValueError(f"Encountered special token {token} in text.")
        elif isinstance(allowed_special, set):
            special = {k: v for k, v in self.special_tokens.items() if k in allowed_special}
        else:
            raise ValueError(f"allowed_special={allowed_special} not understood")
        if not special:
            return self.encode_ordinary(text)
        # Split text into parts using the special tokens as separators.
        special_pattern = "(" + "|".join(re.escape(token) for token in special) + ")"
        parts = re.split(special_pattern, text)
        ids = []
        for part in parts:
            if part in special:
                ids.append(special[part])
            else:
                ids.extend(self.encode_ordinary(part))
        return ids

    def decode(self, ids):
        """
        Decode a list of token ids into a string.
        """
        parts = []
        for idx in ids:
            if idx in self.vocab:
                parts.append(self.vocab[idx])
            elif idx in self.inverse_special_tokens:
                parts.append(self.inverse_special_tokens[idx].encode("utf-8"))
            else:
                raise ValueError(f"Invalid token id: {idx}")
        text_bytes = b"".join(parts)
        return text_bytes.decode("utf-8", errors="replace")

# -----------------------------------------------------------------------------
# Example usage

if __name__ == "__main__":
    sample_text = "Hello, world! This is a test for the BPE tokenizer. <|endoftext|>"

    print("=== BasicTokenizer ===")
    basic_tokenizer = BasicTokenizer()
    basic_tokenizer.train(sample_text, vocab_size=300, verbose=False)
    encoded_basic = basic_tokenizer.encode(sample_text)
    decoded_basic = basic_tokenizer.decode(encoded_basic)
    print("Encoded:", encoded_basic)
    print("Decoded:", decoded_basic)

    print("\n=== RegexTokenizer ===")
    regex_tokenizer = RegexTokenizer()
    regex_tokenizer.register_special_tokens({"<|endoftext|>": 100257})
    regex_tokenizer.train(sample_text, vocab_size=300, verbose=False)
    encoded_regex = regex_tokenizer.encode(sample_text, allowed_special="all")
    decoded_regex = regex_tokenizer.decode(encoded_regex)
    print("Encoded:", encoded_regex)
    print("Decoded:", decoded_regex)
