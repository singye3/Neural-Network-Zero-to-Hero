
# Tokenizer-BPE
A simple implementation of Byte Pair Encoding (BPE) for tokenization

## Usage
Run the script using Python:
```bash
python3 bpe.py
```
### Example Output:
```
=== BasicTokenizer ===
Encoded: [299, 46, 32, 60, 124, 260, 100, 111, 102, 116, 101, 120, 116, 124, 62]
Decoded: Hello, world! This is a test for the BPE tokenizer. <|endoftext|>

=== RegexTokenizer ===
Encoded: [263, 44, 267, 33, 270, 271, 272, 275, 277, 279, 282, 289, 46, 32, 100257]
Decoded: Hello, world! This is a test for the BPE tokenizer. <|endoftext|>
```
## Reference
Inspired by [karpathy/minbpe](https://github.com/karpathy/minbpe).
