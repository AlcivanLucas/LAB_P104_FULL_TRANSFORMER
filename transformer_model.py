import numpy as np

# Helper functions for activation and loss (NumPy)
def relu(x):
    return np.maximum(0, x)

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True)) # For numerical stability
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

# Vocabulary and tokenization (simplified for toy example)
class Tokenizer:
    def __init__(self, special_tokens=None):
        self.word_to_idx = {}
        self.idx_to_word = []
        self.special_tokens = special_tokens if special_tokens else []
        for token in self.special_tokens:
            self.add_word(token)

    def add_word(self, word):
        if word not in self.word_to_idx:
            self.word_to_idx[word] = len(self.idx_to_word)
            self.idx_to_word.append(word)
        return self.word_to_idx[word]

    def tokenize(self, sentence):
        words = sentence.lower().split()
        return [self.add_word(word) for word in words]

    def decode(self, indices):
        return ' '.join([self.idx_to_word[idx] for idx in indices if idx < len(self.idx_to_word)])

    @property
    def vocab_size(self):
        return len(self.idx_to_word)
