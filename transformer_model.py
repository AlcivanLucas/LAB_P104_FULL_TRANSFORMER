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

# Basic Linear Layer (NumPy)
class Linear:
    def __init__(self, input_dim, output_dim):
        # Initialize weights and biases
        self.weights = np.random.randn(input_dim, output_dim) * 0.01
        self.biases = np.zeros(output_dim)

    def __call__(self, x):
        return np.dot(x, self.weights) + self.biases

# Basic Embedding Layer (NumPy)
class Embedding:
    def __init__(self, vocab_size, embed_dim):
        self.embedding_matrix = np.random.randn(vocab_size, embed_dim) * 0.01

    def __call__(self, x):
        return self.embedding_matrix[x]

# Layer Normalization (NumPy)
class LayerNorm:
    def __init__(self, features, eps=1e-6):
        self.gamma = np.ones(features)
        self.beta = np.zeros(features)
        self.eps = eps

    def __call__(self, x):
        mean = x.mean(axis=-1, keepdims=True)
        std = x.std(axis=-1, keepdims=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta
