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

# 1. Atenção de Produto Escalar (versão NumPy)
def scaled_dot_product_attention(q, k, v, mask=None):
    d_k = q.shape[-1]
    scores = np.matmul(q, k.transpose(0, 1, 3, 2)) / np.sqrt(d_k)
    if mask is not None:
        # Aplica máscara: onde máscara é 0, define scores para um número muito pequeno (menos infinito)
        scores = np.where(mask == 0, -1e9, scores)
    p_attn = softmax(scores)
    return np.matmul(p_attn, v), p_attn

# Encapsulador de Atenção Multi-Cabeça (versão NumPy)
class MultiHeadAttention:
    def __init__(self, d_model, n_heads):
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.w_q = Linear(d_model, d_model)
        self.w_k = Linear(d_model, d_model)
        self.w_v = Linear(d_model, d_model)
        self.fc = Linear(d_model, d_model)

    def __call__(self, q, k, v, mask=None):
        batch_size = q.shape[0]

        # 1) Realiza todas as projeções lineares em lote de d_model => n_heads x d_k
        q_proj = self.w_q(q).reshape(batch_size, -1, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        k_proj = self.w_k(k).reshape(batch_size, -1, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        v_proj = self.w_v(v).reshape(batch_size, -1, self.n_heads, self.d_k).transpose(0, 2, 1, 3)

        # 2) Aplica atenção em todos os vetores projetados em lote
        x, self.attn = scaled_dot_product_attention(q_proj, k_proj, v_proj, mask=mask)

        # 3) "Concatena" usando uma visualização e aplica uma camada linear final
        x = x.transpose(0, 2, 1, 3).reshape(batch_size, -1, self.n_heads * self.d_k)
        return self.fc(x)

# 2. Rede Feed-Forward Posição-por-Posição (versão NumPy)
class FeedForward:
    def __init__(self, d_model, d_ff, dropout_rate=0.1):
        self.w_1 = Linear(d_model, d_ff)
        self.w_2 = Linear(d_ff, d_model)
        self.dropout_rate = dropout_rate # Dropout é tipicamente tratado durante treinamento

    def __call__(self, x):
        # Em inferência, omitimos aplicação explícita de dropout.
        return self.w_2(relu(self.w_1(x)))

# 3. Adição & Normalização (versão NumPy)
class AddNorm:
    def __init__(self, d_model, dropout_rate=0.1):
        self.norm = LayerNorm(d_model)
        self.dropout_rate = dropout_rate # Dropout é tipicamente tratado durante treinamento

    def __call__(self, x, sublayer_output):
        # Em inferência, dropout é geralmente uma operação nula ou aplica escala
        return x + sublayer_output # Conexão residual

# Codificação Posicional (versão NumPy)
class PositionalEncoding:
    def __init__(self, d_model, max_len=5000):
        self.d_model = d_model
        pe = np.zeros((max_len, d_model))
        position = np.arange(0, max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        self.pe = pe[np.newaxis, :, :]

    def __call__(self, x):
        # Adiciona codificação posicional aos embeddings de entrada
        # x é (batch_size, seq_len, d_model)
        seq_len = x.shape[1]
        return x + self.pe[:, :seq_len]
