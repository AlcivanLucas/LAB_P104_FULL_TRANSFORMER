# Laboratório 04: O Transformer Completo "From Scratch"

## Visão Geral

Este repositório contém uma implementação educacional de um **Transformer completo (Encoder-Decoder)** utilizando exclusivamente **Python e NumPy**, sem dependências de frameworks de deep learning.

O projeto foi desenvolvido como parte do **Laboratório 4 da disciplina Tópicos em Inteligência Artificial**, ministrada no **ICEV**.

O objetivo é demonstrar o funcionamento interno da arquitetura **Transformer**, conforme apresentado no artigo:

> **Vaswani et al., 2017 — *Attention Is All You Need***

A implementação reproduz os principais componentes da arquitetura original e realiza **inferência autoregressiva** completa com a frase de entrada **"Thinking Machines"**.

> 🔗 **Google Colab:** [Acessar Notebook](https://colab.research.google.com/drive/1KkN6P3vw2fkFbteYiJaDadT1GnIVabYd?usp=sharing)

---

## Série de Laboratórios

| Lab | Tema | Componentes |
|-----|------|-------------|
| Lab 01 | Mecanismo de Self-Attention | Scaled Dot-Product Attention, cálculo de Q, K, V |
| Lab 02 | Transformer Encoder | Pilha de Encoder, Add & Norm, FFN |
| Lab 03 | Transformer Decoder | Máscara Causal, Cross-Attention, Loop Auto-regressivo |
| **Lab 04** | **Transformer Completo** ← *Você está aqui* | **Integração Encoder-Decoder, inferência fim-a-fim** |

---

## Estrutura do Projeto

```
LAB_P104_FULL_TRANSFORMER/
├── transformer_model.py    # Implementação completa do Transformer
├── README.md               # Esta documentação
└── .gitignore
```

---

## Arquitetura do Transformer

```
INPUT ("Thinking Machines")
    │
    ▼
┌──────────────────────────┐
│       ENCODER            │
│  (3 camadas empilhadas)  │
│                          │
│  Embedding + Pos. Enc.   │
│         │                │
│  ┌──────▼──────┐         │
│  │ Self-Attn   │─┐       │
│  └──────┬──────┘ │       │
│  ┌──────▼──────┐ │       │
│  │ Add & Norm  │◄┘       │
│  └──────┬──────┘         │
│  ┌──────▼──────┐         │
│  │    FFN      │─┐       │
│  └──────┬──────┘ │       │
│  ┌──────▼──────┐ │       │
│  │ Add & Norm  │◄┘       │
│  └──────┬──────┘         │
│         │  (×3 camadas)  │
└─────────┼────────────────┘
          │
          ▼ Z (representação contextualizada)
          │
┌─────────┼────────────────┐
│       DECODER            │
│  (3 camadas empilhadas)  │
│                          │
│  Embedding + Pos. Enc.   │
│         │                │
│  ┌──────▼──────┐         │
│  │ Masked      │─┐       │
│  │ Self-Attn   │ │       │
│  └──────┬──────┘ │       │
│  ┌──────▼──────┐ │       │
│  │ Add & Norm  │◄┘       │
│  └──────┬──────┘         │
│  ┌──────▼──────┐         │
│  │ Cross-Attn  │◄── Z    │
│  │ (Q=dec,     │         │
│  │  K,V=enc)   │─┐       │
│  └──────┬──────┘ │       │
│  ┌──────▼──────┐ │       │
│  │ Add & Norm  │◄┘       │
│  └──────┬──────┘         │
│  ┌──────▼──────┐         │
│  │    FFN      │─┐       │
│  └──────┬──────┘ │       │
│  ┌──────▼──────┐ │       │
│  │ Add & Norm  │◄┘       │
│  └──────┬──────┘         │
│         │  (×3 camadas)  │
└─────────┼────────────────┘
          │
          ▼
   Linear → vocab_size
          │
          ▼
       Softmax
          │
          ▼
    P(next_token) → argmax → token gerado
```

---

## Componentes Implementados

### Tarefa 1 — Refatoração e Integração (Blocos de Montar)

#### Funções de Ativação

- **`relu(x)`** — Ativação ReLU: `max(0, x)`
- **`softmax(x)`** — Softmax numericamente estável para converter logits em probabilidades

#### `Tokenizer`
Vocabulário e tokenização simplificados para o exemplo. Converte frases em sequências de IDs numéricos e vice-versa, com suporte a tokens especiais (`<pad>`, `<start>`, `<eos>`).

#### `Linear`
Camada linear básica: `y = xW + b`. Inicializa pesos com distribuição normal escalada e bias em zero.

#### `Embedding`
Tabela de embeddings que mapeia IDs de tokens para vetores densos de dimensão `d_model`.

#### `LayerNorm`
Layer Normalization com parâmetros treináveis γ (gamma) e β (beta):

```
LayerNorm(x) = γ · (x - μ) / (σ + ε) + β
```

Normaliza a última dimensão (features) para estabilizar o fluxo numérico do modelo.

---

### Tarefa 2 — Mecanismo de Atenção e Encoder

#### `scaled_dot_product_attention(Q, K, V, mask)`

Fórmula central do Transformer:

```
Attention(Q, K, V) = softmax(QKᵀ / √d_k) · V
```

Onde `d_k` é a dimensão das chaves. A divisão por `√d_k` evita que os produtos escalares cresçam demais com o aumento da dimensionalidade. Suporta máscara causal opcional para o Decoder.

#### `MultiHeadAttention`

Implementa múltiplas cabeças de atenção em paralelo:

1. Projeta Q, K, V através de camadas lineares
2. Divide em `n_heads` cabeças de dimensão `d_k = d_model / n_heads`
3. Aplica `scaled_dot_product_attention` em cada cabeça
4. Concatena resultados e aplica projeção linear final

Usada tanto para **Self-Attention** (Q=K=V) quanto para **Cross-Attention** (Q do Decoder, K/V do Encoder).

#### `FeedForward`

Rede Feed-Forward posição-por-posição (aplicada independentemente a cada posição da sequência):

```
FFN(x) = ReLU(xW₁ + b₁)W₂ + b₂
```

Expande a dimensionalidade de `d_model` para `d_ff` (tipicamente `4 × d_model`) e depois contrai de volta.

#### `AddNorm`

Conexão residual que preserva o fluxo de gradientes:

```
Output = x + Sublayer(x)
```

#### `PositionalEncoding`

Codificação posicional usando funções seno e cosseno conforme o paper original:

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

Permite ao modelo distinguir diferentes posições na sequência, já que a atenção é invariante à ordem.

#### `EncoderBlock`

Um bloco do Encoder executando o seguinte fluxo:

```
Input X
    ↓
Self-Attention (Q, K, V = X)
    ↓
Add & Norm
    ↓
Feed-Forward Network
    ↓
Add & Norm
    ↓
Output
```

O Encoder empilha `n_encoder_layers` blocos. A saída final **Z** contém as representações contextualizadas de toda a sequência de entrada.

---

### Tarefa 3 — Decoder e Modelo Completo

#### `DecoderBlock`

Cada bloco do Decoder possui três subcamadas:

```
Input Y
    ↓
1. Masked Self-Attention (com máscara causal)
    ↓
   Add & Norm
    ↓
2. Cross-Attention (Q = Decoder, K/V = Encoder Z)
    ↓
   Add & Norm
    ↓
3. Feed-Forward Network
    ↓
   Add & Norm
    ↓
Output
```

A **máscara causal** impede o modelo de acessar tokens futuros durante a geração:

```
Exemplo de máscara (seq_len=4):
[[ True  False  False  False]
 [ True  True   False  False]
 [ True  True   True   False]
 [ True  True   True   True ]]
```

A **Cross-Attention** é a ponte entre Encoder e Decoder: o Decoder formula **Queries** a partir de sua própria representação, mas consulta **Keys** e **Values** vindos da saída Z do Encoder.

#### `Transformer`

Classe que integra todos os componentes:

1. **Encoder:** `src → Embedding → Positional Encoding → N × EncoderBlock → Z`
2. **Decoder:** `tgt → Embedding → Positional Encoding → N × DecoderBlock(Z) → output`
3. **Output:** `output → Linear(d_model → vocab_size) → logits`

#### `create_masks(src, tgt, pad_idx)`

Gera as máscaras necessárias:
- **src_mask:** Máscara de padding para o Encoder
- **tgt_mask:** Combinação da máscara causal (triangular inferior) com máscara de padding

---

### Tarefa 4 — Inferência Auto-regressiva (A Prova Final)

#### `translate_sentence()`

Implementa o loop auto-regressivo de geração (Greedy Decoding):

```
1. Codificar sentença de entrada pelo Encoder → Z
2. Inicializar sequência do Decoder com <start>
3. while not <eos> and len < max_len:
    a. Criar máscara causal para sequência atual
    b. Forward pass pelo Decoder com Z do Encoder
    c. Projetar saída para vocab_size
    d. Selecionar token com maior probabilidade (argmax)
    e. Concatenar token à sequência
4. Retornar sentença traduzida
```

O Encoder é executado **apenas uma vez**, enquanto o Decoder é chamado iterativamente a cada novo token gerado.

---

## Parâmetros do Modelo

| Parâmetro | Valor | Descrição |
|-----------|-------|-----------|
| `d_model` | 512 | Dimensão dos embeddings e representações internas |
| `n_heads` | 8 | Número de cabeças de atenção (`d_k = 512/8 = 64`) |
| `d_ff` | 2048 | Dimensão interna da FFN (`4 × d_model`) |
| `n_encoder_layers` | 3 | Número de camadas do Encoder |
| `n_decoder_layers` | 3 | Número de camadas do Decoder |
| `max_len` | 100 | Comprimento máximo de sequência |

---

## Como Executar

### Executar localmente

```bash
python transformer_model.py
```

### Saída esperada

```
Tamanho do Vocabulário de Origem: 5
Tamanho do Vocabulário de Alvo: 15
Modelo Transformador instanciado com sucesso usando NumPy!

Traduzindo: 'Thinking Machines'
Traduzido: 'um um <eos>'
```

> **Nota:** Como os pesos são aleatórios (modelo não treinado), a sequência gerada não possui significado semântico. O objetivo é validar que o **fluxo estrutural do Transformer está correto** — os tensores passam por todas as camadas com as dimensões esperadas e o loop auto-regressivo funciona até gerar `<eos>` ou atingir o comprimento máximo.

### Executar no Google Colab

🔗 [Abrir no Google Colab](https://colab.research.google.com/drive/1KkN6P3vw2fkFbteYiJaDadT1GnIVabYd?usp=sharing)

---

## Tecnologias Utilizadas

- **Python 3** — Linguagem principal
- **NumPy** — Operações matriciais e álgebra linear

---

## Conceitos-Chave Aplicados

| Conceito | Referência | Implementação |
|----------|-----------|---------------|
| Scaled Dot-Product Attention | Vaswani et al. (2017) | `scaled_dot_product_attention()` |
| Multi-Head Attention | Vaswani et al. (2017) | `MultiHeadAttention` |
| Positional Encoding (sin/cos) | Vaswani et al. (2017) | `PositionalEncoding` |
| Layer Normalization | Ba et al. (2016) | `LayerNorm` |
| Conexões Residuais | He et al. (2015) | `AddNorm` |
| Máscara Causal | Vaswani et al. (2017) | `create_masks()` |
| Geração Auto-regressiva | Graves (2013) | `translate_sentence()` |

---

## Referências

1. **"Attention is All You Need"** — Vaswani et al., 2017
   - Paper original do Transformer
   - Define arquitetura Encoder-Decoder com atenção
   - [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)

2. **"Layer Normalization"** — Ba et al., 2016
   - Alternativa ao Batch Norm para Transformers
   - [https://arxiv.org/abs/1607.06450](https://arxiv.org/abs/1607.06450)

3. **"Deep Residual Learning"** — He et al., 2015
   - Conexões residuais para redes profundas
   - [https://arxiv.org/abs/1512.03385](https://arxiv.org/abs/1512.03385)

---

## Versionamento Git

```bash
git tag v1.0
git push origin main --tags
```

O commit avaliado possui a tag `v1.0`.

---

## Observação sobre Uso de Inteligência Artificial

Partes do código foram auxiliadas por ferramentas de Inteligência Artificial generativa (assistentes de código baseados em LLMs) para apoiar na estruturação, documentação e revisão do código.

Todo o código foi revisado manualmente para garantir aderência às especificações da atividade e compreensão dos conceitos implementados.

- ✅ Código verificado para correção matemática
- ✅ Lógica de Encoder-Decoder compreendida e validada
- ✅ Fluxo de tensores testado em todas as camadas
- ✅ Máscara causal e Cross-Attention implementadas corretamente
- ✅ Loop auto-regressivo de inferência funcional

---

## Autor

**Alcivan Lucas**
Engenharia de Software — ICEV

Disciplina: Tópicos em Inteligência Artificial
