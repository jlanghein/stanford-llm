# Transformer Explainer -- 20 Steps

## Table of Contents

- [I. The Big Picture (What is happening?)](#i-the-big-picture-what-is-happening)
- [II. Turning Text into Numbers (Input Representation)](#ii-turning-text-into-numbers-input-representation)
- [III. The Core Engine (Transformer Blocks)](#iii-the-core-engine-transformer-blocks)
- [IV. Stability & Regularization (Why it trains properly)](#iv-stability--regularization-why-it-trains-properly)
- [V. Producing the Next-Token Scores (Output Layer)](#v-producing-the-next-token-scores-output-layer)
- [VI. Sampling Strategy (Decision Layer)](#vi-sampling-strategy-decision-layer)

---

## I. The Big Picture (What is happening?)

### 1. What is Transformer?

Transformer is the core architecture behind modern AI, powering models
like ChatGPT and Gemini. Introduced in 2017, it revolutionized how AI
processes information. The same architecture is used for training on
massive datasets and for inference to generate outputs. Here we use
GPT-2 (small), simpler than newer ones but perfect for learning the
fundamentals.

------------------------------------------------------------------------

### 2. How Transformers Work?

Transformers aren't magic---they build text step by step by asking:

**"What is the most probable next word that will follow this input?"**

**Example (from visualization):** Input text: `Mein Name ist Johannes` →
model predicts next token (e.g., `burg`).

------------------------------------------------------------------------

### 3. Transformer Architecture

Transformer has three main parts:

1.  **Embeddings** turn text into numbers.
2.  **Transformer blocks** mix information with Self-Attention and refine it
    with an MLP.
3.  **Probabilities** determine the likelihood of each next token.

```mermaid
flowchart LR
    A[Text] --> B[Embeddings]
    B --> C[Transformer Blocks]
    C --> D[Probabilities]
```

------------------------------------------------------------------------

## II. Turning Text into Numbers (Input Representation)

### 4. Embedding

Before a Transformer can process text, it must convert it into numbers. This happens in two steps:

**Step 1: Tokenization** — The text is split into smaller units called **tokens**. These aren't always whole words; common subwords or characters get their own tokens.

**Step 2: Token → Vector** — Each token ID is used to look up a corresponding **embedding vector** from a learned embedding table.

**Example:** The sentence `Mein Name ist Johannes` becomes:

| Token | Token ID | Embedding (768 numbers) |
|-------|----------|-------------------------|
| `Me` | 5308 | [0.12, -0.45, 0.78, ..., -0.22] |
| `in` | 259 | [0.08, -0.41, 0.65, ..., -0.19] |
| `Name` | 6530 | [0.15, 0.23, -0.54, ..., 0.31] |
| `is` | 318 | [-0.05, 0.18, 0.42, ..., 0.09] |
| `t` | 83 | [0.22, -0.33, 0.11, ..., -0.45] |
| `Johannes` | 38579 | [0.07, -0.28, 0.89, ..., 0.14] |

Notice that "Mein" was split into `Me` + `in` and "ist" into `is` + `t`. The tokenizer breaks words into subwords based on what it learned during training.

**How does Token ID connect to the embedding vector?**

The Token ID is simply an **index into a lookup table**. GPT-2 has a vocabulary of 50,257 tokens. During training, it learns an **embedding table** — a matrix of size `50,257 × 768`. Each row is a unique vector for one token.

```
Embedding Table (learned during training):
┌─────────────────────────────────────────────┐
│ Row 0:     [0.02, -0.15, ..., 0.33]         │
│ Row 1:     [0.11,  0.08, ..., -0.21]        │
│ ...                                         │
│ Row 5308:  [0.12, -0.45, ..., -0.22]        │  ← "Me" looks up this row
│ ...                                         │
│ Row 50256: [...]                            │
└─────────────────────────────────────────────┘
```

That's it — a simple table lookup. The magic is that these 768 numbers per token were **learned during training** to capture useful meaning.

**What might these 768 dimensions represent?**

Each dimension can encode some aspect of a token's meaning. While the model learns these automatically and they're not always human-interpretable, you can imagine dimensions capturing things like:

| Dimension | Possible meaning | "King" | "Queen" | "Apple" |
|-----------|------------------|--------|---------|---------|
| dim 1 | Royalty | 0.9 | 0.85 | -0.1 |
| dim 2 | Gender (masc→fem) | -0.7 | 0.8 | 0.0 |
| dim 3 | Edible | -0.2 | -0.2 | 0.95 |
| dim 4 | Abstract concept | 0.3 | 0.3 | -0.8 |
| ... | ... | ... | ... | ... |

This is why vector math works on embeddings: `King - Man + Woman ≈ Queen` — the dimensions encoding gender shift while royalty stays intact.

------------------------------------------------------------------------

### 5. Positional Encoding

Word order matters in language. Consider:
- "Johannes ist Name" — doesn't make sense
- "Name ist Johannes" — grammatically correct

Without position information, the model would see the same set of tokens and wouldn't know which comes first. **Positional encoding** solves this by giving each token information about its place in the sequence.

**How it works:** GPT-2 has a second lookup table — a **positional embedding table** of size `1024 × 768` (max 1024 positions). Each position gets its own 768-number vector, which is **added** to the token embedding.

**Example:** For `Mein Name ist Johannes`:

| Position | Token | Token ID | Token Embedding | + | Position Embedding | = | Final Embedding |
|----------|-------|----------|-----------------|---|-------------------|---|-----------------|
| 0 | `Me` | 5308 | [0.12, -0.45, ...] | + | [0.01, 0.02, ...] | = | [0.13, -0.43, ...] |
| 1 | `in` | 259 | [0.08, -0.41, ...] | + | [0.03, -0.01, ...] | = | [0.11, -0.42, ...] |
| 2 | `Name` | 6530 | [0.15, 0.23, ...] | + | [0.02, 0.05, ...] | = | [0.17, 0.28, ...] |
| 3 | `is` | 318 | [-0.05, 0.18, ...] | + | [-0.01, 0.03, ...] | = | [-0.06, 0.21, ...] |
| 4 | `t` | 83 | [0.22, -0.33, ...] | + | [0.04, -0.02, ...] | = | [0.26, -0.35, ...] |
| 5 | `Johannes` | 38579 | [0.07, -0.28, ...] | + | [0.02, 0.01, ...] | = | [0.09, -0.27, ...] |

Now each token's embedding contains both **what** the token is and **where** it appears.

**What might positional dimensions represent?**

Like token embeddings, positional embeddings are learned during training. Each dimension might encode patterns about position:

| Dimension | Possible meaning | Pos 0 | Pos 1 | Pos 5 | Pos 1023 |
|-----------|------------------|-------|-------|-------|----------|
| dim 1 | Start of sequence | 0.95 | 0.6 | 0.1 | -0.3 |
| dim 2 | Even/odd position | 0.8 | -0.8 | -0.8 | -0.8 |
| dim 3 | Early vs late | 0.9 | 0.85 | 0.5 | -0.9 |
| dim 4 | Sentence boundary | 0.7 | 0.1 | 0.0 | 0.0 |
| ... | ... | ... | ... | ... | ... |

The model learns which positional patterns matter for language — like "the first word is often a subject" or "words near each other are likely related."

Newer models may use other methods like **RoPE** (Rotary Position Embedding), which encodes position by rotating vectors rather than adding to them.

------------------------------------------------------------------------

## III. The Core Engine (Transformer Blocks)

**Goal:** Contextualize token representations.

**This entire section is one repeated unit:**
```
LayerNorm
→ Self-Attention (Q/K/V, masking, softmax)
→ Residual
→ LayerNorm
→ MLP
→ Residual
```

**This is the heart of the model.**

### 6. Repetitive Transformer Blocks

A Transformer block is the main unit of processing in the model. It has
two parts:

Multi-head self-attention -- lets tokens share information\
MLP -- refines each token's details

Models stack many blocks so token representations become richer as they
pass through. GPT-2 (small) has 12 of them.

**Example (from visualization):** The same block is repeated 12 times
before producing final logits.

------------------------------------------------------------------------

### 7. Multi-Head Self Attention

Self-attention lets the model decide which parts of the input are most
relevant to each token. This helps it capture meaning and relationships,
even between far-apart words.

In multi-head form, the model runs several attention processes in
parallel, each focusing on different patterns in the text.

**Example (from visualization):** Head 6 of 12 shown; different colored
attention lines across tokens.

------------------------------------------------------------------------

### 8. Query, Key, Value

To perform self-attention, each token's embedding is transformed into
three new embeddings---Query, Key, and Value. This transformation is
done by applying different weights and biases to each token embedding.
These parameters (weights and biases), are optimized through training.

Once created, Queries compare with Keys to measure relevance, and this
relevance is used to weight the Values.

**Example (from visualization):** Each token → Q(64), K(64), V(64) per
head.

------------------------------------------------------------------------

### 9. Multi-head

After creating Q, K, and V embeddings, the model splits them into
several heads (12 in GPT-2 small). Each head works with its own smaller
set of Q/K/V, focusing on different patterns in the text---like grammar,
meaning, or long-range links.

Multiple heads let the model learn many kinds of relationships in
parallel, making its understanding richer.

**Example (from visualization):** 12 heads × 64 dimensions = 768 total
dimensions.

------------------------------------------------------------------------

### 10. Masked Self Attention

In each head, the model decides how much each token focuses on others:

Dot Product -- Multiply matching numbers in Query/Key vectors, sum to
get attention scores.\
Mask -- Hide future tokens so it can't peek ahead.\
Softmax -- Convert scores to probabilities, each row summing to 1,
showing focus on earlier tokens.

**Example (from visualization):** Dot products like -10.5 to 134.1 →
masked → softmax → attention weights.

------------------------------------------------------------------------

### 11. Attention Output & Concatenation

Each head multiplies its attention scores with the Value embeddings to
produce its attention output---a refined representation of each token
after considering context.

GPT-2 (small) has 12 such outputs, which are concatenated to form a
single vector of the original size (768 numbers).

**Example (from visualization):** 12 outputs concatenated → vector(768).

------------------------------------------------------------------------

### 12. MLP (Multi-Layer Perceptron)

The attention output goes through an MLP to refine token
representations. A Linear layer changes embedding values and size using
learned weights and bias, then a non-linear activation decides how much
each value passes.

Many activation types exist; GPT-2 uses GELU, which lets small values
pass partially and large values pass fully, helping capture both subtle
and strong patterns.

**Example (from visualization):** Linear(768 → 3072) → GELU →
Linear(3072 → 768).

------------------------------------------------------------------------

## IV. Stability & Regularization (Why it trains properly)

**Goal:** Understand architectural helpers that ensure stable training.

**These aren't part of the core "math of attention," but they are critical for:**
- Training stability
- Generalization
- Preventing exploding activations

### 13. Layer Normalization

Layer Normalization helps stabilize both training and inference by
adjusting input numbers so their mean and variance stay consistent. This
makes the model less sensitive to its starting weights and helps it
learn more effectively. In GPT-2, it's applied before self-attention,
before the MLP, and once more before the final output.

**Example (from visualization):** LayerNorm applied before attention and
before MLP inside each block.

------------------------------------------------------------------------

### 14. Dropout

During training, dropout randomly turns off some connections between
numbers so the model doesn't overfit to specific patterns. This helps it
learn features that generalize better. GPT-2 uses it, but newer LLMs
often skip it because they train on huge datasets and overfitting is
less of a problem. In inference, dropout is turned off.

**Example (from visualization):** Dropout shown inside block during
training; disabled during inference.

------------------------------------------------------------------------

## V. Producing the Next-Token Scores (Output Layer)

**Goal:** Convert final token representation into probability distribution.

**This phase converts:**
```
Final token embedding → 50,257 logits → probability distribution
```

**Now the model knows how likely each token is.**

### 15. Output Logit

After all Transformer blocks, the last token's output embedding,
enriched with context from all previous tokens, is multiplied by learned
weights in a final layer.

This produces logits, 50,257 numbers---one for each token in GPT-2's
vocabulary---that indicate how likely each token is to come next.

**Example (from visualization):** Logits include values like -77.82,
-78.36, -78.38, etc.

------------------------------------------------------------------------

### 16. Probabilities

Logits are just raw scores. To make them easier to interpret, we convert
them into probabilities between 0 and 1, where all add up to 1. This
tells us the likelihood of each token being the next word.

Instead of always picking the highest-probability token, we can use
different selection strategies to balance safety and creativity in the
generated text.

**Example (from visualization):** `burg` → 37.70%\
`von` → 19.28%\
`de` → 18.80%

------------------------------------------------------------------------

## VI. Sampling Strategy (Decision Layer)

**Goal:** Determine how tokens are selected from probability distribution.

**This is not part of the model itself — it's a decoding strategy layered on top.**

**This determines:**
- Deterministic vs creative
- Conservative vs diverse outputs

### 17. Temperature

Temperature works by scaling the logits before turning them into
probabilities. A low temperature (e.g., 0.2) makes large logits even
larger and small ones smaller, favoring the highest-scoring tokens and
leading to more predictable choices. A high temperature (e.g., 1.0 or
above) flattens the differences, making less likely tokens more
competitive and leading to more creative outputs.

**Example (from visualization):** logit -77.82 ÷ 0.8 → scaled logit
-97.28 before softmax.

------------------------------------------------------------------------

### 18. Top-K Sampling

Top-K sampling filters the probability distribution by only keeping the
K most likely tokens. This prevents the model from sampling very unlikely
tokens that would produce nonsensical output. It works by setting all
probabilities outside the top-K to zero, then renormalizing so they sum
to 1 again.

**Example (from visualization):** If K = 10, only the 10 highest-probability
tokens are candidates; others are excluded.

------------------------------------------------------------------------

### 19. Top-P (Nucleus) Sampling

Top-P sampling uses a different strategy: it keeps the smallest set of
tokens whose cumulative probability exceeds threshold P (commonly 0.9).
This adapts to the shape of the probability distribution---when confidence
is high, fewer tokens are selected; when uncertain, more tokens are included.

**Example (from visualization):** Select tokens until cumulative probability
reaches 90%, regardless of how many that includes.

------------------------------------------------------------------------

### 20. Combining Strategies

In practice, temperature, top-K, and top-P are often used together to
balance diversity and coherence. Temperature reshapes the distribution,
while top-K and top-P act as safety filters that prevent sampling from
the tail of unlikely tokens.

**Example (from visualization):** Apply temperature scaling first, then
filter with top-K and/or top-P to finalize token selection.

------------------------------------------------------------------------

## Summary: The Complete Pipeline

1. **Input Representation:** Text → Tokens → Embeddings + Positional Encoding
2. **Context Processing:** 12 stacked Transformer blocks, each with Self-Attention + MLP
3. **Stabilization:** LayerNorm and Dropout ensure stable learning
4. **Scoring:** Output layer produces logits; softmax converts to probabilities
5. **Decoding:** Temperature, top-K, top-P guide final token selection

---

## Mental Model: 5 Essential Layers

If teaching executives or engineers, compress to:

1. **Objective:** Predict next token.
2. **Representation:** Embedding + Positional Encoding.
3. **Context Engine:** Repeated blocks:
   - Self-attention
   - MLP
   - Residuals
   - LayerNorm
4. **Scoring:** Linear layer → logits → softmax.
5. **Selection:** Temperature / top-k / top-p sampling.
