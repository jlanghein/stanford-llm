# Session 2: Tokenization

**Course:** Stanford LLM (ICME)

---

## Table of Contents

- [Overview: What is Tokenization?](#overview-what-is-tokenization)
- [The Tokenization Spectrum](#the-tokenization-spectrum)
- [Tokenization Methods](#tokenization-methods)
  - [1. Character-Level Tokenization](#1-character-level-tokenization)
  - [2. Word-Level Tokenization](#2-word-level-tokenization)
  - [3. Subword-Level Tokenization (BPE / WordPiece)](#3-subword-level-tokenization-bpe--wordpiece)
  - [4. Arbitrary Tokenization](#4-arbitrary-tokenization)
- [Visual Example: "A cute teddy bear is reading."](#visual-example-a-cute-teddy-bear-is-reading)
- [Comparison Table](#comparison-table)
- [Trade-offs Visualization](#trade-offs-visualization)
- [Why Subword Tokenization Wins](#why-subword-tokenization-wins)
- [Key Algorithms](#key-algorithms)
  - [Byte Pair Encoding (BPE)](#byte-pair-encoding-bpe)
  - [WordPiece](#wordpiece)
  - [SentencePiece](#sentencepiece)
- [The OOV Problem](#the-oov-problem)
- [Summary: The Big Picture](#summary-the-big-picture)
- [Quick Reference Card](#quick-reference-card)

---

## Overview: What is Tokenization?

[Back to Table of Contents](#table-of-contents)

Tokenization is the process of breaking text into smaller units (tokens) that a model can process.

```mermaid
flowchart LR
    A["Raw Text<br/>'Hello world!'"] --> B["Tokenizer"]
    B --> C["Tokens<br/>[Hello] [world] [!]"]
    C --> D["Token IDs<br/>[4521, 893, 2]"]
    D --> E["Model"]
    
    style A fill:#e1f5fe
    style B fill:#fff3e0
    style C fill:#e8f5e9
    style D fill:#f3e5f5
    style E fill:#ffcdd2
```

**Why does tokenization matter?**

```mermaid
flowchart TB
    subgraph impact["Tokenization Impacts"]
        direction TB
        V["Vocabulary Size<br/>How many unique tokens?"]
        S["Sequence Length<br/>How many tokens per input?"]
        O["OOV Handling<br/>What about unknown words?"]
        G["Generalization<br/>Can model handle new words?"]
    end
    
    V --> TRADE["All connected via trade-offs"]
    S --> TRADE
    O --> TRADE
    G --> TRADE
    
    style impact fill:#e3f2fd
```

---

## The Tokenization Spectrum

[Back to Table of Contents](#table-of-contents)

```mermaid
flowchart LR
    subgraph spectrum["Tokenization Spectrum"]
        direction LR
        CHAR["Character<br/>Level"] --> SUB["Subword<br/>Level"] --> WORD["Word<br/>Level"] --> ARB["Arbitrary<br/>Chunks"]
    end
    
    CHAR --> C1["Finest<br/>granularity"]
    ARB --> C2["Coarsest<br/>granularity"]
    
    style CHAR fill:#bbdefb
    style SUB fill:#c8e6c9
    style WORD fill:#fff9c4
    style ARB fill:#ffcdd2
```

```mermaid
flowchart TB
    subgraph tradeoff["The Fundamental Trade-off"]
        direction LR
        
        LEFT["Smaller Tokens"]
        RIGHT["Larger Tokens"]
        
        LEFT --> L1["+ Lower OOV risk"]
        LEFT --> L2["+ Better generalization"]
        LEFT --> L3["- Longer sequences"]
        LEFT --> L4["- Slower computation"]
        
        RIGHT --> R1["+ Shorter sequences"]
        RIGHT --> R2["+ Faster computation"]
        RIGHT --> R3["- Higher OOV risk"]
        RIGHT --> R4["- Worse generalization"]
    end
    
    style LEFT fill:#bbdefb
    style RIGHT fill:#ffcdd2
```

---

## Tokenization Methods

### 1. Character-Level Tokenization

[Back to Table of Contents](#table-of-contents)

**Concept:** Split text into individual characters (including spaces).

```mermaid
flowchart TB
    subgraph char_tok["Character-Level Tokenization"]
        direction LR
        INPUT["'cute'"] --> OUTPUT["[c] [u] [t] [e]"]
    end
    
    subgraph props["Properties"]
        direction TB
        P1["Vocabulary: ~100-300 characters"]
        P2["Sequence Length: Very long"]
        P3["OOV Risk: Minimal"]
    end
    
    style char_tok fill:#bbdefb
```

**Example with full sentence:**

```
Input:  "A cute teddy bear"
Tokens: [A] [ ] [c] [u] [t] [e] [ ] [t] [e] [d] [d] [y] [ ] [b] [e] [a] [r]
Count:  17 tokens
```

```mermaid
flowchart LR
    subgraph char_pros["Pros"]
        CP1["Almost zero OOV"]
        CP2["Robust to typos"]
        CP3["Handles any language"]
        CP4["Tiny vocabulary"]
    end
    
    subgraph char_cons["Cons"]
        CC1["Very long sequences"]
        CC2["Slow computation"]
        CC3["Hard to learn meaning"]
        CC4["Context window wasted"]
    end
    
    style char_pros fill:#c8e6c9
    style char_cons fill:#ffcdd2
```

**Think:** Raw symbols - maximum flexibility, minimum efficiency

---

### 2. Word-Level Tokenization

[Back to Table of Contents](#table-of-contents)

**Concept:** Split text on whitespace and punctuation.

```mermaid
flowchart TB
    subgraph word_tok["Word-Level Tokenization"]
        direction LR
        INPUT2["'A cute teddy bear is reading.'"] --> OUTPUT2["[A] [cute] [teddy] [bear] [is] [reading] [.]"]
    end
    
    subgraph props2["Properties"]
        direction TB
        P4["Vocabulary: 50K-500K+ words"]
        P5["Sequence Length: Short"]
        P6["OOV Risk: High"]
    end
    
    style word_tok fill:#fff9c4
```

**Example:**

```
Input:  "A cute teddy bear is reading."
Tokens: [A] [cute] [teddy] [bear] [is] [reading] [.]
Count:  7 tokens
```

```mermaid
flowchart LR
    subgraph word_pros["Pros"]
        WP1["Simple to implement"]
        WP2["Human interpretable"]
        WP3["Short sequences"]
        WP4["Clear word boundaries"]
    end
    
    subgraph word_cons["Cons"]
        WC1["OOV problem"]
        WC2["No morphology awareness"]
        WC3["Huge vocabulary needed"]
        WC4["Cannot handle 'readings' if only saw 'reading'"]
    end
    
    style word_pros fill:#c8e6c9
    style word_cons fill:#ffcdd2
```

**The OOV Problem Illustrated:**

```mermaid
flowchart TB
    subgraph oov_problem["The OOV Problem"]
        TRAIN["Training Vocabulary:<br/>'read', 'reading', 'reader'"]
        TEST["New Word:<br/>'readings'"]
        RESULT["Result: [UNK] token<br/>All meaning lost!"]
        
        TRAIN --> TEST
        TEST --> RESULT
    end
    
    style RESULT fill:#ffcdd2
```

**Think:** Split on spaces, done - simple but fragile

---

### 3. Subword-Level Tokenization (BPE / WordPiece)

[Back to Table of Contents](#table-of-contents)

**Concept:** Learn frequently occurring subword units from data.

```mermaid
flowchart TB
    subgraph subword_tok["Subword-Level Tokenization"]
        direction LR
        INPUT3["'reading'"] --> OUTPUT3["[read] [##ing]"]
    end
    
    subgraph props3["Properties"]
        direction TB
        P7["Vocabulary: 30K-100K tokens"]
        P8["Sequence Length: Medium"]
        P9["OOV Risk: Low"]
    end
    
    style subword_tok fill:#c8e6c9
```

**Example:**

```
Input:  "A cute teddy bear is reading."
Tokens: [A] [cute] [ted] [##dy] [bear] [is] [read] [##ing] [.]
   or:  [A] [cute] [teddy] [bear] [is] [read] [##ing] [.]
Count:  ~8-9 tokens
```

The `##` prefix indicates a continuation token (used by WordPiece).

```mermaid
flowchart TB
    subgraph morphology["Morphology Awareness"]
        direction TB
        
        BASE["[read]"] --> V1["reading → [read][##ing]"]
        BASE --> V2["reader → [read][##er]"]
        BASE --> V3["reads → [read][##s]"]
        BASE --> V4["reread → [re][##read]"]
    end
    
    BENEFIT["Same base token 'read'<br/>shared across all variations!"]
    
    morphology --> BENEFIT
    
    style BENEFIT fill:#c8e6c9
```

```mermaid
flowchart LR
    subgraph sub_pros["Pros"]
        SP1["Reuses prefixes/suffixes"]
        SP2["Learned from data"]
        SP3["Low OOV risk"]
        SP4["Efficient vocab/length trade-off"]
    end
    
    subgraph sub_cons["Cons"]
        SC1["Slight OOV risk remains"]
        SC2["Less human-interpretable"]
        SC3["Requires training tokenizer"]
    end
    
    style sub_pros fill:#c8e6c9
    style sub_cons fill:#ffcdd2
```

**This is what modern LLMs use!**

**Think:** Learned reusable components - the sweet spot

---

### 4. Arbitrary Tokenization

[Back to Table of Contents](#table-of-contents)

**Concept:** Tokens can be any frequently occurring chunk (learned purely from statistics).

```mermaid
flowchart TB
    subgraph arb_tok["Arbitrary Tokenization"]
        direction LR
        INPUT4["'teddy bear'"] --> OUTPUT4["[teddy bear]<br/>(single token!)"]
    end
    
    subgraph props4["Properties"]
        direction TB
        P10["Vocabulary: Potentially huge"]
        P11["Sequence Length: Very short"]
        P12["OOV Risk: Medium"]
    end
    
    style arb_tok fill:#ffcdd2
```

**Example:**

```
Input:  "A cute teddy bear is reading."
Tokens: [A] [cute] [teddy bear] [is] [reading] [.]
Count:  6 tokens (phrase "teddy bear" merged!)
```

```mermaid
flowchart LR
    subgraph arb_pros["Pros"]
        AP1["Very compact"]
        AP2["Captures phrases"]
        AP3["Can encode idioms"]
    end
    
    subgraph arb_cons["Cons"]
        AC1["Huge vocabulary"]
        AC2["Generalization suffers"]
        AC3["Overfits to training data"]
    end
    
    style arb_pros fill:#c8e6c9
    style arb_cons fill:#ffcdd2
```

**Think:** Data-driven chunking without linguistic rules

---

## Visual Example: "A cute teddy bear is reading."

[Back to Table of Contents](#table-of-contents)

```mermaid
flowchart TB
    subgraph example["Same Sentence, Four Ways"]
        direction TB
        
        subgraph char_ex["Character-Level"]
            CE["[A] [ ] [c] [u] [t] [e] [ ] [t] [e] [d] [d] [y] [ ] [b] [e] [a] [r] ..."]
        end
        
        subgraph word_ex["Word-Level"]
            WE["[A] [cute] [teddy] [bear] [is] [reading] [.]"]
        end
        
        subgraph sub_ex["Subword-Level"]
            SE["[A] [cute] [ted] [##dy] [bear] [is] [read] [##ing] [.]"]
        end
        
        subgraph arb_ex["Arbitrary"]
            AE["[A] [cute] [teddy bear] [is] [reading] [.]"]
        end
    end
    
    style char_ex fill:#bbdefb
    style word_ex fill:#fff9c4
    style sub_ex fill:#c8e6c9
    style arb_ex fill:#ffcdd2
```

**Token counts comparison:**

```mermaid
xychart-beta
    title "Token Count by Method"
    x-axis ["Character", "Subword", "Word", "Arbitrary"]
    y-axis "Number of Tokens" 0 --> 35
    bar [30, 9, 7, 6]
```

---

## Comparison Table

[Back to Table of Contents](#table-of-contents)

| Method | Vocabulary Size | Sequence Length | OOV Risk | Generalization |
|--------|----------------|-----------------|----------|----------------|
| **Character** | Very small (~100) | Very long | Minimal | High |
| **Subword** | Medium (~30K-100K) | Medium | Low | High |
| **Word** | Large (~100K+) | Short | High | Low |
| **Arbitrary** | Potentially huge | Very short | Medium | Depends |

```mermaid
quadrantChart
    title Tokenization Method Trade-offs
    x-axis Low Vocabulary Size --> High Vocabulary Size
    y-axis Long Sequences --> Short Sequences
    quadrant-1 Arbitrary: compact but huge vocab
    quadrant-2 Word: simple but OOV issues
    quadrant-3 Character: robust but slow
    quadrant-4 Subword: balanced sweet spot
    Character: [0.15, 0.1]
    Subword: [0.45, 0.5]
    Word: [0.75, 0.8]
    Arbitrary: [0.9, 0.9]
```

---

## Trade-offs Visualization

[Back to Table of Contents](#table-of-contents)

```mermaid
flowchart TB
    subgraph tradeoffs["Key Trade-offs"]
        direction TB
        
        subgraph vocab_vs_seq["Vocabulary vs Sequence Length"]
            VS1["Larger vocab → Shorter sequences"]
            VS2["Smaller vocab → Longer sequences"]
        end
        
        subgraph oov_vs_gen["OOV vs Generalization"]
            OG1["Character: No OOV, max generalization"]
            OG2["Word: High OOV, poor generalization"]
            OG3["Subword: Low OOV, good generalization"]
        end
        
        subgraph compute["Computational Cost"]
            CM1["Longer sequences = More computation"]
            CM2["Transformers: O(n^2) attention"]
        end
    end
```

```mermaid
flowchart LR
    subgraph sweet_spot["The Sweet Spot"]
        direction TB
        GOAL["Goal: Balance all factors"]
        
        GOAL --> B1["Not too many tokens"]
        GOAL --> B2["Not too large vocabulary"]
        GOAL --> B3["Handle unseen words"]
        GOAL --> B4["Preserve meaning"]
        
        RESULT2["= Subword Tokenization"]
    end
    
    B1 --> RESULT2
    B2 --> RESULT2
    B3 --> RESULT2
    B4 --> RESULT2
    
    style RESULT2 fill:#c8e6c9
```

---

## Why Subword Tokenization Wins

[Back to Table of Contents](#table-of-contents)

```mermaid
flowchart TB
    subgraph why_subword["Why Modern LLMs Use Subword"]
        direction TB
        
        R1["Balances vocab size and sequence length"]
        R2["Handles rare/unseen words gracefully"]
        R3["Captures morphological patterns"]
        R4["Efficient for transformer attention"]
        R5["Learned from actual data distribution"]
    end
    
    why_subword --> MODELS["Used by GPT, BERT, LLaMA, T5, etc."]
    
    style why_subword fill:#c8e6c9
    style MODELS fill:#fff3e0
```

**Real-world examples:**

| Model | Tokenizer | Vocab Size |
|-------|-----------|------------|
| GPT-2/3/4 | BPE | ~50K |
| BERT | WordPiece | ~30K |
| T5 | SentencePiece | ~32K |
| LLaMA | SentencePiece (BPE) | ~32K |

---

## Key Algorithms

[Back to Table of Contents](#table-of-contents)

### Byte Pair Encoding (BPE)

```mermaid
flowchart TB
    subgraph bpe["BPE Algorithm"]
        direction TB
        
        S1["1. Start with character vocabulary"]
        S2["2. Count all adjacent pairs"]
        S3["3. Merge most frequent pair"]
        S4["4. Add merged token to vocab"]
        S5["5. Repeat until vocab size reached"]
        
        S1 --> S2 --> S3 --> S4 --> S5
        S5 --> |"Repeat"| S2
    end
    
    style bpe fill:#e3f2fd
```

**BPE Example:**

```mermaid
flowchart TB
    subgraph bpe_ex["BPE Merge Example"]
        direction TB
        
        INIT["Initial: [l] [o] [w] [e] [r]"]
        M1["Merge 'l'+'o' → [lo] [w] [e] [r]"]
        M2["Merge 'lo'+'w' → [low] [e] [r]"]
        M3["Merge 'e'+'r' → [low] [er]"]
        M4["Merge 'low'+'er' → [lower]"]
        
        INIT --> M1 --> M2 --> M3 --> M4
    end
```

### WordPiece

```mermaid
flowchart TB
    subgraph wordpiece["WordPiece (BERT)"]
        direction TB
        
        WP1["Similar to BPE but..."]
        WP2["Uses likelihood-based merging"]
        WP3["Prefixes continuation tokens with ##"]
        
        EX["'unhappiness' → [un] [##happi] [##ness]"]
    end
    
    style wordpiece fill:#fff3e0
```

### SentencePiece

```mermaid
flowchart TB
    subgraph sentencepiece["SentencePiece (Google)"]
        direction TB
        
        SP1["Language-agnostic"]
        SP2["Treats text as raw bytes"]
        SP3["No pre-tokenization needed"]
        SP4["Used by T5, LLaMA, etc."]
    end
    
    style sentencepiece fill:#f3e5f5
```

---

## The OOV Problem

[Back to Table of Contents](#table-of-contents)

```mermaid
flowchart TB
    subgraph oov["Out-of-Vocabulary (OOV) Problem"]
        direction TB
        
        subgraph word_oov["Word-Level: OOV Disaster"]
            WO1["Training: 'happy', 'happiness'"]
            WO2["Test: 'happily'"]
            WO3["Result: [UNK]"]
            
            WO1 --> WO2 --> WO3
        end
        
        subgraph sub_oov["Subword-Level: OOV Handled"]
            SO1["Training: [happi], [ness], [ly]"]
            SO2["Test: 'happily'"]
            SO3["Result: [happi] [ly]"]
            
            SO1 --> SO2 --> SO3
        end
    end
    
    style word_oov fill:#ffcdd2
    style sub_oov fill:#c8e6c9
```

**OOV scenarios and solutions:**

```mermaid
flowchart LR
    subgraph scenarios["Common OOV Scenarios"]
        S1["New words: 'COVID'"]
        S2["Typos: 'teh' for 'the'"]
        S3["Names: 'Schwarzenegger'"]
        S4["Technical terms: 'transformers'"]
        S5["Morphological variants: 'unhappiest'"]
    end
    
    subgraph solutions["Subword Solutions"]
        SOL1["Break into known pieces"]
        SOL2["Character fallback"]
        SOL3["Compose from parts"]
    end
    
    scenarios --> solutions
    
    style solutions fill:#c8e6c9
```

---

## Summary: The Big Picture

[Back to Table of Contents](#table-of-contents)

```mermaid
flowchart TB
    subgraph big_picture["Tokenization: The Big Picture"]
        direction TB
        
        subgraph spectrum2["The Spectrum"]
            direction LR
            FINE["Fine-grained<br/>(Character)"] --> MED["Medium<br/>(Subword)"] --> COARSE["Coarse-grained<br/>(Word/Arbitrary)"]
        end
        
        subgraph winner["The Winner"]
            W["Subword Tokenization<br/>(BPE, WordPiece, SentencePiece)"]
        end
        
        subgraph why["Why?"]
            Y1["Balanced trade-offs"]
            Y2["Low OOV risk"]
            Y3["Good generalization"]
            Y4["Efficient computation"]
        end
        
        spectrum2 --> winner
        winner --> why
    end
    
    style winner fill:#c8e6c9
```

```mermaid
flowchart LR
    subgraph pipeline["Tokenization in the LLM Pipeline"]
        direction LR
        TEXT["Raw Text"] --> TOK["Tokenizer<br/>(BPE/WordPiece)"]
        TOK --> IDS["Token IDs"]
        IDS --> EMB["Embeddings"]
        EMB --> MODEL["Transformer"]
        MODEL --> OUT["Output"]
    end
    
    style TOK fill:#fff3e0
```

---

## Quick Reference Card

[Back to Table of Contents](#table-of-contents)

```mermaid
flowchart TB
    subgraph methods_ref["Methods"]
        MR1["Character = Every symbol"]
        MR2["Word = Split on spaces"]
        MR3["Subword = Learned chunks"]
        MR4["Arbitrary = Frequency-based"]
    end
    
    subgraph algorithms_ref["Algorithms"]
        AR1["BPE = GPT family"]
        AR2["WordPiece = BERT"]
        AR3["SentencePiece = T5, LLaMA"]
    end
    
    subgraph rules_ref["Rules of Thumb"]
        RR1["Smaller tokens = Lower OOV"]
        RR2["Larger tokens = Shorter sequences"]
        RR3["Subword = Best balance"]
    end
    
    subgraph modern_ref["Modern LLMs"]
        MOD["All use subword tokenization<br/>~30K-100K vocabulary"]
    end
    
    style modern_ref fill:#c8e6c9
```

**Key Takeaway:**

> Subword tokenization (BPE/WordPiece) is the standard for modern LLMs because it achieves the optimal balance between vocabulary size, sequence length, and OOV handling.
