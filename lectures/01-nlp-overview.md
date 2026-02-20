# Session 1: NLP Overview

**Course:** Stanford LLM (ICME)

---

## NLP Tasks Overview

```mermaid
flowchart LR
    subgraph tasks["Three Core NLP Task Types"]
        direction TB
        C["Classification"]
        M["Multi-Classification"]
        G["Generation"]
    end
    
    C --> C1["1 input → 1 label"]
    M --> M1["1 input → N labels"]
    G --> G1["1 input → new text"]
```

---

### 1. Classification

```mermaid
flowchart LR
    A["📄 Input Text<br/>'I love this movie!'"] --> B["🤖 Model"]
    B --> C["🏷️ Label<br/>POSITIVE"]
    
    style A fill:#e1f5fe
    style B fill:#fff3e0
    style C fill:#e8f5e9
```

A single input produces a single categorical output.

**Examples:**

```mermaid
mindmap
  root((Classification))
    Sentiment
      Positive
      Negative
      Neutral
    Intent Detection
      Book flight
      Check weather
      Play music
    Language Detection
      English
      Spanish
      French
    Topic Modeling
      Sports
      Politics
      Technology
```

---

### 2. Multi-Classification (Sequence Labeling)

```mermaid
flowchart LR
    A["📄 Input Text<br/>'John works at Google'"] --> B["🤖 Model"]
    B --> C["🏷️ Multiple Labels<br/>PERSON | VERB | PREP | ORG"]
    
    style A fill:#e1f5fe
    style B fill:#fff3e0
    style C fill:#e8f5e9
```

Each token or segment in the input receives its own label.

**Named Entity Recognition (NER) Example:**

```mermaid
flowchart TB
    subgraph input["Input Sentence"]
        W1["John"] --> W2["works"] --> W3["at"] --> W4["Google"] --> W5["in"] --> W6["California"]
    end
    
    subgraph output["NER Labels"]
        L1["PERSON"] 
        L2["O"]
        L3["O"]
        L4["ORG"]
        L5["O"]
        L6["LOCATION"]
    end
    
    W1 -.-> L1
    W2 -.-> L2
    W3 -.-> L3
    W4 -.-> L4
    W5 -.-> L5
    W6 -.-> L6
    
    style L1 fill:#ffcdd2
    style L4 fill:#c8e6c9
    style L6 fill:#bbdefb
```

**Part of Speech (PoS) Tagging Example:**

```mermaid
flowchart TB
    subgraph sent["The cat sat on the mat"]
        T1["The"] --> T2["cat"] --> T3["sat"] --> T4["on"] --> T5["the"] --> T6["mat"]
    end
    
    subgraph pos["PoS Tags"]
        P1["DET"]
        P2["NOUN"]
        P3["VERB"]
        P4["PREP"]
        P5["DET"]
        P6["NOUN"]
    end
    
    T1 -.-> P1
    T2 -.-> P2
    T3 -.-> P3
    T4 -.-> P4
    T5 -.-> P5
    T6 -.-> P6
```

---

### 3. Generation

```mermaid
flowchart LR
    A["📄 Input Text<br/>'Translate to French:<br/>Hello world'"] --> B["🤖 Model"]
    B --> C["📝 Output Text<br/>'Bonjour le monde'"]
    
    style A fill:#e1f5fe
    style B fill:#fff3e0
    style C fill:#f3e5f5
```

The model generates new text based on the input.

**Generation Task Types:**

```mermaid
flowchart TB
    subgraph Generation Tasks
        MT["Machine Translation"]
        QA["Question Answering"]
        SUM["Summarization"]
        NLG["Text Generation"]
    end
    
    MT --> MT1["English → French"]
    QA --> QA1["Question → Answer"]
    SUM --> SUM1["Long doc → Short summary"]
    NLG --> NLG1["Prompt → Completion"]
```

---

## NLP Landscape

### Model Architecture Evolution

```mermaid
timeline
    title Evolution of NLP Architectures
    section Pre-Transformer
        2013 : Word2Vec
        2014 : GloVe
        2014 : LSTM/GRU
    section Transformer Era
        2017 : Transformer
        2018 : BERT (Encoder)
        2018 : GPT (Decoder)
        2019 : T5 (Enc-Dec)
    section LLM Era
        2022 : ChatGPT
        2023 : LLaMA
        2023 : GPT-4
```

### Transformer Architecture Types

```mermaid
flowchart TB
    subgraph arch["Transformer Architectures"]
        direction LR
        
        subgraph enc["Encoder-Only"]
            BERT["BERT"]
        end
        
        subgraph dec["Decoder-Only"]
            GPT["GPT"]
            LLAMA["LLaMA"]
        end
        
        subgraph encdec["Encoder-Decoder"]
            T5["T5"]
        end
    end
    
    enc --> U1["Best for:<br/>Classification<br/>NER, PoS"]
    dec --> U2["Best for:<br/>Text Generation<br/>Completion"]
    encdec --> U3["Best for:<br/>Translation<br/>Summarization"]
    
    style enc fill:#bbdefb
    style dec fill:#c8e6c9
    style encdec fill:#fff9c4
```

### Prompting Techniques

```mermaid
flowchart TB
    subgraph cot["Chain of Thought (CoT)"]
        direction LR
        Q1["Question"] --> S1["Step 1"] --> S2["Step 2"] --> S3["Step 3"] --> A1["Answer"]
    end
    
    subgraph tot["Tree of Thought (ToT)"]
        Q2["Question"] --> B1["Branch 1"]
        Q2 --> B2["Branch 2"]
        Q2 --> B3["Branch 3"]
        B1 --> A2["Answer 1"]
        B2 --> A3["Answer 2"]
        B3 --> A4["Answer 3"]
    end
    
    subgraph sc["Self-Consistency (SC)"]
        Q3["Question"] --> P1["Path 1 → Ans A"]
        Q3 --> P2["Path 2 → Ans A"]
        Q3 --> P3["Path 3 → Ans B"]
        P1 --> V["Vote: A wins"]
        P2 --> V
        P3 --> V
    end
```

### RAG (Retrieval-Augmented Generation)

```mermaid
flowchart LR
    Q["User Query"] --> R["Retriever"]
    R --> DB[("Document<br/>Database")]
    DB --> D["Retrieved<br/>Documents"]
    D --> G["Generator<br/>(LLM)"]
    Q --> G
    G --> A["Grounded<br/>Answer"]
    
    style DB fill:#e1f5fe
    style G fill:#fff3e0
    style A fill:#e8f5e9
```

---

## Training Strategies

### The Modern LLM Training Pipeline

```mermaid
flowchart TB
    subgraph stage1["Stage 1: Pre-training"]
        PT["Massive Text Corpus"] --> BASE["Base Model"]
        PT1["MLM / Next Token Prediction"]
    end
    
    subgraph stage2["Stage 2: Fine-tuning"]
        BASE --> SFT["SFT<br/>Supervised Fine-Tuning"]
        SFT --> INST["Instruction-Tuned Model"]
        
        PEFT["PEFT Methods"]
        PEFT --> |"LoRA, Adapters"| SFT
    end
    
    subgraph stage3["Stage 3: Alignment"]
        INST --> ALIGN["Alignment"]
        ALIGN --> FINAL["Aligned Model"]
        
        subgraph methods["Alignment Methods"]
            RLHF["RLHF"]
            DPO["DPO"]
        end
        methods --> ALIGN
    end
    
    style stage1 fill:#e3f2fd
    style stage2 fill:#fff3e0
    style stage3 fill:#f3e5f5
```

### RLHF Pipeline (Detailed)

```mermaid
flowchart TB
    subgraph rlhf["RLHF: Reinforcement Learning from Human Feedback"]
        direction TB
        
        SFT2["SFT Model"] --> GEN["Generate<br/>Responses"]
        GEN --> HUMAN["Human<br/>Rankings"]
        HUMAN --> RM["Train Reward<br/>Model (RM)"]
        
        SFT2 --> PPO2["PPO Training"]
        RM --> |"Reward Signal"| PPO2
        PPO2 --> FINAL2["Aligned<br/>Model"]
    end
    
    style HUMAN fill:#ffcdd2
    style RM fill:#fff9c4
    style FINAL2 fill:#c8e6c9
```

### DPO vs RLHF

```mermaid
flowchart LR
    subgraph rlhf_path["RLHF Path"]
        direction TB
        R1["1. Train Reward Model"] --> R2["2. Run PPO"] --> R3["3. Aligned Model"]
    end
    
    subgraph dpo_path["DPO Path (Simpler)"]
        direction TB
        D1["1. Preference Data"] --> D2["2. Direct Optimization"] --> D3["3. Aligned Model"]
    end
    
    rlhf_path --> |"Complex,<br/>Unstable"| OUT["Result"]
    dpo_path --> |"Simpler,<br/>Stable"| OUT
    
    style dpo_path fill:#c8e6c9
```

---

## Common Tasks (Visual Reference)

```mermaid
mindmap
  root((NLP Tasks))
    Understanding
      NER
        Named Entity Recognition
      PoS
        Part of Speech
      MLM
        Masked Language Modeling
      NSP
        Next Sentence Prediction
    Generation
      MT
        Machine Translation
      QA
        Question Answering
      NLG
        Natural Language Generation
```

---

## Benchmark Datasets

```mermaid
flowchart TB
    subgraph glue["GLUE Benchmark"]
        direction LR
        MNLI["MNLI<br/>Natural Language<br/>Inference"]
        WNLI["WNLI<br/>Winograd NLI"]
        MRPC["MRPC<br/>Paraphrase"]
    end
    
    subgraph other["Other Key Datasets"]
        SQUAD["SQuAD<br/>Question Answering"]
        C4["C4<br/>Pre-training Corpus"]
    end
    
    subgraph usage["Dataset Usage"]
        PRE["Pre-training"]
        EVAL["Evaluation"]
    end
    
    C4 --> PRE
    glue --> EVAL
    SQUAD --> EVAL
    
    style glue fill:#e1f5fe
    style other fill:#fff3e0
```

---

## Evaluation Metrics

### Metrics by Task Type

```mermaid
flowchart TB
    subgraph metrics["Evaluation Metrics"]
        direction TB
        
        subgraph class_metrics["Classification"]
            F1["F1 Score<br/>Precision + Recall"]
        end
        
        subgraph gen_metrics["Generation"]
            BLEU["BLEU<br/>Translation"]
            ROUGE["ROUGE<br/>Summarization"]
            PPL["Perplexity<br/>Language Modeling"]
            METEOR["METEOR<br/>Translation"]
        end
        
        subgraph speech_metrics["Speech"]
            WER["WER<br/>Word Error Rate"]
        end
        
        subgraph llm_metrics["LLM Evaluation"]
            LAAJ["LLM-as-a-Judge"]
        end
    end
    
    style class_metrics fill:#bbdefb
    style gen_metrics fill:#c8e6c9
    style speech_metrics fill:#fff9c4
    style llm_metrics fill:#f3e5f5
```

### Understanding Key Metrics

```mermaid
flowchart LR
    subgraph f1["F1 Score"]
        P["Precision"] --> F["F1 = 2×(P×R)/(P+R)"]
        R["Recall"] --> F
    end
    
    subgraph ppl["Perplexity"]
        PPL2["Lower = Better<br/>Model is less 'surprised'<br/>by the text"]
    end
    
    subgraph bleu_rouge["BLEU & ROUGE"]
        BL["BLEU: n-gram precision<br/>(vs reference)"]
        RO["ROUGE: n-gram recall<br/>(vs reference)"]
    end
```

---

## Summary: The Big Picture

```mermaid
flowchart TB
    subgraph evolution["NLP Evolution"]
        direction LR
        TRAD["Traditional<br/>LSTM, GRU"] --> TRANS["Transformers<br/>BERT, GPT"] --> LLM["Large LLMs<br/>ChatGPT, LLaMA"]
    end
    
    subgraph pipeline["Modern Training Pipeline"]
        direction LR
        PRE2["Pre-train<br/>(MLM, CLM)"] --> FT["Fine-tune<br/>(SFT, PEFT)"] --> AL["Align<br/>(RLHF, DPO)"]
    end
    
    subgraph eval["Evaluation"]
        direction LR
        CLASS2["Classification<br/>→ F1"] 
        GEN2["Generation<br/>→ BLEU, ROUGE"]
        OPEN["Open-ended<br/>→ LLM-as-Judge"]
    end
    
    evolution --> pipeline --> eval
    
    style evolution fill:#e3f2fd
    style pipeline fill:#fff3e0
    style eval fill:#e8f5e9
```

---

## Quick Reference Card

```mermaid
flowchart TB
    subgraph models["Models"]
        M1["BERT = Encoder"]
        M2["GPT/LLaMA = Decoder"]
        M3["T5 = Encoder-Decoder"]
    end
    
    subgraph training["Training"]
        T1["SFT = Supervised"]
        T2["PEFT = Efficient"]
        T3["RLHF/DPO = Alignment"]
    end
    
    subgraph techniques["Techniques"]
        TE1["CoT = Step-by-step"]
        TE2["RAG = Retrieval+Gen"]
        TE3["SC = Vote on paths"]
    end
    
    subgraph metrics2["Metrics"]
        ME1["F1 = Classification"]
        ME2["BLEU = Translation"]
        ME3["PPL = LM Quality"]
    end
```
