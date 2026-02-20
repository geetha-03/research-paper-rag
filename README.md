🧠 ML Research RAG Baseline

🔍 Overview

This project implements a CPU-based Retrieval-Augmented Generation (RAG) system indexing 100+ Machine Learning research papers across:

Natural Language Processing (NLP) Computer Vision (CV) Reinforcement Learning (RL)

The system is designed as a structured, evaluated RAG baseline that prioritizes:

Section-aware chunking Diversity-aware retrieval Two-stage summarization Advanced RAG evaluation metrics

This Space represents the CPU baseline version prior to GPU-based model upgrades.

🏗 System Architecture

User Query
    ↓
Query Expansion
    ↓    
BGE Embedding Model (BAAI/bge-small-en-v1.5)
    ↓    
FAISS Vector Index
    ↓    
Metadata Filtering (Category / Year)
    ↓    
Section-Aware Ranking
    ↓    
Diversity-Aware Chunk Selection
    ↓    
Two-Stage Summarization (FLAN-T5-large)
    ↓    
Generated Answer
    ↓    
Advanced RAG Evaluation
📚 Dataset

100+ research papers collected from arXiv Domains: NLP, CV, RL PDF text extracted and indexed References removed before embedding

Metadata stored per chunk: Paper name Category Year Section Chunk ID

🧠 Retrieval Strategy

Query Expansion: User queries are expanded to emphasize innovation and research contributions.

Section-Aware Chunking: Papers are split into structured sections: Abstract (high priority) Introduction Methods Results Conclusion

Section priority influences retrieval ranking.

Diversity-Aware Selection: Ensures retrieved chunks come from different papers to encourage cross-document synthesis.

✍️ Generation Strategy

Model: google/flan-t5-large (CPU)

Two-Stage Summarization

1️⃣ Stage 1: Summarize each retrieved chunk independently

2️⃣ Stage 2: Synthesize summaries into a final multi-document answer

This improves cross-paper abstraction compared to single-pass generation.

📊 Evaluation Framework

The system includes both heuristic and LLM-based evaluation metrics.

Heuristic Metrics:

Context Precision Diversity Score Faithfulness Proxy

LLM-Based Metrics:

Faithfulness (LLM judged) Answer Relevance Context Utilization Coherence

📈 CPU Baseline Results

Measured on diffusion-model-related queries:

Context Precision: 1.00 Diversity Score: 1.00 Faithfulness Proxy: 0.94 Faithfulness (LLM): 5 Answer Relevance: 3 Context Utilization: 3 Coherence: 4 Latency: ~60–80 seconds

🔬 Key Insight

High retrieval quality and faithfulness do not automatically imply strong cross-document synthesis. This baseline highlights the limitations of CPU-based encoder-decoder models for multi-document abstraction.

⚠ Current Limitations

Extractive summarization tendencies Moderate cross-paper synthesis CPU latency (~60–80 seconds per query)

🚀 Next Phase

The next stage of this project will introduce:

GPU-based LLM upgrade (Mistral-7B / Llama 3) Quantitative comparison with CPU baseline Improved synthesis metrics Latency optimization

This Space serves as the documented baseline for that upgrade.
