# PsyMedRAG
# PsyMedRAG

# 🧠 Medical QA System (RAG for Mental Health)

This project implements a **Retrieval-Augmented Generation (RAG)** based **medical question-answering system**, with a special focus on **mental health**.
 It explores **multi-path retrieval, encoder fine-tuning, and domain-adapted LLMs** to deliver more accurate, context-aware, and trustworthy answers in medical QA.

------

## ✨ Key Features

### 1. 🔍 Multi-path Retrieval & Re-ranking

- Employs **multi-path retrieval** to combine different retrieval strategies and ensure comprehensive coverage.
- Both **Bi-Encoder** and **Cross-Encoder** are fine-tuned:
  - **Bi-Encoder**: fast large-scale recall with high coverage.
  - **Cross-Encoder**: re-ranking for precision, significantly improving **Recall** and **Mean Reciprocal Rank (MRR)**.
- Final results are fused across retrieval paths before being passed to the generation stage.
- Hugging Face: https://huggingface.co/Elearon/deepseek-medical-lora

------

### 2. 🧑‍⚕️ Domain-specific Model for Mental Health

- A domain-adapted **DeepSeek-llm-7B-chat** is fine-tuned on mental health data.
- Ensures that generated answers are **safer, more appropriate, and contextually relevant** for mental health queries.
- Covers knowledge related to **common disorders, clinical symptoms, diagnosis, and treatments**.

------

### 3. 🧠 Multi-LLM Orchestrated Pipeline

The system is orchestrated by **multiple LLMs**, each responsible for a specific stage of reasoning:

1. **Routing LLM** – Classifies incoming questions and decides whether retrieval is required.
2. **Retrieval LLM** – Manages query construction, document recall, and re-ranking using Bi-Encoder + Cross-Encoder.
3. **History-Aware LLM** – Maintains context consistency across **multi-turn dialogue**.
4. **Answer Generation LLM** – Synthesizes retrieved knowledge and conversation history to produce the final response (using fine-tuned **DeepSeek-llm-7B-chat**).

This design follows a **multi-agent paradigm**, ensuring robustness and modularity.

------

### 4. 🔄 Multi-turn, Context-aware QA

- Supports **multi-turn dialogue**, where the system leverages past interactions to refine current answers.
- Retrieval and generation are both dynamically conditioned on dialogue history.
- Provides a more **realistic conversational experience** compared to single-turn QA.

------

## 📦 Installation & Usage

Clone this repo and install dependencies:

```
git clone https://github.com/zpf-wq/PsyMedRAG.git
```

````
cd PsyMedRAG
````

````
pip install -r requirements.txt
````

Make sure you have access to the fine-tuned models (Bi-Encoder, Cross-Encoder, DeepSeek-llm-7B-chat).

**Start the retrieval & QA service**:

```
python Agent/main.py
```

This will load the retrievers and the fine-tuned DeepSeek model.

------

## 📊 Applications

- **Research**: A testbed for exploring **RAG pipelines** in medical NLP, including retrieval optimization, encoder fine-tuning, and domain-specific LLM adaptation.
- **Education & Clinical Support**: Helps students, researchers, and clinicians access mental health knowledge through natural Q&A.
- **Engineering**: Modular design allows easy replacement of LLMs, retrieval strategies, and backends to adapt to other medical or specialized domains.

------

## 🛠️ Tech Stack

- **Retrieval**: Fine-tuned Bi-Encoder & Cross-Encoder
- **LLMs**: DeepSeek-llm-7B-chat (domain adaptation for mental health)
- **Framework**: RAG (retrieval + generation)
- **Architecture**: Multi-LLM orchestration (routing, retrieval, dialogue memory, generation)
- **Dialogue**: Multi-turn, history-aware QA

------

## 📚 Summary

This project demonstrates how **RAG + multi-path retrieval + multi-LLM orchestration** can improve **accuracy, safety, and contextual relevance** in medical QA.
 By combining **retrieval optimization** with a **domain-specific fine-tuned LLM**, the system provides a practical blueprint for **intelligent medical assistants**, with a particular emphasis on **mental health**.
## 📚 Summary

This project demonstrates how **RAG + multi-path retrieval + multi-LLM orchestration** can improve **accuracy, safety, and contextual relevance** in medical QA.
 By combining **retrieval optimization** with a **domain-specific fine-tuned LLM**, the system provides a practical blueprint for **intelligent medical assistants**, with a particular emphasis on **mental health**.
