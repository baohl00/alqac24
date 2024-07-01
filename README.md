# ALQAC24

Team: **se7enese.**

Member: **Hoang-Bao Le.** 

Affiliation: **Dublin City University.**  

# Methodology

1. **Pipeline**: 

Preprocessing corpus → Retrieval model → Update training data → Pretrained model → Preprocessing answer 

1. **Preprocessing data**:
    
    **Stage 1 (corpus)**
    
    - Remove “\n\n” character.
    - Keep the first sentence as the topic sentence for the following article.
    
    **Stage 2 (update)**
    
    - Add “related article”
    - Format the input as the following instruction:
    
    > You are a helpful Vietnamese legal assistant with the mission of answering the question based on the given article without explanation.
    >  
    > \### Article: {article}
    >  
    > \### Question: {question}
    >  
    > {choices}
    >  
    > \### Answer: {answer} 
   
    Only add the {choices} part in case the question_type is “Trắc nghiệm”.
    
    **Stage 3 (final answer)**
    
    - Keep the first word (answer): “Đúng” and “Sai” for “Đúng/Sai” question; option (A, B, C, or D) for multiple choice question. For the “Tự luận” question, we
    - Manually check with the generated answer of the model to fill in the correct answer.
2. **Retrieval model:**

1. **Pretrained model**: 
- 

# Data

For the training data, we combine two sets train and unverified train to extend the diversity of itself (named as “Total train” in the table below).

 

|  | Đúng/Sai | Trắc nghiệm | Tự luận |
| --- | --- | --- | --- |
| Train | 50 | 40 | 10 |
| Unverified train | 208 | 173 | 49 |
| Total train | 258 | 213 | 59 |
| Public test | 132 | 76 | 0 |
| Private test | 48 | 43 | 9 |

# Results

- BM25s for Information Retrieval

| Model | k | Pre | Rec | F2 |
| --- | --- | --- | --- | --- |
| bm25 attention | 0.84 | 71.15 | 69.71 | 69.87 |
|  | 1.5 | 62.98 | 61.78 | 61.91 |
|  | 0 | 63.46 | 62.26 | 62.39 |
| bm25 attention cnn | 0.5 | 63.46 |  |  |
| bm25s |  | 72.16 |  |  |
1. **LLMs for Text Generation**
    - Meta-Llama-3-8B-Instruct
    - vinallama-2.7-chat
    - vinallama-legal-chat
    
    | Model | Accuracy | Note |
    | --- | --- | --- |
    | Meta | 51.92 |  |
    | vinallama | x |  |
    | vinallama-legal | 53.85 |  |
2. **Pretrained model in train_updated dataset:**
    
    
    | Model | Accuracy | Note |
    | --- | --- | --- |
    | Meta | 75 |  |
    | vinallama-legal | x |  |
