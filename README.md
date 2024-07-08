# ALQAC24

Team: **se7enese.**

Member: **Hoang-Bao Le.** 

Affiliation: **Dublin City University.**  

_For further information, please visit [here](<https://baohl00.github.io/>)._

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
    
    We implemented the retrieval model in three different ways:
    
    - The first idea is using BM25 combined with attention.
    - The second idea is using BM25 combined with attention and CNN networks.
    - The third idea is using new [BM25s](https://bm25s.github.io).

3. **Pretrained model**: 
    - To enhance the knowledge of the dataset size, we concatenate two train sets into a new one added the related article and propose a prompt to put every useful information into an input for the model.
    - We also apply LoRA for fine-tuning model. Here is our experiment details. We fine-tune the model in 1, 3, and 5 epochs and report it in the following section.
    - Finally, after the inference stage, we should manually process the generated answer again to remove the irrelevant parts and adjust the answer suitably.

# Data

For the training data, we combine two sets train and unverified train to extend the diversity of itself (named as “Total train” in the table below).

 

|  | Đúng/Sai | Trắc nghiệm | Tự luận |
| --- | --- | --- | --- |
| Train | 50 | 40 | 10 |
| Unverified train | 208 | 173 | 49 |
| Total train | 258 | 213 | 59 |
| Public test | 132 | 76 | 0 |
| Private test | 48 | 43 | 9 |

# How to run?

- For BM25s and non-fine-tuning LLMs, please run the bash file [`scripts/run.sh`](scripts/run.sh):

```bash
#MODEL="./Vistral-7B-Chat"
#MODEL="./Meta-Llama-3-8B-Instruct"
#MODEL="NousResearch/Llama-2-7b-chat-hf"
#MODEL="chillies/vistral-legal-chat-q4"
MODEL="chillies/vinallama-legal-chat"
DATAPATH="./data/public_test.json"

python3 main.py \
	--model_id $MODEL \
	--file $DATAPATH
```

However, this file is only run with 1 GPU. If you aim to use multi-gpus, we highly recommend use torchrun instead of deepspeed.  

- For fine-tuning the LLMs, please run the bash file [`scripts/lora_finetune.sh`](scripts/lora_finetune.sh):

```bash
BASE_DIR=./
MODEL=Meta-Llama-3-8B-Instruct
CONFIG=${BASE_DIR}/scripts/zero3_offload.json
OUTDIR=${BASE_DIR}/ckpt/${MODEL}
TRAIN_FILE=${BASE_DIR}/data/train_updated.json
BATCHSIZE=4
EPOCH=5

export CUDA_VISIBLE_DEVICES=0,1

torchrun --nnodes=1 --nproc_per_node=2 --master_port=25035 \
	${BASE_DIR}/train.py \
	--model_name_or_path ./${MODEL} \
	--data_path ${TRAIN_FILE} \
	--lora_enable True\
	--lora_r 16 \
	--lora_alpha 16 \
	--lora_dropout 0.05 \
	--dataloader_num_workers 8 \
	--fp16 \
	--output_dir ${OUTDIR}_lora_$EPOCH-epo_1.0 \
	--per_device_train_batch_size ${BATCHSIZE} \
	--gradient_accumulation_steps 1 \
	--num_train_epochs $EPOCH \
	--fp16 False \
	--save_strategy "steps" \
	--save_total_limit 1 \
	--learning_rate 2e-5 \
	--warmup_ratio 0.03 \
	--lr_scheduler_type "cosine" \
	--logging_steps 1 \
	--logging_dir "$OUTDIR" \
	--report_to wandb \
	--run_name LoraFT_Llama3_$EPOCH-epo_1.0
```

# Results

**Document Retrieval**

| Model | k | Pre | Rec | F2 |
| --- | --- | --- | --- | --- |
| bm25 attention | 0.84 | 71.15 | 69.71 | 69.87 |
|  | 1.5 | 62.98 | 61.78 | 61.91 |
|  | 0 | 63.46 | 62.26 | 62.39 |
| bm25 attention cnn | 0.5 | 63.46 | x | x |
| bm25s | x | 72.16 | x | x |

**Fine-tuned LLaMA3 (version 8B Instruct) and Inference**

| Epoch(s) | Accuracy | Note |
| --- | --- | --- |
| 1 | 75 | Keep the first word |
| 3 | 81.73 | Keep the first word |
| 5 | 84.14 | + Rule base for special cases  |
