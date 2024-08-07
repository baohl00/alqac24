BASE_DIR=.
MODEL=Meta-Llama-3-8B-Instruct
#MODEL=vilm/vinallama-2.7b-chat
CONFIG=${BASE_DIR}/scripts/zero3_offload.json
OUTDIR=${BASE_DIR}/ckpt/${MODEL}
TRAIN_FILE=${BASE_DIR}/data/train_updated.json
BATCHSIZE=4
EPOCH=5

export CUDA_VISIBLE_DEVICES=0,1
#python3 ${BASE_DIR}/train.py \
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
