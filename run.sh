#MODEL="./Vistral-7B-Chat"
#MODEL="/home/support/llm/Meta-Llama-3-8B-Instruct"
#MODEL="vilm/vinallama-2.7b-chat"
#MODEL="uonlp/Vistral-7B-Chat-gguf"
#MODEL="NousResearch/Llama-2-7b-chat-hf"
#MODEL="chillies/vistral-legal-chat-q4"
MODEL="chillies/vinallama-legal-chat"
DATAPATH="./data/public_test.json"

python3 main.py \
	--model_id $MODEL \
	--file $DATAPATH
