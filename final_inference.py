import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd 
from tqdm import tqdm 

model_name_or_path = './Meta-Llama-3-8B-Instruct/'
lora_path = './ckpt/Meta-Llama-3-8B-Instruct_lora_5-epo_1.0'

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(model_name_or_path)

print(f"Loading LoRA weights from {lora_path}")
model = PeftModel.from_pretrained(model, lora_path)
print(f"Merging weights")
model = model.merge_and_unload()
model.to('cuda')
print("Done")


def generate_response(
            prompt, 
            do_sample:bool = True, 
            top_p:float=0.95, 
            temperature:float=0.7, 
            num_beams:int = 2,
            max_length:int = 512):
    with torch.inference_mode():
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to('cuda')
        output = model.generate(
                input_ids,
                num_return_sequences=1, 
                num_beams=num_beams,
                temperature=temperature, 
                do_sample=do_sample, 
                top_p=top_p,
                max_length=max_length)
        response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

INS_PROMPT = ''' You are a helpful Vietnamese legal assisant with the mission of answering the question based on the given article without explanation.

### Article: {article}

### Question: {question}\n'''
CHOICE = '{choices}'

INS_ANS = '''### Answer: '''

# Instruction

def formatting_option(options):
    return f"A) {options['A']}\nB) {options['B']}\nC) {options['C']}\nD) {options['D']}"

def formatting_func(sample):
    text = INS_PROMPT.format(article = sample['article'], question = sample['text'])
    if sample['question_type'] == 'Trắc nghiệm':
        text += CHOICE.format(choices = formatting_option(sample['choices']))
    
    text += INS_ANS
#    text = INS_PROMPT.format(question=sample['question'], choices=choices, answer=ans)
    return text


generate_args = {
        "num_beams": 2,
        "temperature": 0.7, 
        "do_sample": True, 
        "top_p": 0.8,
        "max_length": 2560,
        "do_sample": True, 
        }

data = pd.read_json('./data/public_test_updated.json')

answers = list()
for i in tqdm(range(len(data))):
    data_i = data.iloc[i]
    prompt = formatting_func(data_i)
    response = generate_response(prompt, **generate_args)
    parsed_response = response[len(prompt):]
    print(parsed_response)
    answers.append(parsed_response)

new_data = pd.DataFrame({'question_id': data['question_id'].tolist(),
    'answer': answers})

new_data.to_json('./results/task2/lora_llama3_5epo.json', orient='records', force_ascii = False)
