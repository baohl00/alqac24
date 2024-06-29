from transformers import (
        AutoModelForCausalLM, AutoTokenizer, 
        )
import torch 

device = 'cuda'

def get_formatted_input(messages, context):
    #system = "System: This is a chat between a user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions based on the context. The assistant should also indicate when the answer cannot be found in the context."
    system = "System: You are a law assistant that supports user with a helpful, detailed, and polite answer to their questions based on the context. The assistant should also indicate when the answer cannot be found in the context."
    instruction = "Please give a directly short answer for the question!"
    
    for item in messages:
        if item['role'] == "user":
        ## only apply this instruction for the first user turn
            item['content'] = instruction + " " + item['content']
            break

    conversation = '\n\n'.join(["User: " + item["content"] if item["role"] == "user" else "Assistant: " + item["content"] for item in messages]) + "\n\nAssistant:"
    formatted_input = system + "\n\n" + context + "\n\n" + conversation
        
    return formatted_input

def formatted_options(options): # options = {"A": , "B": ...}
    return f"A) {options['A']}\nB) {options['B']}\nC) {options['C']}\nD) {options['D']}"

# (tf, op, fr) = (true/false, options, free text)
general_context = "Bạn là hệ thống trả lời pháp luật thông minh. Nhiệm vụ của bạn là trả lời ngắn gọn và chính xác dựa trên văn bản, yêu cầu và câu hỏi, và thực hiện đúng theo lưu ý.\n\n"
context_tf = "###Yêu cầu: Bạn phải trả lời đúng hoặc sai cho câu hỏi tương ứng.\n###Câu hỏi: {question}"

context_op = "###Yêu cầu: Bạn phải lựa chọn đáp án đúng nhất trong bốn đáp án A, B, C và D.\n###Câu hỏi: {question}\n{options}"

context_fr = "###Yêu cầu: Bạn phải trả lời ngắn gọn nhất cho câu hỏi tương ứng.\n###Câu hỏi: {question}"

note = "\nLưu ý: Bạn CHỈ trả lời đáp án và KHÔNG GIẢI THÍCH thêm."
note_op = "\nLưu ý: Với dạng câu chọn A, B, C, và D, bạn chỉ cần đưa ra đáp án."
note_tf = "\nLưu ý: Bạn CHỈ trả lời ĐÚNG hoặc SAI!"

def formatted_input_by_type(question, context, datatype, options):

    content = general_context + "### Ngữ cảnh: {context}"
    if datatype == "Đúng/Sai":
        content += context_tf.format(question = question) + note
    elif datatype == "Trắc nghiệm":
        content += context_op.format(question = question, options = formatted_options(options)) + note
    else:
        content += context_fr.format(question = question) + note_tf

    conversation = 'Con người: ' + content + '\n\nHệ thống: \n### Trả lời: '

    return conversation

class model_qa():
    def __init__(self, 
            model_id = 'nvidia/Llama3-ChatQA-1.5-8B'):
         
        #self.question = question
        # chillies/vistral-legal-chat-q4
        if 'vistral-legal' in model_id:
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, gguf_file="vinallama-legal-chat-unsloth.Q4_K_M.gguf")
            self.model = AutoModelForCausalLM.from_pretrained(model_id, gguf_file="vinallama-legal-chat-unsloth.Q4_K_M.gguf", torch_dtype=torch.float16, device_map="auto").to(device)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto").to(device)
 
    def answering(self, question, document, question_type=None, options=None):
        
#        message = [{"role": "user", "content": question}]

#        formatted_input = get_formatted_input(message, document)
        formatted_input = formatted_input_by_type(question, document, question_type, options)

        tokenized_prompt = self.tokenizer(self.tokenizer.bos_token + formatted_input, return_tensors="pt").to(device)

        terminators = [
                    self.tokenizer.eos_token_id,
                    self.tokenizer.convert_tokens_to_ids("<|eot_id|>")]

        outputs = self.model.generate(
                input_ids=tokenized_prompt.input_ids, 
                attention_mask=tokenized_prompt.attention_mask, 
                max_new_tokens=256, 
                eos_token_id=terminators)

        response = outputs[0][tokenized_prompt.input_ids.shape[-1]:]
        
        return self.tokenizer.decode(response, skip_special_tokens=True)
