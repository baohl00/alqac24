import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name_or_path = './Meta-Llama-3-8B-Instruct/'
lora_path = './ckpt/Meta-Llama-3-8B-Instruct_lora_1-epo'

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(model_name_or_path)

print(f"Loading LoRA weights from {lora_path}")
model = PeftModel.from_pretrained(model, lora_path)
print(f"Merging weights")
model = model.merge_and_unload()
print("Done")

def generate_response(
            prompt, 
            do_sample:bool = True, 
            top_p:float=0.95, 
            temperature:float=0.7, 
            num_beams:int = 2,
            max_length:int = 512):
    with torch.inference_mode():
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
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

INS_ANS = '''\n### Answer: '''

question = 'Lưu trú là việc công dân sinh sống tại một địa điểm thuộc đơn vị hành chính cấp xã hoặc đơn vị hành chính cấp huyện ở nơi không có đơn vị hành chính cấp xã (sau đây gọi chung là đơn vị hành chính cấp xã), đúng hay sai?'
qtype = 'Đúng\/Sai'
article = 'Giải thích từ ngữ.Trong Luật này, các từ ngữ dưới đây được hiểu như sau:.1. Chỗ ở hợp pháp là nơi được sử dụng để sinh sống, thuộc quyền sở hữu hoặc quyền sử dụng của công dân, bao gồm nhà ở, tàu, thuyền, phương tiện khác có khả năng di chuyển hoặc chỗ ở khác theo quy định của pháp luật. 2. Cư trú là việc công dân sinh sống tại một địa điểm thuộc đơn vị hành chính cấp xã hoặc đơn vị hành chính cấp huyện ở nơi không có đơn vị hành chính cấp xã (sau đây gọi chung là đơn vị hành chính cấp xã). 3. Cơ sở dữ liệu về cư trú là cơ sở dữ liệu chuyên ngành, tập hợp thông tin về cư trú của công dân, được số hóa, lưu trữ, quản lý bằng cơ sở hạ tầng thông tin, được kết nối, chia sẻ với Cơ sở dữ liệu quốc gia về dân cư và cơ sở dữ liệu khác theo quy định của pháp luật. 4. Cơ quan đăng ký cư trú là cơ quan quản lý cư trú trực tiếp thực hiện việc đăng ký cư trú của công dân, bao gồm Công an xã, phường, thị trấn; Công an huyện, quận, thị xã, thành phố thuộc tỉnh, thành phố thuộc thành phố trực thuộc trung ương ở nơi không có đơn vị hành chính cấp xã. 5. Đăng ký cư trú là việc thực hiện thủ tục đăng ký thường trú, đăng ký tạm trú, khai báo tạm vắng, thông báo lưu trú và khai báo thông tin, điều chỉnh thông tin về cư trú. 6. Lưu trú là việc công dân ở lại một địa điểm không phải nơi thường trú hoặc nơi tạm trú trong thời gian ít hơn 30 ngày. 7. Tạm vắng là việc công dân vắng mặt tại nơi cư trú trong một khoảng thời gian nhất định. 8. Nơi thường trú là nơi công dân sinh sống ổn định, lâu dài và đã được đăng ký thường trú. 9. Nơi tạm trú là nơi công dân sinh sống trong một khoảng thời gian nhất định ngoài nơi thường trú và đã được đăng ký tạm trú. 10. Nơi ở hiện tại là nơi thường trú hoặc nơi tạm trú mà công dân đang thường xuyên sinh sống; trường hợp không có nơi thường trú, nơi tạm trú thì nơi ở hiện tại là nơi công dân đang thực tế sinh sống.'
choices = [
        "A) Công an nhân dân",
        "B) Hải quan",
        "C) Bộ đội biên phòng",
        "D) Cả A, B, C đều đúng"
        ]
choices = '\n' + '\n'.join(choices) + '\n'

question1 = 'Các cơ quan nào sau đây thực hiện chuyên trách phòng, chống tội phạm về ma túy ?'
qtype1 = 'Trắc nghiệm'
article1 = 'Cơ quan chuyên trách phòng, chống tội phạm về ma túy.1. Cơ quan chuyên trách phòng, chống tội phạm về ma túy bao gồm:.a) Cơ quan chuyên trách phòng, chống tội phạm về ma túy thuộc Công an nhân dân;.b) Cơ quan chuyên trách phòng, chống tội phạm về ma túy thuộc Bộ đội Biên phòng, Cảnh sát biển Việt Nam và Hải quan. 2. Cơ quan chuyên trách phòng, chống tội phạm về ma túy thuộc Công an nhân dân, trong phạm vi nhiệm vụ, quyền hạn của mình, chủ trì, phối hợp với cơ quan, tổ chức có liên quan thực hiện các hoạt động phòng ngừa, ngăn chặn và đấu tranh chống tội phạm về ma túy. 3. Cơ quan chuyên trách phòng, chống tội phạm về ma túy thuộc Bộ đội Biên phòng, Cảnh sát biển Việt Nam, Hải quan, trong phạm vi nhiệm vụ, quyền hạn của mình, chủ trì, phối hợp với cơ quan công an, cơ quan, tổ chức khác có liên quan thực hiện các hoạt động phòng ngừa, ngăn chặn và đấu tranh chống tội phạm về ma túy tại khu vực hoặc địa bàn quản lý, kiểm soát. 4. Trên cùng một địa bàn khi phát hiện hành vi vi phạm pháp luật liên quan đến nhiệm vụ, quyền hạn của nhiều cơ quan thì cơ quan phát hiện trước có trách nhiệm xử lý theo thẩm quyền do pháp luật quy định; trường hợp vụ việc không thuộc thẩm quyền của mình thì chuyển giao hồ sơ, người, tang vật vi phạm pháp luật cho cơ quan có thẩm quyền chủ trì giải quyết. 5. Chính phủ quy định việc phối hợp của các cơ quan chuyên trách phòng, chống tội phạm về ma túy.'

prompt = INS_PROMPT.format(article = article, question = question) 

prompt += CHOICE.format(choices = choices ) if qtype == 'Trắc nghiệm' else '' 

prompt += INS_ANS

print(prompt)

generate_args = {
        "num_beams": 2,
        "temperature": 0.7, 
        "do_sample": True, 
        "top_p": 0.8,
        "max_length": 2048,
        }

response = generate_response(prompt, **generate_args)
parsed_response = response[len(prompt):]

print(parsed_response)
