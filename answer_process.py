import pandas as pd 
import json
import re 

def preprocess(ans):
    # ans = ans.strip()

    # Simplify check using any()
    if any(keyword in ans for keyword in ['Correct']):
        return 'Đúng'

    if any(ans == incorrect_ans for incorrect_ans in ['False']):
        return 'Sai'    

    ans = ans.replace('\n', ' ').replace('.', ' ').replace(')', ' ').replace('\b','')
    ans = ''.join(x for x in ans if x.isalpha() or x == ' ')
    ans = ans.strip()
    ans = ans.split()
    ans = ans[0]

    if any(keyword in ans for keyword in ['Correct', 'đúng', 'Đúng', 'Có']):
        return 'Đúng'

    if any(ans == incorrect_ans for incorrect_ans in ['False', 'sai', 'Sai', 'Không']):
        return 'Sai' 
    # # Simplify checks using any() for both conditions
    # if any(ans == correct_ans for correct_ans in ['đúng', 'Có']):
    #     return 'Đúng'
    
    return ans

def preprocess_data(row):
    if "TL" in row['question_id']:
        if "TL-7" in row['question_id']:
            index = row['answer'].find("4.")
            return row['answer'][:index]
        else:
            index = row['answer'].find("#")
            first_sentence = row['answer'][:index-1]
            return first_sentence
    else: 
        return preprocess(row['answer'])

ans_path = "./private_test/" + "3epo.json"
data = pd.read_json(ans_path)

data['answer'] = data.apply(preprocess_data, axis=1)

#print(data.head)
new_name = "./private_test/preprocessed/" + "[se7enese] 3epo_new.json"
data.to_json(new_name, orient='records', force_ascii=False)
print("DONE!")
