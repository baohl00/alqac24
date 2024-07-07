import argparse
import llm
import ir 
import dataset
from utils import *
import pandas as pd
from tqdm import tqdm
#from result_export import export

def final(args):
    corpus = read_json("./data/corpus.json")
#    dataset = pd.read_json("./data/public_test.json")
    dataset = pd.read_json(args.file)
#    dataset = dataset.loc[98:112]
    questions = list(dataset.text.values)
    question_type = list(dataset.question_type.values)
    data_length = len(questions)
    list_choices = list()
    
    for i in range(data_length):
        if question_type[i] == "Trắc nghiệm":
            list_choices.append(dataset.iloc[i]['choices'])
        else:
            list_choices.append(None)

    ir_model = ir.retrieval(corpus, top_k = 1)

#    model_id = "openbmb/MiniCPM-Llama3-V-2_5"
#    model_id = "tiiuae/falcon-7b-instruct"
#    model_id = "/home/support/llm/Meta-Llama-3-8B-Instruct"
#    model_id = "Viet-Mistral/Vistral-7B-Chat"
    qa_model = llm.model_qa(args.model_id)

    predicts = list()
    answers = list()
    for i in tqdm(range(data_length)):
        print(questions[i])
        documents = ir_model.main(questions[i])
        #print(f"Cau hoi: {question}\nDap an: {documents}")
        #ans = f"{documents[0]['law_id']}@{documents[0]['article_id']}"
        for doc in documents:
            ans = qa_model.answering(question = questions[i], 
                    document = doc['article'], 
                    question_type = question_type[i],
                    options = list_choices[i])
            predicts.append([{'law_id': doc['law_id'],
                'article_id': doc['article_id']}])
            answers.append(ans)
            print(predicts[-1])
            print(ans)

    dataset['relevant_articles'] = predicts
    dataset['answer'] = answers
    #export_task1(dataset, './results/task1.json')
    export(dataset, task = 1)
    export(dataset, args.model_id, task = 2, note = "prompt1")
    
    print('DONE!')

#final()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_id", default="./Meta-Llama-3-8B-Instruct", help="path to model id")
    parser.add_argument("--file", default="./data/public_test.json", help="path to data file")
#    parser.add_argument("--output_dir")
    args = parser.parse_args()

    final(args)
