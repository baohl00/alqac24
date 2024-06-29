import json

def read_json(path):
    
    with open(path, 'r') as file:
        data = json.load(file)
    
    return data

corpus = read_json('./data/corpus.json')

#  Luat_dan_su@2 
#     ->  {'law_id': 'Luat_dan_su',
#         'aritcle_id': 2}
def split_id(id):

    index_at = id.index('@')
    law_id = id[:index_at]
    article_id = id[index_at+1:]

    return law_id, article_id

# Search id based on article in the corpus
def id_return(article):

    for id, content in corpus.items():
        if content == article:
            law_id, article_id = split_id(id)                
            return {'law_id': law_id, 'article_id': article_id, 'article': article}
    return None

# Read article based on law_id & article_id
def read_article(law_id, article_id):
    return corpus[f'{law_id}@{article_id}']

# Update article for dataset
def update_data(data):
    articles = list()

    for i in range(len(data)):
        data_i = data.iloc[i]
        data_i_article = data_i['relevant_articles']
        article = read_article(
                law_id = data_i_article[0]['law_id'],
                article_id = data_i_article[0]['article_id']
                ) 
        articles.append(article)

    data['article'] = articles

    new_name = f'./data/public_test_updated.json'
    data.to_json(new_name, orient = 'records', force_ascii = False)

    return data

# Export json result 
def export(data, model_id = "", task = 1, note = None):
    
    id_s = model_id.rfind('/')
    model_name = model_id[id_s+1:]

    file_name = f"./results/task{task}/{model_name}_{note}.json"
    
    if task == 1:
        data = data[["question_id", "relevant_articles"]]
    else:
        data = data[["question_id", "answer"]]

    data.to_json(file_name, orient = 'records', force_ascii = False)

    print(f'JSON <Task {task}> EXPORT DONE!')


