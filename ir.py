from utils import *
import bm25s

class retrieval():
    def __init__(self, corpus, top_k = 3):

        self.corpus = corpus
        self.top_k = top_k
    
        # Tokenize corpus & query
        self.corpus_articles = list(self.corpus.values())
        self.corpus_tokens = bm25s.tokenize(self.corpus_articles)

        # Create the BM25 model and index the corpus
        self.retriever = bm25s.BM25()
        self.retriever.index(self.corpus_tokens)

    def main(self, query):

        query_token = bm25s.tokenize(query)

        # Get top-k results 
        results, scores = self.retriever.retrieve(
                query_token, 
                corpus = self.corpus_articles, 
                k = self.top_k)
        
        article_ids = list()
        docs = list()
        for i in range(self.top_k):
            doc = results[0, i]
            docs.append(doc)
            article_ids.append(id_return(doc))

        return article_ids
