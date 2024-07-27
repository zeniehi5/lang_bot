from typing import List, Dict
from store.store import Store
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from konlpy.tag import Kkma
from langchain.schema import Document
from langchain_core.runnables import chain


class Retriever:
    def __init__(self, ensemble: bool, weights: List[Dict[str, float]]=[{'keyword': 0.0}, {'similarity': 1.0}]):
        self.store = None
        self.ensemble = ensemble
        self.kkma = Kkma()
        self.max_tokens = 500
        self.weights = []
        for key in ['keyword', 'similarity']:
            for weight_dict in weights:
                if key in weight_dict and isinstance(weight_dict[key], float):
                    self.weights.append(weight_dict[key])

    def retrieve(self, store: Store, question: str, top_k: int = 5):
        try:
            self.store = store
            documents = self.store.get_all_documents()
            self.bm25_retriever = BM25Retriever.from_documents(documents=documents, preprocess_func=self.kkma_tokenize)
            self.bm25_retriever.k = 2
            self.ensemble_retriever = EnsembleRetriever(
                retrievers=[self.bm25_retriever, self.store.as_retriever(k=3)],
                weights=self.weights,
                search_type="similarity",
                top_k=top_k
            )
            result = self.ensemble_retriever.invoke(question)
            # result = self.similarity_search_with_score(question)
            print("result:", result)
            print("result.len:", len(result))
            return result
        except Exception as e:
            print(f"An error occurred during retrieval: {str(e)}")
            return None
        
    def similarity_search_with_score(self, question: str) -> List[Document]:
        docs, scores = zip(*self.store.similarity_search_with_score(question))
        for doc, score in zip(docs, scores):
            doc.metadata["score"] = score

        print("docs: ", docs)
        return docs
        
    def kkma_tokenize(self, text):
        try:
            return self.kkma.morphs(text)
        except Exception as e:
            raise ValueError(f"An error occurred during tokenize: {str(e)}")
        
    def tokenize_and_split(self, text: str):
        tokens = self.kkma_tokenize(text)
        return [tokens[i:i + self.max_tokens] for i in range(0, len(tokens), self.max_tokens)]
