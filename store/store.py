from typing import Dict, List, Optional, Tuple
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document
from sklearn.mixture import GaussianMixture

import numpy as np
import umap
import pandas as pd

class Store:
    def __init__(self, model='text-embedding-ada-002', documents: List[Document]=None):
        DB_PATH = "./chroma_db"
        self.embedding_model = OpenAIEmbeddings(api_key="", model=model)
        self.vector_store = Chroma(
            embedding_function=self.embedding_model,
            persist_directory=DB_PATH
        )
        self.documents = []
        self.texts = []
        self.RANDOM_SEED = 224

        if documents:
            self.add_documents(documents)
    
    def global_cluster_embeddings(
        self,
        embeddings: np.ndarray,
        dim: int,
        n_neighbors: Optional[int] = None,
        metric: str = "cosine",
    ) -> np.ndarray:
        if n_neighbors is None:
            n_neighbors = int((len(embeddings) - 1) ** 0.5)
        return umap.UMAP(
            n_neighbors=n_neighbors, n_components=dim, metric=metric
        ).fit_transform(embeddings)
    
    def local_cluster_embeddings(
        self, embeddings: np.ndarray, dim: int, num_neighbors: int = 10, metric: str = "cosine"
    ) -> np.ndarray:
        return umap.UMAP(
            n_neighbors=num_neighbors, n_components=dim, metric=metric
        ).fit_transform(embeddings)
    
    def get_optimal_clusters(
        self, embeddings: np.ndarray, max_clusters: int = 50, random_state: int = 224
    ) -> int:
        max_clusters = min(max_clusters, len(embeddings))
        n_clusters = np.arange(1, max_clusters)
        bics = []
        for n in n_clusters:
            gm = GaussianMixture(n_components=n, random_state=random_state)
            gm.fit(embeddings)
            bics.append(gm.bic(embeddings))
        return n_clusters[np.argmin(bics)]
    
    def GMM_cluster(self, embeddings: np.ndarray, threshold: float, random_state: int = 0):
        n_clusters = self.get_optimal_clusters(embeddings)
        gm = GaussianMixture(n_components=n_clusters, random_state=random_state)
        gm.fit(embeddings)
        probs = gm.predict_proba(embeddings)
        labels = [np.where(prob > threshold)[0] for prob in probs]
        return labels, n_clusters
    
    def perform_clustering(
        self,
        embeddings: np.ndarray,
        dim: int,
        threshold: float,
    ) -> List[np.ndarray]:
        if len(embeddings) <= dim + 1:
            return [np.array([0]) for _ in range(len(embeddings))]

        reduced_embeddings_global = self.global_cluster_embeddings(embeddings, dim)
        global_clusters, n_global_clusters = self.GMM_cluster(
            reduced_embeddings_global, threshold
        )

        all_local_clusters = [np.array([]) for _ in range(len(embeddings))]
        total_clusters = 0

        for i in range(n_global_clusters):
            global_cluster_embeddings_ = embeddings[
                np.array([i in gc for gc in global_clusters])
            ]

            if len(global_cluster_embeddings_) == 0:
                continue
            if len(global_cluster_embeddings_) <= dim + 1:
                local_clusters = [np.array([0]) for _ in global_cluster_embeddings_]
                n_local_clusters = 1
            else:
                reduced_embeddings_local = self.local_cluster_embeddings(
                    global_cluster_embeddings_, dim
                )
                local_clusters, n_local_clusters = self.GMM_cluster(
                    reduced_embeddings_local, threshold
                )

            for j in range(n_local_clusters):
                local_cluster_embeddings_ = global_cluster_embeddings_[
                    np.array([j in lc for lc in local_clusters])
                ]
                indices = np.where(
                    (embeddings == local_cluster_embeddings_[:, None]).all(-1)
                )[1]
                for idx in indices:
                    all_local_clusters[idx] = np.append(
                        all_local_clusters[idx], j + total_clusters
                    )
            total_clusters += n_local_clusters

        return all_local_clusters

    def embed(self, texts):
        text_embeddings = self.embedding_model.embed_documents(texts)
        text_embeddings_np = np.array(text_embeddings)
        return text_embeddings_np
    
    def embed_cluster_texts(self, texts):
        text_embeddings_np = self.embed(texts)
        cluster_labels = self.perform_clustering(
            text_embeddings_np, 10, 0.1
        )
        df = pd.DataFrame()
        df["text"] = texts
        df["embed"] = list(text_embeddings_np)
        df["cluster"] = cluster_labels
        return df
    
    def fmt_txt(self, df: pd.DataFrame) -> str:
        unique_txt = df["text"].tolist()
        return "--- --- \n --- --- ".join(unique_txt)
    
    def embed_cluster_summarize_texts(self, texts: List[str], level: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        df_clusters = self.embed_cluster_texts(texts)
        expanded_list = []

        for index, row in df_clusters.iterrows():
            for cluster in row["cluster"]:
                expanded_list.append(
                    {"text": row["text"], "embed": row["embed"], "cluster": cluster}
                )
        expanded_df = pd.DataFrame(expanded_list)
        all_clusters = expanded_df["cluster"].unique()
        template = """
            Here is a sub-set of LangChain Expression Language doc.
            LangChain Expression Langauge provides a way to compose chain in LangChain.
            Give a detailed summary of the documentation provided.

            Documentation:
            {context}
        """
        prompt = ChatPromptTemplate.from_template(template)
        model = ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            streaming=True
        )
        chain = prompt | model | StrOutputParser()

        summaries = []
        for i in all_clusters:
            df_cluster = expanded_df[expanded_df["cluster"] == i]
            formatted_txt = self.fmt_txt(df_cluster)
            summaries.append(chain.invoke({"context", formatted_txt}))

        df_summary = pd.DataFrame(
            {
                "summaries": summaries,
                "level": [level] * len(summaries),
                "cluster": list(all_clusters)
            }
        )
        return df_clusters, df_summary
    
    def recursive_embed_cluster_summarize(
            self, texts: List[str], level: int = 1, n_levels: int = 3
    ) -> Dict[int, Tuple[pd.DataFrame, pd.DataFrame]]:
        results = {}
        df_clusters, df_summary = self.embed_cluster_summarize_texts(texts, level)
        results[level] = (df_clusters, df_summary)

        unique_clusters = df_summary["cluster"].nunique()
        if level < n_levels and unique_clusters > 1:
            new_texts = df_summary["summaries"].tolist()
            next_level_results = self.recursive_embed_cluster_summarize(
                new_texts, level + 1, n_levels
            )
            results.update(next_level_results)
        return results

    def add_documents(self, documents: List[Document]):
        texts: List[str] = []
        for doc in documents:
            texts.append(doc.page_content)

        summarized_documents = self.get_summarized_documents(texts)
        all_documents = documents + summarized_documents

        self.vector_store.from_documents(documents=all_documents, embedding=self.embedding_model, persist_directory="./chroma_db")
        self.vector_store.persist()
        self.documents = all_documents

    def get_summarized_documents(self, texts: List[str]):
        results = self.recursive_embed_cluster_summarize(texts, level=1, n_levels=3)
        for level in sorted(results.keys()):
            summaries = results[level][1]["summaries"].tolist()
        
        return [Document(page_content=summary, metadata={"source":"summary"}) for summary in summaries]
        

    def get_db(self):
        return self.vector_store
    
    def as_retriever(self, k: int = 2):
        return self.vector_store.as_retriever(search_kwargs={"k": k})
    
    def similarity_search(self, question):
        return self.vector_store.similarity_search(question)
    
    def get_all_documents(self) -> List[Document]:
        return self.documents
    
    def get_all_texts(self) -> List[str]:
        return self.texts

    def similarity_search_with_score(self, question):
        return self.vector_store.similarity_search_with_relevance_scores(question)