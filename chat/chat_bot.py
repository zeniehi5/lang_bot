from loader.document_loader import DocumentLoader
from retriever.similarity_retriever import Retriever
from generator.response_generator import ResponseGenerator

class ChatBot:
    def __init__(self):
        self.doc_loader = DocumentLoader()
        self.retriever = Retriever(ensemble=True, weights=[{'keyword': 0.3}, {'similarity': 0.7}])
        self.response_generator = ResponseGenerator()

    def load_documents(self, file_path):
        return self.doc_loader.load_document(file_path)
        
    def get_response(self, question, store):
        try:
            context = self.retriever.retrieve(store, question)
            response = self.response_generator.generate(question, context)
        except Exception as e:
            raise ValueError(f"응답 생성 중 에러가 발생했습니다 : {str(e)}")
        return response