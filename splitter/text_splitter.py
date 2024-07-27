from langchain.text_splitter import RecursiveCharacterTextSplitter

class TextSplitter:
    def __init__(self, chunk_size=500, overlap_size=25):
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
    
    def split_documents(self, document):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, 
            chunk_overlap=self.overlap_size, 
            length_function=len
        )
        chunks = splitter.split_documents(document)
        return chunks
    
    def split_texts(self, texts):
        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=self.chunk_size,
            chunk_overlap=self.overlap_size,
        )
        chunks = splitter.split_text(texts)
        return chunks