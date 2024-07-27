import os
import tiktoken
import pandas as pd
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyMuPDFLoader
from typing import List
from langchain.schema import Document
from unstructured.partition.pdf import partition_pdf

class DocumentLoader:
    def __init__(self):
        pass

    def num_tokens_from_string(self, string: str, encoding_name: str) -> int:
        encoding = tiktoken.get_encoding(encoding_name)
        num_tokens = len(encoding.encode(string))
        return num_tokens

    def load_document(self, file_path):
        file_extension = os.path.splitext(file_path)[1].lower()

        if file_extension == '.txt':
            docs = self.load_text(file_path)
        elif file_extension == '.pdf':
            docs = self.load_pdf(file_path)
        elif file_extension == '.xlsx':
            docs = self.load_excel(file_path)
        else:
            raise ValueError(f"{file_extension} is unsupported file type.")
        return docs
        
    def load_text(self, file_path):
        try:
            loader = TextLoader(file_path)
            document = loader.load()
            return document
        except FileNotFoundError:
            raise ValueError(f"Couldn't find {file_path}")
        except Exception as e:
            raise ValueError(f"An error occurred while loading the text file : {str(e)}")

    def load_pdf(self, file_path):
        try:
            loader = PyMuPDFLoader(file_path)
            text_document = loader.load()
            other_document = self.load_images_and_tables(file_path)
            return text_document + other_document
        except FileNotFoundError:
            raise ValueError(f"Couldn't fild {file_path}")
        except Exception as e:
            raise ValueError(f"An error occurred while loading the pdf file : {str(e)}")

    def load_excel(self, file_path):
        try:
            df = pd.read_excel(file_path)
            document = []
            for index, row in df.iterrows():
                content = row.to_string()
                document.append(Document(page_content=content, metadata={'source': file_path}))
            return document
        except FileNotFoundError:
            raise ValueError(f"Couldn't fild {file_path}")
        except Exception as e:
            raise ValueError(f"An error occurred while loading the excel file : {str(e)}")
        
    def load_images_and_tables(self, file_path):
        try:
            pdf_elements = self.extract_pdf_elements(file_path)
            tables, images = self.categorize_elements(pdf_elements)
            tables_documents = [Document(page_content=table, metadata={"type": "table", "source": file_path}) for table in tables]
            images_documents = [Document(page_content=image, metadata={"type": "image", "source": file_path}) for image in images]
            return tables_documents + images_documents
        except Exception as e:
            raise ValueError(f"An error occured while loading the pdf file with images : {str(e)}")
        
    def extract_pdf_elements(self, file_path):
        try:
            return partition_pdf(
                filename=file_path,
                extract_images_in_pdf=True,
                infer_table_structure=True,
                strategy="hi_res",
                extract_image_block_output_dir="./images/",
                languages=["eng", "kor"]
            )
        except Exception as e:
            raise ValueError(f"An error occurred while extracting elements from the pdf file : {str(e)}")
    
    def categorize_elements(self, pdf_elements):
        tables = []
        images = []
        for element in pdf_elements:
            if "unstructured.documents.elements.Image" in str(type(element)):
                images.append(str(element))
            elif "unstructured.documents.elements.Table" in str(type(element)):
                tables.append(str(element))
        return tables, images