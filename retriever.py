import os
from typing import List, Dict, Union, Optional
from pathlib import Path
import openai
import pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone as LangchainPinecone
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import numpy as np
from tqdm import tqdm
load_dotenv()


class SlidingWindowTextSplitter:
    # initialize the embedding system
    # sliding window in order to save complete question context
    def __init__(
        self,
        window_size: int = 1000,
        stride: int = 200,
        min_length: int = 100
    ):
        self.window_size = window_size
        self.stride = stride
        self.min_length = min_length

    def split_text(self, text: str) -> List[str]:
        if not text:
            return []

        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = min(start + self.window_size, text_length)
            chunk = text[start:end]

            # Only add chunks that meet minimum length
            # Find first space for clean start
            if len(chunk) >= self.min_length:
                if start > 0:
                    first_space = chunk.find(' ')
                    if first_space != -1:
                        chunk = chunk[first_space+1:]
                        
                if end < text_length:
                    last_period = chunk.rfind('.')
                    last_space = chunk.rfind(' ')
                    break_point = max(last_period, last_space)
                    if break_point != -1:
                        chunk = chunk[:break_point+1]

                chunks.append(chunk)
            start += self.stride

        return chunks

class AdvancedEmbeddingSystem:
    def __init__(
        self,
        index_name: str,
        environment: str = "us-east-1",
        dimension: int = 1536,
        window_size: int = 1000,
        stride: int = 200,
        min_length: int = 100
    ):
        
        self.embedder = OpenAIEmbeddings()
        self.pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
        self.index_name = index_name
        self.environment = environment
        self.text_splitter = SlidingWindowTextSplitter(
            window_size=window_size,
            stride=stride,
            min_length=min_length
        )
        if self.index_name not in [index.name for index in self.pc.list_indexes()]:
            self.pc.create_index(
                name=index_name,
                dimension=dimension, 
                metric='euclidean',
                spec=ServerlessSpec(
                    cloud='aws',
                    region=self.environment
                )
            )
        else:
            print(f"Index {index_name} already exists.")
        

    def process_document(
        self, 
        file_path: Union[str, Path],
        metadata: Optional[Dict] = None
    ) -> LangchainPinecone:
        file_path = Path(file_path)
        if file_path.suffix.lower() == '.pdf':
            loader = PyPDFLoader(str(file_path))
            documents = loader.load_and_split()
        elif file_path.suffix.lower() in ['.txt', '.md']:
            loader = TextLoader(str(file_path))
            documents = loader.load()
        else:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")
        texts = []
        for doc in documents:
            chunks = self.text_splitter.split_text(doc.page_content)
            for chunk in chunks:
                doc_metadata = doc.metadata.copy()
                if metadata:
                    doc_metadata.update(metadata)
                texts.append({
                    'content': chunk,
                    'metadata': doc_metadata
                })
        print(f"Creating embeddings for {len(texts)} chunks...")
        vectorstore = LangchainPinecone.from_texts(
            texts=[t['content'] for t in texts],
            embedding=self.embedder,
            index_name=self.index_name,
            metadatas=[t['metadata'] for t in texts]
        )

        return vectorstore

if __name__ == "__main__":
    embedding_system = AdvancedEmbeddingSystem(
        index_name="bloom-testing",
        window_size=1000,
        stride=200
    )

    vectorstore = embedding_system.process_document(
        "/Users/yonganyu/Desktop/Edu-benchmark/Calculus_McGill/Problem sets & Solutions/Math 140 Tutorial 7.pdf",
        metadata={"subject": "mathematics", "type": "tutorial"}
    )
    
