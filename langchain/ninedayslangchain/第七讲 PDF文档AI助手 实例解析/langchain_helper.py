#from langchain.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.indexes.vectorstore import VectorstoreIndexCreator
from langchain.vectorstores.chroma import Chroma
#from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain.indexes.vectorstore import VectorStoreIndexWrapper

from dotenv import load_dotenv

import os
os.environ['OPENAI_API_KEY'] = "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"

load_dotenv()

file_path = "zwd.pdf"

local_persist_path = "vector_store"

def get_index_path(index_name):
    return os.path.join(local_persist_path, index_name)

def load_pdf_and_save_to_index(file_path, index_name):

    loader = PyPDFLoader(file_path)

    index = VectorstoreIndexCreator(vectorstore_kwargs={"persist_directory":get_index_path(index_name)}).from_loaders([loader])

    index.vectorstore.persist()

def load_index(index_name):
    index_path = get_index_path(index_name)
    embedding = OpenAIEmbeddings()
    vectordb = Chroma(persist_directory=index_path, embedding_function=embedding)

    return VectorStoreIndexWrapper(vectorstore=vectordb)

def query_index_lc(index, query):
    ans = index.query_with_sources(query, chain_type="map_reduce")
    return ans['answer']
#ans = index.query_with_sources("how to write good code?", chain_type="map_reduce")

#print(ans)
#load_pdf_and_save_to_index(file_path, "test")

#index = load_index("test")

#ans = index.query_with_sources("how to write good code?", chain_type="map_reduce")
#print(ans)