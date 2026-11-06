import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

# 1 - Carregando documento
loader = TextLoader("documento.md", encoding="utf-8")
docs = loader.load()

# 2 - Dividindo em chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512, chunk_overlap=50, separators=["\n\n", "\n", ".", " "]
)

chunks = text_splitter.split_documents(docs)

# 3 - Criando embeddings
embeddings = NVIDIAEmbeddings(
    model="nvidia/nemotron-3-embed-1b", api_key=os.getenv("NVIDIA_API_KEY")
)

# 4 - FAISS
verctorstore = FAISS.from_documents(chunks, embeddings)
verctorstore.save_local("faiss_index")
print("Ingestão concluída")
