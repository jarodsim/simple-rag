import os
from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_cloudflare import ChatCloudflareWorkersAI

load_dotenv()

embeddings = NVIDIAEmbeddings(
    model="nvidia/nemotron-3-embed-1b", api_key=os.getenv("NVIDIA_API_KEY")
)

vectorstore = FAISS.load_local(
    "faiss_index", embeddings, allow_dangerous_deserialization=True
)

llm = ChatCloudflareWorkersAI(
    model_name="@cf/meta/llama-3.1-8b-instruct", temperature=0.1
)

template = """Responda à pergunta baseando-se APENAS no contexto fornecido abaixo.
Se a resposta não estiver no contexto, diga que não sabe.

Contexto:
{context}

Pergunta: {question}
"""
prompt = ChatPromptTemplate.from_template(template)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


def get_answer(question: str):
    return rag_chain.invoke(question)
