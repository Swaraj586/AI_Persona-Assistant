import os

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import Language, MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_community import embeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_SECRET")

loader = TextLoader("info.txt", encoding="utf-8")
# text_splitter = RecursiveCharacterTextSplitter.from_language(language=Language.MARKDOWN, chunk_size=2000, chunk_overlap=1000)
text_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[('##','Section')])
docs = loader.load()
splits = text_splitter.split_text(docs[0].page_content)
for i, split in enumerate(splits):
    print(f"Split {i+1}:")
    print(split.page_content)


embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2")
vector_store = FAISS.from_documents(splits, embeddings)

print(vector_store.index_to_docstore_id)

vector_store.save_local("faiss_index")
print("FAISS index saved locally as 'faiss_index'.")