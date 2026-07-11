from fastapi import FastAPI, HTTPException, BackgroundTasks
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_ollama.chat_models import ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda, chain
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from pydantic import BaseModel
import time
from typing import Dict
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings


load_dotenv()

app = FastAPI()
origins = [
    "*"  # Your production frontend
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


sessions_db: Dict[str, dict] = {}
session_expiry_time = 1800 

class QuestionRequest(BaseModel):
    question: str
    session_id: str

class Reply(BaseModel):
    reply: str

os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_SECRET")

chat = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2")
# embeddings = OllamaEmbeddings(model="embeddinggemma")
vector_store = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
# chat = ChatOllama(model="gemma3:4b", temperature=0.2)
chat_history = ChatMessageHistory()
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an AI version of [YourName]. You are given the following context, information from a document, and a question. Answer the question based on the context provided just like how [YourName] will answer. If the answer is not contained within the context, respond with "I don't know." or if it is too personal answer with "I cannot answer that." Do not make up answers. If the user is greeting you, respond with a greeting. If the user is asking for your name, respond with "I am [YourName].", and if the user is asking for your profession, respond with "I am a [YourProfession]".
    When answering questions like "tell me about yourself", "who are you?", "what do you do?", respond with a short introduction about yourself based on the context provided include or highlight projects worked on (not all but major ones in short).
    Don' t reveal your own AI identity, You are representing [YourName], answer as [YourName] would. 
     Context:
    {context}

    The reply should be short and concise, within 100 words."""),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}")
])

def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])



def get_update_session(session_id: str):
    if session_id not in sessions_db:
        sessions_db[session_id] = {"last_accessed": time.time(), "chat_history": ChatMessageHistory()}

    else:
        sessions_db[session_id]["last_accessed"] = time.time()

    return sessions_db[session_id]["chat_history"]

def cleanup_sessions():
    current_time = time.time()
    expired_sessions = [session_id for session_id, session_data in sessions_db.items() if current_time - session_data["last_accessed"] > session_expiry_time]

    for session_id in expired_sessions:
        del sessions_db[session_id]
        print(f"Session {session_id} has been cleaned up due to inactivity.")


@app.post("/ask")
async def ask_question(question_request: QuestionRequest, background_tasks: BackgroundTasks):
    background_tasks.add_task(cleanup_sessions)
    try:
        
        question = question_request.question
        session_id = question_request.session_id

        chat_history = get_update_session(session_id)
        parallel_chain = RunnableParallel({
            'context': retriever | RunnableLambda(format_docs),
            'question': RunnablePassthrough(),
            'history': RunnableLambda(lambda x: chat_history.messages)
        })
        chain = parallel_chain | prompt | chat | StrOutputParser()

        reply = chain.invoke(question)
        chat_history.add_user_message(question)
        chat_history.add_ai_message(reply)
        return Reply(reply=reply)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/cleanup_sessions")
async def cleanup_sessions_endpoint(session_id: str):
    if session_id in sessions_db:
        del sessions_db[session_id]
        return {"message": f"Session {session_id} has been cleaned up."}
    else:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found.")
    
@app.get("/health")
async def health_check():
    return {"status": "ok", "active_sessions": len(sessions_db)}

