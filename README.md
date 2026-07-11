# AI Persona Assistant

An interactive, production-ready FastAPI backend application that simulates a personal AI clone/twin using Retrieval-Augmented Generation (RAG). The system consumes structured raw data, vectors it dynamically via Google Gemini Embeddings, handles similarity querying using a FAISS index, and responds identically to the user's voice using Google's `gemini-3.1-flash-lite` model. 

It implements isolated session-based conversational memories, dropping stale connections gracefully over time.

---

## System Architecture

The following lifecycle details the two core processes: the offline ingestion pipeline (`load.py`) and the online production serving pipeline (`main.py`).

![System Architecture](./architecture.svg)

---

## File Directory Blueprint

*   `info.txt`: The personal raw contextual knowledge base written using explicit markdown section structures (`## Header`).
*   `load.py`: Data ingestion script. Parses raw context records, applies semantic split transformations, wraps text vectors using Google AI embedding models, and saves an indexed repository locally.
*   `main.py`: Interactive serving layer built on FastAPI. Hosts explicit endpoints for context retrieval orchestration, concurrent chat session management, and output serialization.

---

## Step-by-Step Quick Start

### 1. Environmental Provisions
Construct a localized `.env` variables file in the project workspace:

```env
    GEMINI_SECRET="your_google_gemini_api_key_here"
```

### 2. Package Install
To install packages from a requirements.txt file, run the following command in your terminal or command prompt:

```bash
    pip install -r requirements.txt
```

### 3. Build & Populate Vector Base
Execute the document ingestion component to tokenize the unstructured database asset (info.txt) into structured FAISS index matrices:
```bash
   python load.py
```
Expected Terminal Outputs: Logs confirmation arrays verifying structural page chunks and dumps a local index directory named faiss_index/.

### 4. Boot Up Serving API Engine
Launch the active async processing server using an underlying Uvicorn ASGI configuration loop:
```bash
   uvicorn main:app --reload --port 8000
```

## Essential Application Routing

### 1. POST `/ask`

Submits a brand new tracking question string relative to a concrete session token identifier.
```JSON
   {
      "question": "What major software engineering projects have you completed?",
      "session_id": "user_session_alpha_99"
   }
```
JSON success response:
```JSON
   {
       "reply": "I built an innovative cloud architecture utilizing LangChain, FAISS, and Gemini integrations..."
   }
```
### 2. DELETE `/cleanup_sessions`
Forces immediate eviction and structural teardown of a designated session profile data cache.

Query parameter structure: `?session_id=user_session_alpha_99`

### 3. GET `/health`
Returns system status metrics showing runtime stability and total concurrent active sessions tracked within memory buffers.


## Architecture Customization Engine
To tailor this application engine explicitly to your persona:
1. Open `info.txt` and substitute standard brackets `[Add info here]` with explicit details regarding your engineering projects, credentials, and experience.
2. Open `main.py` and customize the base Prompt Template configuration rules inside the `ChatPromptTemplate.from_messages` sequence block to align directly with your desired tone, title, and conversational style.




