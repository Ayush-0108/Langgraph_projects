from langgraph.graph import StateGraph,START,END
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_groq import ChatGroq
from langgraph.graph.message import add_messages
from typing import TypedDict,Annotated,Dict,Optional,Any
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import sqlite3
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
import requests
import os
import tempfile

load_dotenv()

model = ChatGroq(model = "openai/gpt-oss-20b")
embeddings = HuggingFaceEmbeddings(model = "sentence-transformers/all-MiniLM-L6-v2")
_THREAD_RETRIEVERS: Dict[str, Any] = {}
_THREAD_METADATA: Dict[str, dict] = {}


def _get_retriever(thread_id: Optional[str]):
    """Fetch the retriever for a thread if available."""
    if thread_id and thread_id in _THREAD_RETRIEVERS:
        return _THREAD_RETRIEVERS[thread_id]
    return None


def ingest_pdf(file_bytes: bytes, thread_id: str, filename: Optional[str] = None) -> dict:
    """
    Build a FAISS retriever for the uploaded PDF and store it for the thread.

    Returns a summary dict that can be surfaced in the UI.
    """
    if not file_bytes:
        raise ValueError("No bytes received for ingestion.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(docs)

        vector_store = FAISS.from_documents(chunks, embeddings)
        retriever = vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 4}
        )

        _THREAD_RETRIEVERS[str(thread_id)] = retriever
        _THREAD_METADATA[str(thread_id)] = {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }

        return {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }
    finally:
        # The FAISS store keeps copies of the text, so the temp file is safe to remove.
        try:
            os.remove(temp_path)
        except OSError:
            pass

@tool
def rag_tool(query: str, config: RunnableConfig) -> dict:
    """
    Retrieve relevant information from the uploaded PDF for this chat thread.
    The current chat thread is provided automatically.
    """
    thread_id = config.get("configurable", {}).get("thread_id")
    if thread_id is not None:
        thread_id = str(thread_id)

    retriever = _get_retriever(thread_id)
    if retriever is None:
        return {
            "error": "No document indexed for this chat. Upload a PDF first.",
            "query": query,
        }

    result = retriever.invoke(query)
    context = [doc.page_content for doc in result]
    metadata = [doc.metadata for doc in result]

    return {
        "query": query,
        "context": context,
        "metadata": metadata,
        "source_file": _THREAD_METADATA.get(str(thread_id), {}).get("filename"),
    }

search_tool = DuckDuckGoSearchRun(region="us-en")
@tool
def calculator(first_num:float,second_num:float,operation:str)->dict:
    """
    Perform A basic arithematic operation on two numbers.
    Supported operations:add,sub,mul,div
    """
    try:
        if operation =="add":
            result = first_num+second_num
        elif operation =="sub":
            result = first_num-second_num
        elif operation =="mul":
            result = first_num*second_num
        elif operation =="div":
            if second_num>0:    
                result = first_num/second_num
            else:
                return {'error':"Division by zero jaake maths padh"}
        else:
            return {"error":f"unsuported operations{operation}"}
    except Exception as e:
        return {"error":str(e)}

@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') 
    using Alpha Vantage with API key in the URL.
    """
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=XZW9VTJGTHA8GXEU"
    r = requests.get(url)
    return r.json()

# Define tools
tools = [get_stock_price,search_tool,calculator,rag_tool]
llm_with_tools = model.bind_tools(tools)

#Define State
class ChatState(TypedDict):
    message: Annotated[list[BaseMessage],add_messages]

#Define Node Func
def ChatBot(state:ChatState)->ChatState:
    message = state["message"]
    response = llm_with_tools.invoke(
        [
            SystemMessage(
                content=(
                    "For questions about an uploaded PDF, call rag_tool before "
                    "answering. Do not claim that no PDF is loaded or ask the user "
                    "to upload it unless rag_tool reports that no document is "
                    "indexed for this chat. Base document answers on the retrieved "
                    "context, and say when the context does not contain the answer."
                )
            ),
            *message,
        ]
    )
    return {"message":[response]}

tool_node = ToolNode(tools,messages_key="message")

conn=sqlite3.connect(database='chatbot.db',check_same_thread=False)
conn.execute(
    """
    CREATE TABLE IF NOT EXISTS chat_titles (
        thread_id TEXT PRIMARY KEY,
        title TEXT NOT NULL
    )
    """
)
conn.commit()

#Define Graph
checkpointer = SqliteSaver(conn=conn)
graph = StateGraph(ChatState)
graph.add_node("ChatBot",ChatBot)
graph.add_node("tools",tool_node)

graph.add_edge(START,"ChatBot")
graph.add_conditional_edges("ChatBot",lambda state: tools_condition(state, messages_key="message"))
graph.add_edge('tools',"ChatBot")

workflow = graph.compile(checkpointer=checkpointer)
def retrieve_all_threads():
    latest_checkpoint = {}

    for checkpoint in checkpointer.list(None):
        thread_id = str(checkpoint.config["configurable"]["thread_id"])
        timestamp = checkpoint.checkpoint["ts"]

        if thread_id not in latest_checkpoint or timestamp > latest_checkpoint[thread_id]:
            latest_checkpoint[thread_id] = timestamp

    return sorted(
        latest_checkpoint,
        key=lambda thread_id: latest_checkpoint[thread_id],
        reverse=True,
    )
# thread_id = 1
# config= {'configurable':{'thread_id':thread_id}}
# response=workflow.invoke({"message":"what is the capital of india,with acknowledgement to my name "},config=config)
# print(response)
def load_titles():
    cursor = conn.execute(
        "SELECT thread_id, title FROM chat_titles"
    )

    return dict(cursor.fetchall())

def save_title(thread_id, title):
    conn.execute(
        """
        INSERT INTO chat_titles (thread_id, title)
        VALUES (?, ?)
        ON CONFLICT(thread_id)
        DO UPDATE SET title=excluded.title
        """,
        (str(thread_id), title)
    )
    conn.commit()