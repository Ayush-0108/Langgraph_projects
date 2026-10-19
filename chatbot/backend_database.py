from langgraph.graph import StateGraph,START,END
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_groq import ChatGroq
from langgraph.graph.message import add_messages
from typing import TypedDict,Annotated
from dotenv import load_dotenv
import sqlite3
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
import requests

load_dotenv()

#Define model
model = ChatGroq(model="openai/gpt-oss-20b")

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
tools = [get_stock_price,search_tool,calculator]
llm_with_tools = model.bind_tools(tools)

#Define State
class ChatState(TypedDict):
    message: Annotated[list[BaseMessage],add_messages]

#Define Node Func
def ChatBot(state:ChatState)->ChatState:
    message = state["message"]
    response = llm_with_tools.invoke(message)
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