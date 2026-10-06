from langgraph.graph import StateGraph,START,END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_groq import ChatGroq
from langgraph.graph.message import add_messages
from typing import TypedDict,Annotated
from dotenv import load_dotenv

load_dotenv()

#Define model
model = ChatGroq(model="openai/gpt-oss-20b")

#Define State
class ChatState(TypedDict):
    message: Annotated[list[BaseMessage],add_messages]

#Define Node Func
def ChatBot(state:ChatState)->ChatState:
    message = state["message"]
    response = model.invoke(message)
    return {"message":[response]}

#Define Graph
checkpointer = MemorySaver()
graph = StateGraph(ChatState)
graph.add_node("ChatBot",ChatBot)

graph.add_edge(START,"ChatBot")
graph.add_edge("ChatBot",END)

workflow = graph.compile(checkpointer=checkpointer)

# thread_id = 1
# config= {'configurable':{'thread_id':thread_id}}
# while