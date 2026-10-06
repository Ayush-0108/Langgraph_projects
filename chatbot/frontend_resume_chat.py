import streamlit as st
from backend import workflow
from langchain_core.messages import HumanMessage, AIMessage
import uuid
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
model = ChatGroq(model="openai/gpt-oss-20b")

# -------------------- Utility Functions ------------------------

def generate_thread_id():
    return str(uuid.uuid4())


def reset_chat():
    thread_id = generate_thread_id()

    st.session_state["thread_id"] = thread_id

    add_thread(thread_id)

    st.session_state["message_history"] = []


def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def load_conversation(thread_id):
    return workflow.get_state(
        config={
            "configurable": {
                "thread_id": thread_id
            }
        }
    ).values.get("message", [])


# -------------------- Session State ----------------------------

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = []

if "chat_titles" not in st.session_state:
    st.session_state["chat_titles"] = {}


# Add current thread to thread list
add_thread(st.session_state["thread_id"])


# -------------------- Sidebar ---------------------------------

st.sidebar.title("LangGraph Chatbot")


# New Chat button
if st.sidebar.button("New Chat"):

    reset_chat()

    # Rerun so the sidebar immediately updates
    st.rerun()


st.sidebar.header("My Conversation")


# Display conversations newest first
for thread_id in st.session_state["chat_threads"][::-1]:

    # Get title for this thread
    title = st.session_state["chat_titles"].get(
        thread_id,
        "New Chat"
    )

    # Unique key for every button
    if st.sidebar.button(
        title,
        key=f"chat_{thread_id}"
    ):

        # Make this thread the active thread
        st.session_state["thread_id"] = thread_id

        # Load messages from LangGraph
        conversation = load_conversation(thread_id)

        temp_messages = []

        for msg in conversation:

            if isinstance(msg, HumanMessage):

                role = "user"

            elif isinstance(msg, AIMessage):

                role = "assistant"

            else:

                continue

            temp_messages.append({
                "role": role,
                "content": msg.content
            })

        # Replace current chat with selected conversation
        st.session_state["message_history"] = temp_messages

        # Rerun so selected conversation is displayed
        st.rerun()


# -------------------- Chat Input ------------------------------

user_input = st.chat_input("Type here")


# -------------------- Display Previous Messages ---------------

for chat_message in st.session_state["message_history"]:

    with st.chat_message(chat_message["role"]):

        st.text(chat_message["content"])


# -------------------- New User Message -------------------------

if user_input:

    # Get currently active thread
    current_thread_id = st.session_state["thread_id"]


    # ------------------------------------------------------------
    # Create title using first user message
    # ------------------------------------------------------------

    if current_thread_id not in st.session_state["chat_titles"]:
        heading = model.invoke(f"Generate a short title of 3-5 words for this conversation. "
    f"Return only the title.\n\n"
    f"User message: {user_input}")
        st.session_state["chat_titles"][current_thread_id] = heading.content


    # ------------------------------------------------------------
    # Add user message to Streamlit history
    # ------------------------------------------------------------

    st.session_state["message_history"].append({
        "role": "user",
        "content": user_input
    })


    # Display user message
    with st.chat_message("user"):

        st.text(user_input)


    # ------------------------------------------------------------
    # LangGraph configuration
    # ------------------------------------------------------------

    config = {
        "configurable": {
            "thread_id": current_thread_id
        }
    }


    # ------------------------------------------------------------
    # Get AI response
    # ------------------------------------------------------------

    with st.chat_message("assistant"):

        ai_message = st.write_stream(

            message_chunk.content

            for message_chunk, metadata in workflow.stream(

                {
                    "message": [
                        HumanMessage(content=user_input)
                    ]
                },

                config=config,

                stream_mode="messages"
            )
        )


    # ------------------------------------------------------------
    # Save AI response in Streamlit history
    # ------------------------------------------------------------

    st.session_state["message_history"].append({
        "role": "assistant",
        "content": ai_message
    })


    # ------------------------------------------------------------
    # Rerun so sidebar title updates immediately
    # ------------------------------------------------------------

    st.rerun()