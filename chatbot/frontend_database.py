import streamlit as st
from backend_database import workflow,retrieve_all_threads,load_titles,save_title
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import uuid
from langchain_groq import ChatGroq

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
    st.session_state["chat_threads"] = retrieve_all_threads()

if "chat_titles" not in st.session_state:
    loaded_titles = load_titles()
    st.session_state["chat_titles"] = loaded_titles


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
for thread_id in st.session_state["chat_threads"]:

    # Get title for this thread
    title = st.session_state["chat_titles"].get(
        thread_id,"new chat"
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

        st.markdown(chat_message["content"])


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
        title = heading.content.strip()

        st.session_state["chat_titles"][current_thread_id] = title

        save_title(current_thread_id, title)


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
        },
        "metadata":{
            "thread_id": current_thread_id
        },
        "run_name":"chat_turn"
    }


    # ------------------------------------------------------------
    # Get AI response
    # ------------------------------------------------------------

    with st.chat_message("assistant"):

        status_holder = {"box": None}

        def ai_only_stream():
            for message_chunk, metadata in workflow.stream(
                {
                    "message": [
                        HumanMessage(content=user_input)
                    ]
                },
                config=config,
                stream_mode="messages",
            ):
                # Lazily create and update the same status container when a tool runs.
                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, "name", "tool")
                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(
                            f"🔧 Using `{tool_name}` …", expanded=True
                        )
                    else:
                        status_holder["box"].update(
                            label=f"🔧 Using `{tool_name}` …",
                            state="running",
                            expanded=True,
                        )

                # Stream only assistant tokens.
                if isinstance(message_chunk, AIMessage):
                    yield message_chunk.content

        ai_message = st.write_stream(ai_only_stream())


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