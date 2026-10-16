import streamlit as st
from backend import workflow
from langchain_core.messages import HumanMessage
import uuid
config = {'configurable':{"thread_id":1}}
# --------utility functions-----
def generate_thread_id():
    return str(uuid.uuid4())
def 
# --------session setup-----
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []
if 'thread_id' not in st.session_state:
    st.session_state['thread_id']=generate_thread_id()
# ---------sidebar ui-----
st.sidebar.title("Chatty")
st.sidebar.button("New Chat")
st.sidebar.header("Converstions")
# for thread_id in st.session_state['thread_id']:
#     st.sidebar.button(thread_id)
st.sidebar.text(st.session_state['thread_id'])
# ---------main ui-----------
user_input=st.chat_input("type here")

for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])

if user_input:
    st.session_state['message_history'].append({'role':"user","content":user_input})

    with st.chat_message('user'):
        st.text(user_input)

    # st.session_state['message_history'].append({'role':"assistant","content":chatanswer})
    config={'configurable':{"thread_id":st.session_state['thread_id']}}
    with st.chat_message('assistant'):
        ai_message=st.write_stream(
            message_chunk.content for message_chunk,metadata in workflow.stream(
                {"message":[HumanMessage(content=user_input)]},
                config = config,  
                stream_mode='messages' 
            )
        ) 
    st.session_state['message_history'].append({'role':"assistant","content":ai_message})
