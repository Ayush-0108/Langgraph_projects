import streamlit as st
from backend import workflow
from langchain_core.messages import HumanMessage
config = {'configurable':{"thread_id":1}}
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []


user_input=st.chat_input("type here")

for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])

if user_input:
    st.session_state['message_history'].append({'role':"user","content":user_input})

    with st.chat_message('user'):
        st.text(user_input)

    response = workflow.invoke({'message':[HumanMessage(content=user_input)]},config=config)

    chatanswer = response['message'][-1].content
    st.session_state['message_history'].append({'role':"assistant","content":chatanswer})

    with st.chat_message('assistant'):
        st.text(chatanswer)
