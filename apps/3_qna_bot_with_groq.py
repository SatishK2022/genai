from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
import streamlit as st

model = ChatGroq(model="openai/gpt-oss-20b", streaming=True)
search = GoogleSerperAPIWrapper()

if "memory" not in st.session_state:
    st.session_state.memory = InMemorySaver()
    st.session_state.history = []


agent = create_agent(
    model=model,
    checkpointer=st.session_state.memory,
    tools=[search.run],
    system_prompt="You are an helpful assistant",
)

# Web Interface
st.subheader("QnA Chatbot")

for message in st.session_state.history:
    role = message["role"]
    content = message["content"]
    st.chat_message(role).markdown(content)

query = st.chat_input(placeholder="Ask anything....")

if query:
    st.chat_message("user").markdown(query)
    st.session_state.history.append({"role": "user", "content": query})

    response = agent.stream(
        {"messages": [{"role": "user", "content": query}]},
        {"configurable": {"thread_id": "123"}},
        stream_mode="messages",
    )

    # answer = response["messages"][-1].content
    # st.chat_message("ai").markdown(answer)

    ai_container = st.chat_message("ai")
    with ai_container:
        space = st.empty()

        message = ""

        for chunk in response:
            message = message + chunk[0].content
            space.write(message)

        st.session_state.history.append({"role": "assistant", "content": message})