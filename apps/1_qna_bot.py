from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
import streamlit as st

load_dotenv()
# llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")
llm = ChatOpenAI(model="gpt-4o")

st.title("QnA Chatbot")
st.markdown("QnA Chatbot with Langchain and Streamlit")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    role = message["role"]
    content = message["content"]
    st.chat_message(role).markdown(content)


query = st.chat_input("Ask anything...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    st.chat_message("user").markdown(query)

    with st.spinner("Thinking..."):
        res = llm.invoke(query)
        ai_response = res.content

    st.session_state.messages.append({"role": "ai", "content": ai_response})
    st.chat_message("ai").markdown(ai_response)


# while True:
#     query = input("User: ")

#     if query.lower() in ["exit", "quit", "bye"]:
#         print("GoodBye")
#         break

#     result = llm.invoke(query)
#     print("AI: ", result.content, "\n")
