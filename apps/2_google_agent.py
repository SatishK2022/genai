from dotenv import load_dotenv
load_dotenv()

from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

model = ChatGroq(model="openai/gpt-oss-20b")
search = GoogleSerperAPIWrapper()

agent = create_agent(
    model=model,
    tools=[search.run],
    checkpointer=InMemorySaver(),
    system_prompt="You are a helpful assistant",
)

while True:
    query = input("You: ")

    if query.lower() in ["bye", "exit"]:
        print("Good Bye")
        break

    result = agent.invoke(
        {"messages": [{"role": "user", "content": query}]},
        {"configurable": {"thread_id": "test"}},
    )
    print("AI: ", result["messages"][-1].content)
