from pydantic import BaseModel
from langgraph.graph import StateGraph, START, END


class GreetState(BaseModel):
    message: str = ""


graph = StateGraph(GreetState)


# Nodes - Python functions
def Greet(state: GreetState):
    state.message = f"{state.message}! All good."
    return state

def upperCase(state: GreetState):
    state.message = state.message.upper()
    return state



graph.add_node("greet", Greet)
graph.add_node("upper", upperCase)

graph.add_edge(START, "greet")
graph.add_edge("greet", "upper")
graph.add_edge("upper", END)


finalGraph = graph.compile()

res = finalGraph.invoke({"message": "I Love langgraph"})

print(res)
