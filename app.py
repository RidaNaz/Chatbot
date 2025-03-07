import os
import chainlit as cl
from typing import Literal, Union
from langchain_core.tools import tool
from psycopg_pool import ConnectionPool
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import MessagesState
from langgraph.graph import END, StateGraph, START
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema.runnable.config import RunnableConfig
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, RemoveMessage
# from langgraph.checkpoint.memory import MemorySaver


NEON_DB_URI = os.getenv("NEON_DB_URI")

os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')
os.environ['SERPER_API_KEY'] = os.getenv('SERPER_API_KEY')
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "chatbot"

# Postgres DB

# Connection pool for efficient database access
connection_kwargs = {"autocommit": True, "prepare_threshold": 0}

# Create a persistent connection pool
pool = ConnectionPool(conninfo=NEON_DB_URI, max_size=30, kwargs=connection_kwargs)

# Initialize PostgresSaver checkpointer
checkpointer = PostgresSaver(pool)
checkpointer.setup()  # Ensure database tables are set up

# memory = MemorySaver()

# State

class State(MessagesState):
    summary: str

# SerpAPI Search Tool Node

search_tool = GoogleSerperAPIWrapper()

# Define a tool using the correct decorator
@tool
def search(query: str) -> str:
    """Search the web using SerpAPI."""
    return search_tool.run(query)

tools = [search]

tool_node = ToolNode(tools=[search])

# Model

model = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0.2
)

# We can bind the llm to a tool definition
model = model.bind_tools(tools)

# Summarization

def summarize_conversation(state: State):
    """
    Summarizes the conversation if the number of messages exceeds 6 messages.
    
    Args:
        state (State): The current conversation state.
        model (object): The model to use for summarization.

    Returns:
        Dict[str, object]: A dictionary containing updated messages.
    """
   
    # Get any existing summary
    summary = state.get("summary", "")

    # Create summarization prompt based on whether there is an existing summary
    if summary:
        summary_message = (
            f"This is the summary of the conversation to date: {summary}\n\n"
            "Extend the summary by taking into account the new messages above:"
        )
    else:
        summary_message = "Create a summary of the conversation above:"

    # Add the summarization prompt to the conversation history
    messages = state["messages"] + [HumanMessage(content=summary_message)]
    response = model.invoke(messages)

    # Delete all but the 2 most recent messages
    delete_messages = [RemoveMessage(id=getattr(m, "id", None)) for m in state["messages"][:-2]]
    
    return {"summary": response.content, "messages": delete_messages}


# Conditional Function

def select_next_node(state: State) -> Union[Literal["tools", "summarize"], str]:

    messages = state["messages"]
    last_message = messages[-1]

    # If there are more than six messages, route to "summarize_conversation"
    if len(messages) > 6:
        return "summarize"
    
    # If the LLM makes a tool call, route to the "tools" node
    if last_message.tool_calls:
        return "tools"
    
    # Otherwise, route to "final" or end
    return END

# Invoke Messages

def call_model(state: State, config: RunnableConfig):

    # Ensure state contains 'messages'
    if "messages" not in state:
        raise ValueError("State must contain a 'messages' key.")
    
    # Initialize messages from the state
    messages = state["messages"]
    
    # Check if a summary exists and prepend it as a system message if present
    summary = state.get("summary", "")
    if summary:
        system_message = f"Summary of conversation earlier: {summary}"
        messages = [SystemMessage(content=system_message)] + messages

    # Safely invoke the model
    try:
        response = model.invoke(messages, config)

    except Exception as e:
        raise RuntimeError(f"Error invoking the model: {e}")

    # Append the response to messages
    messages.append(response)

    # Return the updated state with messages
    return {"messages": messages[-1]}

# Build Graph

builder = StateGraph(State)

builder.add_node("agent", call_model)
builder.add_node("tools", tool_node)
builder.add_node("summarize", summarize_conversation)

builder.add_edge(START, "agent")
builder.add_edge("tools", "agent")
builder.add_edge("summarize", END)

builder.add_conditional_edges(
    "agent",
    select_next_node,
    {"summarize": "summarize", "tools": "tools", END: END},
)

graph = builder.compile(checkpointer=checkpointer)


#### Chainlit

@cl.password_auth_callback
def auth_callback(username: str, password: str):
    # Fetch the user matching username from your database
    # and compare the hashed password with the value stored in the database
    if (username, password) == ("Rida Naz", "user"):
        return cl.User(
            identifier="Rida Naz", metadata={"role": "user", "provider": "credentials"}
        )
    else:
        return None

@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="Discover symptoms and remedies",
            message="What are the common symptoms of seasonal allergies, and what remedies can help alleviate them?",
            icon="/public/search.png",
            ),
        cl.Starter(
            label="Learn about fitness for heart health",
            message="Recommend some beginner-friendly exercises that promote heart health. Explain their benefits in simple terms.",
            icon="/public/idea.png",
            ),

        cl.Starter(
             label="Common medications",
            message="Can you explain the purpose of common over-the-counter medications like ibuprofen or acetaminophen, and when to use them?",
            icon="/public/genai.png",
            ),
        cl.Starter(
            label="Plan a healthy diet for diabetics",
            message="Suggest a simple meal plan for someone managing Type 2 diabetes. Include tips on portion control and healthy food swaps.",
            icon="/public/pen.png",
            )
        ]

@cl.on_message
async def on_message(msg: cl.Message):
    config = {"configurable": {"thread_id": cl.context.session.id}}
    cb = cl.LangchainCallbackHandler()
    final_answer = cl.Message(content="")

    # Stream the conversation using the graph
    for response_msg, metadata in graph.stream(
        {"messages": [HumanMessage(content=msg.content)]},
        stream_mode="messages",
        config=RunnableConfig(callbacks=[cb], **config),
    ):
        # Check if the response is from the agent and not a ToolMessage
        if response_msg.content and isinstance(response_msg, AIMessage):
            # Stream only the agent's response
            await final_answer.stream_token(response_msg.content)

    # Send the final answer once streaming is complete
    await final_answer.send()
