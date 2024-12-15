import os
import chainlit as cl
from dotenv import load_dotenv
from typing import Literal, Dict, Union, Optional
from psycopg_pool import ConnectionPool
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import ToolNode
from langchain.schema.runnable.config import RunnableConfig
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage, RemoveMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import MessagesState
from pydantic import BaseModel
# NOTE: you must use langchain-core >= 0.3 with Pydantic v2

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
NEON_DB_URI = os.getenv("NEON_DB_URI")

os.environ["TAVILY_API_KEY"]= os.getenv("TAVILY_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "chatbot"

# Postgres DB

# Connection pool for efficient database access
connection_kwargs = {"autocommit": True, "prepare_threshold": 0}

# Create a persistent connection pool
pool = ConnectionPool(conninfo=NEON_DB_URI, max_size=50, kwargs=connection_kwargs)

# Initialize PostgresSaver checkpointer
checkpointer = PostgresSaver(pool)
checkpointer.setup()  # Ensure database tables are set up

# State

class State(MessagesState):
    summary: str

# Tavily Search Tool

search_tool = TavilySearchResults(max_results=2, search_depth="advanced")
tools = [search_tool]

tool_node = ToolNode(tools=[search_tool])

# Model

model = ChatGoogleGenerativeAI(
    model="gemini-pro",
    api_key=GEMINI_API_KEY,
    temperature=0.2
)

model = model.bind_tools(tools)

# Summarization

def summarize_conversation(state: State) -> State:
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

    # Save the updated summary in the state
    state["summary"] = response.content

    # Delete all but the 2 most recent messages
    delete_messages = [RemoveMessage(id=getattr(m, "id", None)) for m in state["messages"][:-2]]


# Conditional Function

def select_next_node(state: State) -> Union[Literal["tools", "summarize_conversation"], str]:
    messages = state["messages"]
    last_message = messages[-1]

    # If there are more than six messages, route to "summarize_conversation"
    if len(messages) > 6:
        return "summarize_conversation"
    
    # If the LLM makes a tool call, route to the "tools" node
    if last_message.tool_calls:
        return "tools"
    
    # Otherwise, route to "final" or end
    return END

# Invoke Messages

def call_model(state: State) -> Dict[str, object]:
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
        response = model.invoke(messages)
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

# add summary node
builder.add_node(summarize_conversation)

builder.add_edge(START, "agent")
builder.add_edge("tools", "agent")
builder.add_edge("summarize_conversation", END)

builder.add_conditional_edges(
    "agent",
    select_next_node,
    {"summarize_conversation": "summarize_conversation", "tools": "tools", END: END},
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
            label="Explore AI tools for productivity",
            message="Recommend some AI-powered tools that can help improve productivity, and explain how they work in simple terms.",
            icon="/public/search.png",
            ),
        cl.Starter(
            label="Plan a tech meetup discussion",
            message="Suggest some engaging discussion topics for a tech meetup. I want them to be beginner-friendly but still interesting for tech enthusiasts.",
            icon="/public/idea.png",
            ),

        cl.Starter(
            label="Basics of generative AI",
            message="Explain the basics of generative AI and its real-world applications. Keep it concise and easy to understand.",
            icon="/public/genai.png",
            ),
        cl.Starter(
            label="New automation script for workflows",
            message="Brainstorm an automation script idea to simplify repetitive tasks in my daily workflow. Focus on creativity and practicality.",
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
