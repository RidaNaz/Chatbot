import os
from dotenv import load_dotenv
from typing import Literal, Dict, Union, Optional
from psycopg_pool import ConnectionPool
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import ToolNode
from langchain.schema.runnable.config import RunnableConfig
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage, RemoveMessage, ToolCall
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import MessagesState
from pydantic import BaseModel
# NOTE: you must use langchain-core >= 0.3 with Pydantic v2
import chainlit as cl

load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
NEON_DB_URI = os.getenv("NEON_DB_URI")

os.environ["TAVILY_API_KEY"]= os.getenv("TAVILY_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "chatbot"

# Postgres DB

# Connection pool for efficient database access
connection_kwargs = {"autocommit": True, "prepare_threshold": 0}

# Create a persistent connection pool
pool = ConnectionPool(conninfo=NEON_DB_URI, max_size=20, kwargs=connection_kwargs)

# Initialize PostgresSaver checkpointer
checkpointer = PostgresSaver(pool)
checkpointer.setup()  # Ensure database tables are set up

# State

class State(MessagesState):
    summary: str

# Tavily Search Tool

search_tool = TavilySearchResults(max_results=2)
tools = [search_tool]

tool_node = ToolNode(tools=[search_tool])

# Human Node

class RequestAssistance(BaseModel):
    """Escalate the conversation to an expert. Use this if you are unable to assist directly or if the user requires support beyond your permissions.

    To use this function, relay the user's 'request' so the expert can provide the right guidance.
    """

    request: str

# Helper Function
def create_response(response: str, ai_message: AIMessage):
    if not ai_message.tool_calls:
        raise ValueError("No tool calls found in the AI message.")
    return ToolMessage(
        content=response,
        tool_call_id=ai_message.tool_calls[0]["id"],
    )


def human_node(state: State):
    if not state["messages"]:
        raise ValueError("State 'messages' is empty.")
    new_messages = []
    if not isinstance(state["messages"][-1], ToolMessage):
        # Append a placeholder ToolMessage if needed
        new_messages.append(
            create_response("No response from human.", state["messages"][-1])
        )
    return {
        "messages": new_messages,
        "ask_human": False,
    }

# Model

model = ChatGoogleGenerativeAI(
    model="gemini-pro",
    api_key=gemini_api_key,
    max_retries=2,
    temperature=0.2
)

model = model.bind_tools(tools+[RequestAssistance])

# Summarization

def summarize_conversation(state: State, message_count_threshold: int = 6) -> Dict[str, object]:
    """
    Summarizes the conversation if the number of messages exceeds the threshold.
    """
    # Get the number of messages in the state
    message_count = len(state.get("messages", []))

    # If the number of messages has reached the threshold, create or update the summary
    if message_count >= message_count_threshold:
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

        return {
            "summarize_conversation": response.content,
            "messages": delete_messages
        }
    
    # If the message count hasn't reached the threshold, return the state as is
    return {}


# Conditional Function

def select_next_node(state: State) -> Union[Literal["tools", "human", "summarize_conversation"], str]:
    messages = state["messages"]
    last_message = messages[-1]

    # If there are more than six messages, route to "summarize_conversation"
    if len(messages) > 6:
        return "summarize_conversation"
    
    # If the LLM makes a tool call, route to the "tools" node
    if last_message.tool_calls:
        return "tools"
    
    # If a human input is required, route to the "human" node
    if state.get("ask_human", False):
        return "human"
    
    # Otherwise, route to "final" or end
    return END

# Invoke Messages

def call_model(state: State, config: RunnableConfig) -> Dict[str, object]:
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

    # Check for tool_calls and determine if human assistance is required
    ask_human = (
        hasattr(response, "tool_calls") 
        and response.tool_calls
        and response.tool_calls[0].get("name") == RequestAssistance.__name__
    )

    # Append the response to messages
    messages.append(response)

    # Return the updated state with messages and ask_human flag
    return {"messages": messages[-1], "ask_human": ask_human}

# Build Graph

builder = StateGraph(State)

builder.add_node("agent", call_model)
builder.add_node("tools", tool_node)

# add human node
builder.add_node("human", human_node)

# add summary node
builder.add_node(summarize_conversation)

builder.add_edge(START, "agent")
builder.add_edge("human", "agent")
builder.add_edge("tools", "agent")
builder.add_edge("summarize_conversation", END)

builder.add_conditional_edges(
    "agent",
    select_next_node,
    {"human": "human", "summarize_conversation": "summarize_conversation", "tools": "tools", END: END},
)

graph = builder.compile(checkpointer=checkpointer, interrupt_before=["human"])


####

@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="Text inviting friend to wedding",
            message="Write a text asking a friend to be my plus-one at a wedding next month. I want to keep it super short and casual, and offer an out.",
            icon="/public/favicon.png",
            ),
        cl.Starter(
            label="Morning routine ideation",
            message="Can you help me create a personalized morning routine that would help increase my productivity throughout the day? Start by asking me about my current habits and what activities energize me in the morning.",
            icon="/public/idea.png",
            ),

        cl.Starter(
            label="Explain superconductors",
            message="Explain superconductors like I'm five years old.",
            icon="/public/search.png",
            ),
        cl.Starter(
            label="Python script for daily email reports",
            message="Write a script to automate sending daily email reports in Python, and walk me through how I would set it up.",
            icon="/public/pen.png",
            )
        ]

@cl.on_message
async def on_message(msg: cl.Message):
    config = {"configurable": {"thread_id": cl.context.session.id}}
    cb = cl.LangchainCallbackHandler()
    final_answer = cl.Message(content="")

    # Stream the conversation using the graph
    for msg, metadata in graph.stream(
        {"messages": [HumanMessage(content=msg.content)]}, 
        stream_mode="messages", 
        config=RunnableConfig(callbacks=[cb], **config)
    ):
        # Check if the message has content and the node isn't a HumanMessage
        if msg.content and not isinstance(msg, HumanMessage):
            # If it reaches the end of the stream or the last node in the graph, send the message
            if metadata["langgraph_node"] in ["agent", "tools", "human", "summarize_conversation"]:
                await final_answer.stream_token(msg.content)

    # Send the final answer once streaming is complete
    await final_answer.send()