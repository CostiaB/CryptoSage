from datetime import datetime
from typing import Annotated
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig
from typing_extensions import TypedDict
from langgraph.graph.message import AnyMessage, add_messages
import os
from langchain_openai import ChatOpenAI
from tools import get_coin_list, match_coin, get_historical_data, calculate_rsi, get_trade_signal, \
    ohlc_values, calculate_fibonacci_levels, find_support_resistance, MACD_Alligator_advice
from utilities import _print_event
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import tools_condition
from utilities import create_tool_node_with_fallback
import shutil
import uuid
import pandas_ta as ta
from time import  sleep

with open('openai', 'r') as file:
    OPENAI_API_KEY = file.readline().strip()


os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if OPENAI_API_KEY is None:
    raise ValueError("OpenAI API key is not installed.")


class State(TypedDict):
    """

    """
    messages: Annotated[list[AnyMessage], add_messages]



class Assistant:
    """
    Assistant
    """
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: State, config: RunnableConfig):
        i = 0
        while i < 5:
            configuration = config.get("configurable", {})
            passenger_id = configuration.get("passenger_id", None)
            state = {**state, "user_info": passenger_id}
            result = self.runnable.invoke(state)
            # If the LLM happens to return an empty response, we will re-prompt it
            # for an actual response.
            if not result.tool_calls and (
                not result.content
                or isinstance(result.content, list)
                and not result.content[0].get("text")
            ):
                messages = state["messages"] + [("user", "Respond with a real output.")]
                state = {**state, "messages": messages}
            else:
                break
            i += 1
        return {"messages": result}


# Haiku is faster and cheaper, but less accurate
# llm = ChatAnthropic(model="claude-3-haiku-20240307")
#llm = ChatAnthropic(model="claude-3-sonnet-20240229", temperature=1)
# You could swap LLMs, though you will likely want to update the prompts when
# doing so!

llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model="gpt-3.5-turbo"
)

primary_assistant_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            " You are a knowledgeable support assistant for cryptocurrencies. "
            " You should check the coin name. It should be accepted by coingecko API"
            " Use the provided tools to obtain price characteristics,"
            "  it can be indicators values, and other information to assist the user's queries. "
            " When searching, be persistent. Expand your query bounds if the first search returns no results. "
            " If a search comes up empty, expand your search before giving up."
            "\n\nCurrent user:\n\n{user_info}\n"
            "\nCurrent time: {time}.",
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now())

part_1_tools = [
    get_historical_data,
    calculate_rsi,
    get_trade_signal,
    #ohlc_values,
    calculate_fibonacci_levels,
    find_support_resistance,
    MACD_Alligator_advice

]
part_1_assistant_runnable = primary_assistant_prompt | llm.bind_tools(part_1_tools)



builder = StateGraph(State)


# Define nodes: these do the work
builder.add_node("assistant", Assistant(part_1_assistant_runnable))
builder.add_node("tools", create_tool_node_with_fallback(part_1_tools))
# Define edges: these determine how the control flow moves
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    tools_condition,
)
builder.add_edge("tools", "assistant")

# The checkpointer lets the graph persist its state
# this is a complete memory for the entire graph.
memory = MemorySaver()
part_1_graph = builder.compile(checkpointer=memory)


"""from IPython.display import Image, display

try:
    display(Image(part_1_graph.get_graph(xray=True).draw_mermaid_png()))
except Exception:
    # This requires some extra dependencies and is optional
    pass"""



# Let's create an example conversation a user might have with the assistant
tutorial_questions = [
    "What is the price for LTC",
    "Can you tell me about BTC? its resistance level?",
    "Calculate Fibonacci levels for ETH and give trading advises",
    "What do you know about ARPA and what data do you have about it?",
    "What is the price for Perplexity Protocol",
    "What coins can you recommend for trading?",
    "How to trade APT?",
    "What can you tell about SOLANA? Any traiding advises?"
]

# Update with the backup file so we can restart from the original place in each section
thread_id = str(uuid.uuid4())

config = {
    "configurable": {
        "thread_id": thread_id,
    }
}


_printed = set()
for question in tutorial_questions:
    sleep(60)
    events = part_1_graph.stream(
        {"messages": ("user", question)}, config, stream_mode="values"
    )
    for event in events:
        _print_event(event, _printed)
