from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.tools.types import BaseTool
from llama_index.core.workflow import (
    Context,
    Workflow,
    StartEvent,
    StopEvent,
    step,
)
from openai import OpenAI
from llama_index.core.llms import ChatMessage
from llama_index.core.tools import ToolSelection, ToolOutput
from llama_index.core.workflow import Event
from typing import Any, List, Callable
from ag_ui.core import EventType, StateDeltaEvent
import uuid
import asyncio
from dotenv import load_dotenv
import os
from prompts import system_prompt

load_dotenv()


class InputEvent(Event):
    input: list[ChatMessage]


class StreamEvent(Event):
    delta: str


class ToolCallEvent(Event):
    tool_calls: list[ToolSelection]


class FunctionOutputEvent(Event):
    output: ToolOutput


class FuncationCallingAgent(Workflow):
    def __init__(
        self,
        *args: Any,
        llm: FunctionCallingLLM | None = None,
        tools: List[BaseTool] | None = None,
        messages: List[Any],
        emit_event: Callable,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.tools = tools or []
        modified_messages = messages.copy()
        modified_messages[0].content = system_prompt.replace(
            "{PORTFOLIO_DATA_PLACEHOLDER}", "[]"
        )
        self.messages = modified_messages or []
        self.emit_event = emit_event
        self.llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    @step
    async def prepare_chat_history(self, ctx: Context, ev: StartEvent) -> InputEvent:
        # clear sources
        self.emit_event(
            StateDeltaEvent(
                type=EventType.STATE_DELTA,
                delta=[
                    {
                        "op": "add",
                        "path": "/tool_logs/-",
                        "value": {
                            "message": "Gathering stock data",
                            "status": "processing",
                            "id": str(uuid.uuid4()),
                        },
                    }
                ],
            )
        )
        await asyncio.sleep(0)
        return InputEvent(input=[])

    @step
    async def handle_llm_input(
        self, ctx: Context, ev: InputEvent
    ) -> ToolCallEvent | StopEvent:
        chat_history = ev.input

        # stream the response
        self.messages
        response_stream = self.llm.chat.completions.create(
            model="gpt-4o-mini",
            messages=self.messages,
            tools=[extract_relevant_data_from_user_prompt],
        )
        # response_stream = await self.llm.astream_chat_with_tools(
        #     self.tools, chat_history=chat_history
        # )
        # async for response in response_stream:
        #     ctx.write_event_to_stream(StreamEvent(delta=response.delta or ""))

        # save the final response, which should have all content
        # memory = await ctx.store.get("memory")
        # memory.put(response.message)
        # await ctx.store.set("memory", memory)

        # get tool calls
        tool_calls = response_stream.tool_calls

        if not tool_calls:
            sources = await ctx.store.get("sources", default=[])
            return StopEvent(
                result={"response": response_stream, "sources": [*sources]}
            )
        else:
            return ToolCallEvent(tool_calls=tool_calls)

    @step
    async def handle_tool_calls(self, ctx: Context, ev: ToolCallEvent) -> InputEvent:
        tool_calls = ev.tool_calls
        tools_by_name = {tool.metadata.get_name(): tool for tool in self.tools}

        tool_msgs = []
        sources = await ctx.store.get("sources", default=[])

        # call tools -- safely!
        for tool_call in tool_calls:
            tool = tools_by_name.get(tool_call.tool_name)
            additional_kwargs = {
                "tool_call_id": tool_call.tool_id,
                "name": tool.metadata.get_name(),
            }
            if not tool:
                tool_msgs.append(
                    ChatMessage(
                        role="tool",
                        content=f"Tool {tool_call.tool_name} does not exist",
                        additional_kwargs=additional_kwargs,
                    )
                )
                continue

            try:
                tool_output = tool(**tool_call.tool_kwargs)
                sources.append(tool_output)
                tool_msgs.append(
                    ChatMessage(
                        role="tool",
                        content=tool_output.content,
                        additional_kwargs=additional_kwargs,
                    )
                )
            except Exception as e:
                tool_msgs.append(
                    ChatMessage(
                        role="tool",
                        content=f"Encountered error in tool call: {e}",
                        additional_kwargs=additional_kwargs,
                    )
                )

        # update memory
        memory = await ctx.store.get("memory")
        for msg in tool_msgs:
            memory.put(msg)

        await ctx.store.set("sources", sources)
        await ctx.store.set("memory", memory)

        chat_history = memory.get()
        return InputEvent(input=chat_history)


extract_relevant_data_from_user_prompt = {
    "type": "function",
    "function": {
        "name": "extract_relevant_data_from_user_prompt",
        "description": "Gets the data like ticker symbols, amount of dollars to be invested, interval of investment.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker_symbols": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "description": "A stock ticker symbol, e.g. 'AAPL', 'GOOGL'.",
                    },
                    "description": "A list of stock ticker symbols, e.g. ['AAPL', 'GOOGL'].",
                },
                "investment_date": {
                    "type": "string",
                    "description": "The date of investment, e.g. '2023-01-01'.",
                },
                "amount_of_dollars_to_be_invested": {
                    "type": "array",
                    "items": {
                        "type": "number",
                        "description": "The amount of dollars to be invested, e.g. 10000.",
                    },
                    "description": "The amount of dollars to be invested, e.g. [10000, 20000, 30000].",
                },
                "interval_of_investment": {
                    "type": "string",
                    "description": "The interval of investment, e.g. '1d', '5d', '1mo', '3mo', '6mo', '1y'. If the user did not specify the interval, then assume it as 'single_shot'.",
                },
                "to_be_added_in_portfolio": {
                    "type": "boolean",
                    "description": "True if the user wants to add it to the current portfolio, false for the sandbox portfolio.",
                },
            },
            "required": [
                "ticker_symbols",
                "investment_date",
                "amount_of_dollars_to_be_invested",
                "to_be_added_in_portfolio",
            ],
        },
    },
}
