import json
import re
from typing import Literal

from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field, ValidationError

ToolName = Literal[
    "search_documents", "list_tickets", "get_ticket", "create_ticket", "update_ticket"
]
ChatRunnable = Runnable[LanguageModelInput, BaseMessage]

# Models sometimes wrap JSON in a markdown fence even when told not to.
JSON_CODE_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)


class Plan(BaseModel):
    steps: list[str]
    needs_tools: list[ToolName]


class Answer(BaseModel):
    text: str
    citations: list[str]
    confidence: float = Field(ge=0, le=1)


class Review(BaseModel):
    supported: bool
    reason: str


ProfileFactKey = Literal["preferred_language", "team"]


class ProfileFact(BaseModel):
    key: ProfileFactKey
    value: str


class ProfileFacts(BaseModel):
    facts: list[ProfileFact]


class InjectionVerdict(BaseModel):
    is_injection: bool
    reason: str


class FlaggedChunk(BaseModel):
    index: int
    reason: str


class ChunkScreening(BaseModel):
    flagged: list[FlaggedChunk]


class InvalidModelOutputError(ValueError):
    pass


def parse_model_output[OutputModel: BaseModel](
    reply: str, output_model: type[OutputModel]
) -> OutputModel:
    stripped_reply = reply.strip()
    fenced = JSON_CODE_FENCE.match(stripped_reply)
    json_text = fenced.group(1) if fenced else stripped_reply
    try:
        return output_model.model_validate_json(json_text)
    except ValidationError as error:
        raise InvalidModelOutputError(
            f"LLM reply is not a valid {output_model.__name__}: {error}"
        ) from error


def json_reply_instructions(output_model: type[BaseModel]) -> str:
    json_schema = json.dumps(output_model.model_json_schema())
    return f"Reply with only a JSON object matching this JSON schema:\n{json_schema}"


async def parse_with_one_retry[OutputModel: BaseModel](
    llm: ChatRunnable,
    messages: list[BaseMessage],
    reply: BaseMessage,
    output_model: type[OutputModel],
) -> OutputModel:
    """Parses the reply; if it is invalid, shows the model the error once and parses the retry.

    A second invalid reply raises InvalidModelOutputError so the caller can fail closed.
    """
    try:
        return parse_model_output(reply.text, output_model)
    except InvalidModelOutputError as error:
        correction = HumanMessage(f"{error}\nReply again with only the corrected JSON object.")
        retry_reply = await llm.ainvoke([*messages, reply, correction])
        if isinstance(retry_reply, AIMessage) and retry_reply.tool_calls:
            raise InvalidModelOutputError(
                f"Expected a {output_model.__name__} but the corrected reply called a tool"
            ) from error
        return parse_model_output(retry_reply.text, output_model)


async def ask_for_json[OutputModel: BaseModel](
    llm: ChatRunnable, instructions: str, prompt: str, output_model: type[OutputModel]
) -> OutputModel:
    messages: list[BaseMessage] = [
        SystemMessage(f"{instructions}\n\n{json_reply_instructions(output_model)}"),
        HumanMessage(prompt),
    ]
    reply = await llm.ainvoke(messages)
    return await parse_with_one_retry(llm, messages, reply, output_model)
