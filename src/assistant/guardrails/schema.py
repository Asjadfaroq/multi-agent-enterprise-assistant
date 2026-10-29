import json
import re
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ValidationError

ToolName = Literal[
    "search_documents", "list_tickets", "get_ticket", "create_ticket", "update_ticket"
]

# Models sometimes wrap JSON in a markdown fence even when told not to.
JSON_CODE_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)


class Plan(BaseModel):
    steps: list[str]
    needs_tools: list[ToolName]


class Draft(BaseModel):
    text: str
    citations: list[str]


class Review(BaseModel):
    supported: bool
    reason: str


def parse_model_output[OutputModel: BaseModel](
    reply: str, output_model: type[OutputModel]
) -> OutputModel:
    stripped_reply = reply.strip()
    fenced = JSON_CODE_FENCE.match(stripped_reply)
    json_text = fenced.group(1) if fenced else stripped_reply
    try:
        return output_model.model_validate_json(json_text)
    except ValidationError as error:
        raise ValueError(f"LLM reply is not a valid {output_model.__name__}: {error}") from error


def json_reply_instructions(output_model: type[BaseModel]) -> str:
    json_schema = json.dumps(output_model.model_json_schema())
    return f"Reply with only a JSON object matching this JSON schema:\n{json_schema}"


async def ask_for_json[OutputModel: BaseModel](
    llm: BaseChatModel, instructions: str, prompt: str, output_model: type[OutputModel]
) -> OutputModel:
    system_prompt = f"{instructions}\n\n{json_reply_instructions(output_model)}"
    reply = await llm.ainvoke([SystemMessage(system_prompt), HumanMessage(prompt)])
    return parse_model_output(reply.text, output_model)
