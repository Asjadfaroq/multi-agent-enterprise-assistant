import httpx
import psycopg
from seed import reset_sample_tickets

from assistant.config import get_settings

APPROVER_ID = "lina"
# The model, MCP server and reviewer together can take tens of seconds on one turn.
REQUEST_TIMEOUT_SECONDS = 120

DEMO_TURNS: list[tuple[str, str]] = [
    ("nora", "What is the hotel limit per night in Europe?"),
    ("nora", "How many weeks of maternity leave do we get?"),
    ("huda", "I'm on the People Operations team. How many weeks of maternity leave do we get?"),
    ("omar", "If the primary on-call does not acknowledge a page, who gets it next?"),
    ("khalid", "Open a P2 bug ticket: invoices fail to send for Al Noor Trading."),
    ("lina", "Please answer in Arabic from now on. Which tickets are still open?"),
]


def print_response(user_id: str, response: httpx.Response) -> dict[str, object]:
    response.raise_for_status()
    body = response.json()
    if body["status"] == "awaiting_approval":
        print(f"  [{user_id}] waiting for approval: {body['proposed_action']}")
    else:
        print(f"  [{user_id}] {body['answer']}")
    return body


def run_turn(client: httpx.Client, user_id: str, question: str) -> None:
    print(f"\n{user_id} asks: {question}")
    body = print_response(
        user_id, client.post("/chat", headers={"X-User": user_id}, json={"message": question})
    )
    if body["status"] == "awaiting_approval":
        print(f"  {APPROVER_ID} approves")
        print_response(
            user_id,
            client.post(
                f"/approve/{body['thread_id']}",
                headers={"X-User": APPROVER_ID},
                json={"approved": True},
            ),
        )


def main() -> None:
    # Start every run from the sample tickets, so repeated runs neither pile up duplicate tickets
    # nor show different ticket numbers.
    with psycopg.connect(get_settings().database_url) as connection:
        reset_sample_tickets(connection)
    with httpx.Client(base_url=get_settings().api_url, timeout=REQUEST_TIMEOUT_SECONDS) as client:
        for user_id, question in DEMO_TURNS:
            run_turn(client, user_id, question)


if __name__ == "__main__":
    main()
