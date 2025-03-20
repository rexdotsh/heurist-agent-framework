import json
import time
from typing import Any

from dotenv import load_dotenv
from heurist_mesh_client.client import MeshClient


def print_json(data: Any) -> None:
    if hasattr(data, "model_dump"):
        data = data.model_dump()
    print(json.dumps(data, indent=2))


def wait_for_task(client: MeshClient, task_id: str) -> None:
    """Helper to wait for task completion and print progress."""
    while True:
        result = client.query_task(task_id=task_id)
        if result.status == "finished":
            print("\nTask completed:")
            if result.result:
                print_json(result.result)
            break
        print("Task in progress...")
        if result.reasoning_steps:
            for step in result.reasoning_steps:
                print(f"Step: {step.content}")
        time.sleep(1)


def main():
    load_dotenv()
    client = MeshClient()

    # example 1: async task with natural language query
    print("\nExample 1: Natural Language Query (Async)")
    print("-" * 50)
    task = client.create_task(
        agent_id="CoinGeckoTokenInfoAgent",
        query="What is the current price of Bitcoin and its market cap?",
    )
    print(f"Task created with ID: {task.task_id}")
    wait_for_task(client, task.task_id)

    # example 2: sync request with tool and raw data
    print("\nExample 2: Tool with Raw Data (Sync)")
    print("-" * 50)
    response = client.sync_request(
        agent_id="CoinGeckoTokenInfoAgent",
        tool="get_token_info",
        tool_arguments={"coingecko_id": "ethereum"},
        raw_data_only=True,
    )
    print_json(response)

    # example 3: async task with tool but natural language response
    print("\nExample 3: Tool with Natural Language Response (Async)")
    print("-" * 50)
    task = client.create_task(
        agent_id="CoinGeckoTokenInfoAgent",
        tool="get_token_info",
        tool_arguments={"coingecko_id": "solana"},
        raw_data_only=False,
    )
    print(f"Task created with ID: {task.task_id}")
    wait_for_task(client, task.task_id)

    # example 4: sync request with natural language query
    print("\nExample 4: Natural Language Query (Sync)")
    print("-" * 50)
    response = client.sync_request(
        agent_id="CoinGeckoTokenInfoAgent",
        query="What is the current price of Solana and its market cap?",
        raw_data_only=False,
    )
    print_json(response)


if __name__ == "__main__":
    main()
