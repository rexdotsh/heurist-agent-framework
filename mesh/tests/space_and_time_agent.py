import asyncio
import logging
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

# Adjust the path to access the parent directory
sys.path.append(str(Path(__file__).parent.parent.parent))
from mesh.space_and_time_agent import SpaceAndTimeAgent  # noqa: E402

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()


async def run_agent():
    logger.info("Initializing SpaceAndTimeAgent")
    agent = SpaceAndTimeAgent()

    try:
        # First, test authentication explicitly
        auth_success = await agent._authenticate()
        logger.info(f"Authentication test result: {auth_success}")
        if not auth_success:
            logger.error("Authentication failed, check your SPACE_AND_TIME_API_KEY environment variable")
            return

        # Test with a natural language query
        logger.info("Testing natural language query about Ethereum blocks")
        ethereum_query = {"query": "Get the number of blocks created on Ethereum per day over the last month"}
        ethereum_result = await agent.handle_message(ethereum_query)
        print(f"Ethereum Query Result: {ethereum_result}")

        # Test with a direct tool call for schema information
        logger.info("Testing direct tool call to list schemas")
        schema_input = {
            "tool": "list_schemas",
            "tool_arguments": {},
        }
        schema_result = await agent.handle_message(schema_input)
        print(f"Schema List Result: {schema_result}")

        # Test direct access to a specific schema's tables
        logger.info("Testing listing tables for HEURIST schema")
        tables_input = {
            "tool": "list_tables",
            "tool_arguments": {"schema": "HEURIST"},
        }
        tables_result = await agent.handle_message(tables_input)
        print(f"Tables List Result: {tables_result}")

        # Test column information for a specific table
        logger.info("Testing listing columns for a specific table")
        columns_input = {
            "tool": "list_table_columns",
            "tool_arguments": {"schema": "HEURIST", "table": "S1_GPU_PERFORMANCE"},
        }
        columns_result = await agent.handle_message(columns_input)
        print(f"Columns List Result: {columns_result}")

        # Test with raw data only
        logger.info("Testing raw data query about NFT collections")
        raw_input = {
            "query": "What are the top 10 NFT collections by trading volume on Ethereum?",
            "raw_data_only": True,
        }
        raw_result = await agent.handle_message(raw_input)
        print(f"Raw Data Result: {raw_result}")

        # Test with a more complex query targeting a specific schema
        logger.info("Testing complex query with specific schema")
        specific_query = {
            "tool": "query_space_and_time",
            "tool_arguments": {
                "query": "What's the daily average gas price on Ethereum for the past week?",
                "schema": "ETHEREUM",
            },
        }
        specific_result = await agent.handle_message(specific_query)
        print(f"Specific Schema Query Result: {specific_result}")

        # Save the test results to a YAML file
        script_dir = Path(__file__).parent
        current_file = Path(__file__).stem
        base_filename = f"{current_file}_example"
        output_file = script_dir / f"{base_filename}.yaml"

        yaml_content = {
            "ethereum_example": {"input": ethereum_query, "output": ethereum_result},
            "schema_example": {"input": schema_input, "output": schema_result},
            "tables_example": {"input": tables_input, "output": tables_result},
            "columns_example": {"input": columns_input, "output": columns_result},
            "raw_example": {"input": raw_input, "output": raw_result},
            "specific_example": {"input": specific_query, "output": specific_result},
        }

        with open(output_file, "w", encoding="utf-8") as f:
            yaml.dump(yaml_content, f, allow_unicode=True, sort_keys=False)

        logger.info(f"Results saved to {output_file}")

    except Exception as e:
        logger.error(f"Error during agent testing: {str(e)}")
    finally:
        logger.info("Cleaning up agent resources")
        await agent.cleanup()


if __name__ == "__main__":
    asyncio.run(run_agent())
