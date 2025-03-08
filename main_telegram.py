import logging

import dotenv

from agents.core_agent import CoreAgent
from interfaces.telegram import TelegramAgent

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_telegram(telegram_agent):
    """Run the Telegram agent"""
    try:
        logger.info("Starting Telegram agent...")
        telegram_agent.run()
    except Exception as e:
        logger.error(f"Telegram agent error: {str(e)}")


def main():
    """
    Main entry point for the Heuman Agent Framework.
    Demonstrates both shared and standalone usage.
    """
    try:
        # Load environment variables
        dotenv.load_dotenv()

        # Example 1: Standalone agent (default)
        # telegram_agent = TelegramAgent()  # Uses its own CoreAgent instance

        # Example 2: Shared core agent (commented out)
        core_agent = CoreAgent()
        telegram_agent = TelegramAgent(core_agent)  # Uses shared core_agent

        # Run the agent
        run_telegram(telegram_agent)

    except KeyboardInterrupt:
        logger.info("Application stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
