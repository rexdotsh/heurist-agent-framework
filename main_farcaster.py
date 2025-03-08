import logging

import dotenv

from interfaces.farcaster_post import FarcasterAgent

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    """
    Main entry point for the Heuman Agent Framework.
    Runs the Farcaster agent for automated casting.
    """
    try:
        # Load environment variables
        dotenv.load_dotenv()

        # Initialize and run Farcaster agent
        logger.info("Starting Farcaster agent...")
        agent = FarcasterAgent()
        agent.run()

    except KeyboardInterrupt:
        logger.info("Application stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
