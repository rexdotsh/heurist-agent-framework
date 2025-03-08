import asyncio
import logging

from interfaces.farcaster_reply import FarcasterReplyAgent

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


async def main():
    """
    Main entry point for the Heuman Agent Framework.
    Runs the Farcaster agent for automated casting.
    """
    try:
        # Initialize and run Farcaster agent
        logger.info("Starting Farcaster agent...")
        agent = FarcasterReplyAgent()
        await agent.start()
    except KeyboardInterrupt:
        logger.info("Application stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
