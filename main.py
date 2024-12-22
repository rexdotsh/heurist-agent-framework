import logging
import dotenv
import threading
import asyncio
from interfaces.api import FlaskAgent
from interfaces.telegram import TelegramAgent
from agents.core_agent import CoreAgent

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_flask(flask_agent):
    """Runs the (blocking) Flask API agent in a separate thread."""
    try:
        logger.info("Starting Flask API agent...")
        flask_agent.run(host='0.0.0.0', port=5005)
    except Exception as e:
        logger.error(f"Flask API agent error: {str(e)}")

def run_telegram(telegram_agent):
    """Run the Telegram agent"""
    try:
        logger.info("Starting Telegram agent...")
        telegram_agent.run()
    except Exception as e:
        logger.error(f"Telegram agent error: {str(e)}")


# async def main_async():
#     """Async main function to handle all async operations"""
#     # Create shared core agent and interfaces
#     core_agent = CoreAgent()
#     flask_agent = FlaskAgent(core_agent)
#     telegram_agent = TelegramAgent(core_agent)

#     # Start Flask in a separate thread
#     flask_thread = threading.Thread(
#         target=run_flask,
#         args=(flask_agent,),
#         daemon=True
#     )
#     flask_thread.start()
#     run_telegram(telegram_agent)

#     # Wait for Flask thread
#     flask_thread.join()

def main():
    """Main entry point"""
    try:
        # Load environment variables
        dotenv.load_dotenv()
            # Create shared core agent and interfaces
        core_agent = CoreAgent()
        flask_agent = FlaskAgent(core_agent)
        telegram_agent = TelegramAgent(core_agent)

        # Start Flask in a separate thread
        flask_thread = threading.Thread(
            target=run_flask,
            args=(flask_agent,),
            daemon=True
        )
        flask_thread.start()
        run_telegram(telegram_agent)

        # Wait for Flask thread
        flask_thread.join()
        # Run the async main function
        #asyncio.run(main_async())
    except KeyboardInterrupt:
        logger.info("Application stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        raise

if __name__ == "__main__":
    main()
