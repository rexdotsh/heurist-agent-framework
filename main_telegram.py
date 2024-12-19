import logging
from interfaces.telegram import TelegramAgent
import dotenv

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """
    Main entry point for the Heuman Agent Framework.
    Currently runs the Telegram agent.
    """
    try:
        # Load environment variables
        dotenv.load_dotenv()
        
        # Initialize and run Telegram agent
        logger.info("Starting Telegram agent...")
        telegram_agent = TelegramAgent()
        telegram_agent.run()
        
    except KeyboardInterrupt:
        logger.info("Application stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        raise

if __name__ == "__main__":
    main()
