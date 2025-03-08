import logging

import dotenv

from interfaces.api import FlaskAgent

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    """
    Main entry point for the Heuman Agent Framework.
    Runs the Flask API agent.
    NOT FOR PRODUCTION
    """
    try:
        # Load environment variables
        dotenv.load_dotenv()

        # Initialize and run Flask agent
        logger.info("Starting Flask API agent...")
        flask_agent = FlaskAgent()
        flask_agent.run(host="0.0.0.0", port=5005)

    except KeyboardInterrupt:
        logger.info("Application stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
