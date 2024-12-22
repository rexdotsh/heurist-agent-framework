from flask import Flask, request, jsonify, send_file
import logging
import os
from pathlib import Path
from agents.core_agent import CoreAgent
import dotenv
from functools import wraps

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

dotenv.load_dotenv()

def require_api_key(f):
    @wraps(f)
    async def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        if not api_key or api_key != os.getenv('API_KEY'):
            return jsonify({'error': 'Invalid or missing API key'}), 401
        return await f(*args, **kwargs)
    return decorated_function

class FlaskAgent(CoreAgent):
    def __init__(self, core_agent=None):
        if core_agent:
            # If core_agent is provided, initialize parent then proxy to provided core_agent
            super().__init__()
            self._proxy_to(core_agent)
        else:
            # Standalone mode - just initialize normally
            super().__init__()
            
        self.app = Flask(__name__)
        self._setup_routes()
        
        # Register this interface with the core agent (either self or proxied)
        self.register_interface('api', self)

    def _setup_routes(self):
        # Example usage:
        # curl -X POST http://localhost:5000/message \
        #   -H "Content-Type: application/json" \
        #   -d '{"message": "Tell me about artificial intelligence"}'
        #
        # Response:
        # {
        #   "text": "AI is a field of computer science...", 
        #   "image_url": "http://example.com/image.jpg"  # Optional
        # }
        @self.app.route('/message', methods=['POST'])
        @require_api_key
        async def handle_message():
            try:
                data = request.get_json()
                if not data or 'message' not in data:
                    return jsonify({'error': 'No message provided'}), 400

                text_response, image_url = await self.handle_message(
                    data['message'],
                    source_interface='api'
                )
                
                if self.is_shared:
                    logger.info("Operating in shared mode with core agent")
                else:
                    logger.info("Operating in standalone mode")
                
                response = {}
                if image_url:
                    response['image_url'] = image_url
                if text_response:
                    response['text'] = text_response
                
                return jsonify(response)
            except Exception as e:
                logger.error(f"Message handling failed: {str(e)}")
                return jsonify({'error': 'Internal server error'}), 500

    def run(self, host='0.0.0.0', port=5005):
        """Start the Flask server"""
        logger.info(f"Starting Flask API on port {port}...")
        self.app.run(host=host, port=port)

def main():
    agent = FlaskAgent()
    agent.run()

if __name__ == "__main__":
    try:
        logger.info("Starting Flask agent...")
        main()
    except KeyboardInterrupt:
        logger.info("\nFlask agent stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")