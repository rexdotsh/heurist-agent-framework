import asyncio

import dotenv

from interfaces.twitter_reply import TwitterReplyAgent


async def main():
    dotenv.load_dotenv()

    # Initialize agent
    agent = TwitterReplyAgent()

    try:
        print("Starting Twitter Reply Agent...")
        # Start monitoring in background thread
        agent.start_monitoring()
        print("Monitoring thread started")

        print("Starting workers... Press Ctrl+C to exit")
        await agent.run_workers(num_workers=2)

    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
