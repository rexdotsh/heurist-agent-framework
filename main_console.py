import asyncio
from agents.core_agent import CoreAgent

async def main():
    # Initialize the core agent
    agent = CoreAgent()
    
    print("Welcome to the Heurist Agent Console!")
    print("Type 'exit' to quit")
    print("-" * 50)

    while True:
        # Get user input
        user_message = input("\nYou: ").strip()
        
        if user_message.lower() == 'exit':
            print("\nGoodbye!")
            break
        
        try:
            # Process the message using the core agent
            response = await agent.handle_message(
                message=user_message,
                source_interface="console",
                skip_embedding=True  # Skip embedding for simple console interaction
            )
            
            # Print the response
            print("\nAgent:", response)
            
        except Exception as e:
            print(f"\nError: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())
