import os
import sys
import time
from openai import OpenAI
from dotenv import load_dotenv
from typing import Optional, Dict, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import local modules
from tools import handle_tool_calls, get_tool_definitions
from terminalstyle import (
    print_assistant_response,
    print_system_message,
    print_code,
    clear_screen,
    print_welcome_message,
    print_divider,
    get_user_input,
    print_tool_usage,
)
from prompts import SUPER_ASSISTANT_INSTRUCTIONS

# Constants
THREAD_ID_FILE = "thread_id.txt"
MODEL_NAME = "gpt-4o-mini"  # Using the latest model

class AssistantManager:
    def __init__(self):
        """Initialize the AssistantManager with OpenAI client and configuration."""
        # Load environment variables with override to ensure we get the .env values
        load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env'), override=True)
        
        # Initialize OpenAI client
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
            
        self.client = OpenAI(api_key=self.api_key)
        self.assistant_id = os.getenv("ASSISTANT_ID")
        if not self.assistant_id:
            raise ValueError("ASSISTANT_ID not found in environment variables")
        
        # Get the existing assistant and update its configuration
        try:
            self.assistant = self.client.beta.assistants.retrieve(self.assistant_id)
            self.update_assistant_configuration()
        except Exception as e:
            raise ValueError(f"Could not retrieve assistant with ID {self.assistant_id}: {str(e)}")
        
        self.thread_id = self.load_thread_id()

    def load_thread_id(self) -> Optional[str]:
        """Load thread ID from file if it exists."""
        if os.path.exists(THREAD_ID_FILE):
            with open(THREAD_ID_FILE, "r") as file:
                return file.read().strip()
        return None

    def save_thread_id(self, thread_id: str) -> None:
        """Save thread ID to file."""
        with open(THREAD_ID_FILE, "w") as file:
            file.write(thread_id)

    def create_assistant(self) -> Dict[str, Any]:
        """Create a new assistant with specified configuration."""
        return self.client.beta.assistants.create(
            name="Super Assistant",
            instructions=SUPER_ASSISTANT_INSTRUCTIONS,
            model=MODEL_NAME,
            tools=get_tool_definitions()
        )

    def cancel_active_runs(self) -> None:
        """Cancel any active runs on the current thread."""
        if not self.thread_id:
            return
            
        try:
            runs = self.client.beta.threads.runs.list(thread_id=self.thread_id)
            for run in runs.data:
                if run.status in ["in_progress", "queued", "requires_action"]:
                    try:
                        self.client.beta.threads.runs.cancel(
                            thread_id=self.thread_id,
                            run_id=run.id
                        )
                    except Exception:
                        pass
        except Exception:
            pass

    def wait_for_completion(self, run_id: str) -> Optional[Any]:
        """Wait for a run to complete and handle any required actions."""
        while True:
            run = self.client.beta.threads.runs.retrieve(
                thread_id=self.thread_id, 
                run_id=run_id
            )
            
            if run.status == "requires_action":
                try:
                    tool_outputs = handle_tool_calls(run)
                    for tool_call in run.required_action.submit_tool_outputs.tool_calls:
                        print_tool_usage(tool_call.function.name)
                    run = self.client.beta.threads.runs.submit_tool_outputs(
                        thread_id=self.thread_id,
                        run_id=run_id,
                        tool_outputs=tool_outputs
                    )
                except Exception as e:
                    print_system_message(f"Error handling tool calls: {str(e)}")
                    return None
                    
            elif run.status == "completed":
                return run
                
            elif run.status in ["failed", "cancelled", "expired"]:
                print_system_message(f"Run ended with status: {run.status}")
                return None
                
            time.sleep(0.5)

    def reset_thread(self) -> None:
        """Reset the conversation thread."""
        self.cancel_active_runs()
        self.thread_id = None
        if os.path.exists(THREAD_ID_FILE):
            os.remove(THREAD_ID_FILE)
        print_system_message("Thread reset. Starting a new conversation.")

    def process_user_input(self, user_input: str) -> bool:
        """Process user input and return whether to continue the conversation."""
        if user_input.lower() == "reset":
            self.reset_thread()
            return True
            
        if user_input.lower() == "quit":
            self.cancel_active_runs()
            print_system_message("Thank you for chatting! Goodbye.")
            return False

        try:
            # Create or ensure thread exists
            if self.thread_id is None:
                thread = self.client.beta.threads.create()
                self.thread_id = thread.id
                self.save_thread_id(self.thread_id)
            else:
                self.cancel_active_runs()

            # Create message and run
            self.client.beta.threads.messages.create(
                thread_id=self.thread_id,
                role="user",
                content=user_input
            )

            run = self.client.beta.threads.runs.create(
                thread_id=self.thread_id,
                assistant_id=self.assistant.id
            )

            # Wait for completion and process response
            if completed_run := self.wait_for_completion(run.id):
                messages = self.client.beta.threads.messages.list(thread_id=self.thread_id)
                for message in messages.data:
                    if message.role == "assistant":
                        print_assistant_response(message.content[0].text.value)
                        break

        except Exception as e:
            print_system_message(f"An error occurred: {str(e)}")
            print_system_message("Starting a new conversation...")
            self.reset_thread()

        return True

    def run(self) -> None:
        """Main conversation loop."""
        try:
            clear_screen()
            print_welcome_message()
            print_divider()

            while True:
                user_input = get_user_input()
                print_divider()
                
                if not self.process_user_input(user_input):
                    break
                    
                print_divider()

        except Exception as e:
            print_system_message(f"Fatal error: {str(e)}")
            sys.exit(1)

    def update_assistant_configuration(self) -> None:
        """Update the assistant with current tools and instructions."""
        try:
            print_system_message("Updating assistant configuration...")
            self.assistant = self.client.beta.assistants.update(
                assistant_id=self.assistant_id,
                instructions=SUPER_ASSISTANT_INSTRUCTIONS,
                tools=get_tool_definitions(),
                model=MODEL_NAME
            )
            print_system_message("Assistant configuration updated successfully!")
        except Exception as e:
            print_system_message(f"Warning: Failed to update assistant configuration: {str(e)}")

def main():
    """Entry point of the application."""
    try:
        assistant_manager = AssistantManager()
        assistant_manager.run()
    except KeyboardInterrupt:
        print_system_message("\nGoodbye!")
    except Exception as e:
        print_system_message(f"Fatal error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
