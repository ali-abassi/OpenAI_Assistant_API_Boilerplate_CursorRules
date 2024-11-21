import os
import json
from functools import lru_cache

AGENT_DIRECTORY = "agent_directory"
THREAD_FILE = "current_thread.json"

# Cache the directory check
@lru_cache(maxsize=1)
def get_agent_directory():
    """Get or create the agent directory."""
    if not os.path.exists(AGENT_DIRECTORY):
        os.makedirs(AGENT_DIRECTORY)
    return AGENT_DIRECTORY

def get_full_path(file_path):
    """Get the full path for a file."""
    return os.path.join(get_agent_directory(), file_path)

def read_file(file_path):
    """Read a file from the agent directory."""
    try:
        with open(get_full_path(file_path), 'r') as file:
            return file.read()
    except Exception as e:
        return f"Error reading file: {str(e)}"

def write_file(file_path, content=""):
    """Write content to a file in the agent directory."""
    try:
        with open(get_full_path(file_path), 'w') as file:
            file.write(content)
        return f"Successfully wrote to {file_path}"
    except Exception as e:
        return f"Error writing file: {str(e)}"

def list_files():
    """List all files in the agent directory."""
    try:
        files = os.listdir(get_agent_directory())
        return json.dumps(files)
    except Exception as e:
        return f"Error listing files: {str(e)}"

def read_thread_id():
    """Read the current thread ID from storage."""
    try:
        with open(get_full_path(THREAD_FILE), 'r') as file:
            data = json.load(file)
            return data.get('thread_id')
    except (FileNotFoundError, json.JSONDecodeError):
        return None

def save_thread_id(thread_id):
    """Save the current thread ID to storage."""
    try:
        with open(get_full_path(THREAD_FILE), 'w') as file:
            json.dump({'thread_id': thread_id}, file)
        return True
    except Exception as e:
        print(f"Error saving thread ID: {str(e)}")
        return False

def clear_thread_id():
    """Clear the stored thread ID."""
    try:
        if os.path.exists(get_full_path(THREAD_FILE)):
            os.remove(get_full_path(THREAD_FILE))
        return True
    except Exception as e:
        print(f"Error clearing thread ID: {str(e)}")
        return False