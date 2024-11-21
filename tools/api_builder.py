from typing import Dict, Any, List
import json
import requests
from functools import lru_cache

@lru_cache(maxsize=1)
def get_discovery_doc(api_name: str, version: str) -> Dict[str, Any]:
    """Fetch and parse a Google API discovery document."""
    url = f"https://www.googleapis.com/discovery/v1/apis/{api_name}/{version}/rest"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

def generate_function_definition(method_data: Dict[str, Any], path: str) -> Dict[str, Any]:
    """Generate a function definition from method data."""
    parameters = {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False
    }
    
    # Add path parameters
    for param_name, param_data in method_data.get("parameters", {}).items():
        parameters["properties"][param_name] = {
            "type": param_data.get("type", "string"),
            "description": param_data.get("description", "")
        }
        if param_data.get("required", False):
            parameters["required"].append(param_name)
    
    # Add request body if present
    if "request" in method_data:
        schema = method_data["request"].get("$ref", {})
        # Would need to resolve schema reference from definitions
    
    return {
        "type": "function",
        "function": {
            "name": method_data["id"],
            "description": method_data.get("description", ""),
            "parameters": parameters,
            "strict": True
        }
    }

def generate_api_tools(api_name: str, version: str) -> List[Dict[str, Any]]:
    """Generate tool definitions from an API discovery document."""
    doc = get_discovery_doc(api_name, version)
    tools = []
    
    for path, resource in doc.get("resources", {}).items():
        for method_name, method_data in resource.get("methods", {}).items():
            tools.append(generate_function_definition(method_data, path))
    
    return tools 