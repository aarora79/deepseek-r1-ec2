import os
import json
import logging
import argparse
import requests
from urllib.parse import urlparse
from typing import Dict, List, Optional, Union, Generator, Any
from pydantic import BaseModel, Field, HttpUrl, field_validator, Extra

# Configure logging with process ID, filename, line number, and millisecond precision
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d - PID:%(process)d - %(filename)s:%(lineno)d - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

DEFAULT_PROMPT: str = "<｜User｜>What is quantum gravity?<｜Assistant｜>"

class CompletionRequest(BaseModel):
    """Pydantic model for request validation with field validators and class method."""

    endpoint_url: HttpUrl = Field(..., description="The API endpoint URL.")
    model: str = Field(..., description="The AI model name.")
    prompt: str = Field(..., description="The input prompt for AI completion.")

    @field_validator("model")
    @classmethod
    def validate_model(cls, value: str) -> str:
        """Ensure the model name is non-empty string."""
        if not value:
            raise ValueError("Model name must be a non-empty string.")
        return value

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, value: str) -> str:
        """Ensure the prompt is not empty."""
        if not value.strip():
            raise ValueError("Prompt cannot be empty.")
        return value

    @classmethod
    def from_args(cls, args) -> "CompletionRequest":
        """
        Creates a CompletionRequest instance from parsed command-line arguments.

        Args:
            args (Namespace): Parsed command-line arguments.

        Returns:
            CompletionRequest: A validated instance.
        """
        return cls(
            endpoint_url=args.endpoint_url,
            model=args.model,
            prompt=args.prompt,
        )

class CompletionResponse(BaseModel):
    """Pydantic model for parsing API response."""
    
    index: int
    content: str    
    generation_settings: Dict
    prompt: str
    timings: Dict
    # Allowing additional fields dynamically without explicitly defining them
    model_config = {
        "extra": "allow"
    }

def send_completion_request(request_data: CompletionRequest):
    """
    Sends a completion request to the specified API endpoint.
    
    Args:
        request_data (CompletionRequest): A validated Pydantic model containing request details.
    
    Returns:
        dict: The JSON response from the API.
    """
    data = {"model": request_data.model, "prompt": request_data.prompt}

    try:
        response = requests.post(str(request_data.endpoint_url), json=data, timeout=300)
        response.raise_for_status()
        response_json = response.json()
        return CompletionResponse(**response_json)

    except requests.exceptions.RequestException as e:
        logger.error(f"Error making request: {e}")
        return {"error": str(e)}

def main() -> None:
    logger.info(f"Starting chat client (PID: {os.getpid()})")
    """Main function to parse command-line arguments and send a completion request."""
    parser = argparse.ArgumentParser(description="Send a completion request to an AI model endpoint.")
    parser.add_argument(
        "--endpoint_url",
        type=str,
        default="http://localhost:8080/completion",
        help="The API endpoint URL (default: http://localhost:8080/completion)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek-r1-variant",
        help="The AI model name (default: deepseek-r1-variant)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=DEFAULT_PROMPT,
        help=f"The input prompt (default: {DEFAULT_PROMPT})",
    )

    args = parser.parse_args()

    try:
        request_data = CompletionRequest(
            endpoint_url=args.endpoint_url,
            model=args.model,
            prompt=args.prompt,
        )
        response = send_completion_request(request_data)
        if response:
            logger.info(response.model_dump_json(indent=2))  # Pretty-print using Pydantic's built-in JSON method
            logger.info(f"content=\n{response.content}")
            tags = ["<think>", "</think>"]
            for tag in tags:
                if tag in response.content:
                    logger.info(f"{tag} found in content")
                else:
                    logger.error(f"{tag} not in content")
        else:
            logger.error("Failed to get a valid response from the API.")
    
    except Exception as e:
        logger.error(f"Error: {e}")

if __name__ == "__main__":
    main()
