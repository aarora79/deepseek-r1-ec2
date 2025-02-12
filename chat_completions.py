import os
import json
import logging
import argparse
import requests
from urllib.parse import urlparse
from typing import Dict, List, Optional, Union, Generator, Any
from pydantic import BaseModel, Field, HttpUrl, field_validator

# Configure logging with process ID, filename, line number, and millisecond precision
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d - PID:%(process)d - %(filename)s:%(lineno)d - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

DEFAULT_CONTENT = (
    "An ant walks along the x-axis by taking a sequence of steps of "
    "length 1. Some, all or none of these steps are in the positive "
    "x-direction; some, all or none of these steps are in the negative "
    "x-direction. The ant begins at x=0, takes a total of n steps, and "
    "ends at x=d. For each such sequence, let c be the number of times "
    "that the ant changes direction. Determine the number of different "
    "sequences of steps for which n=9 and d=5."
)

class Message(BaseModel):
    """Represents a chat message with role and content."""
    role: str = Field(..., description="Role of the message sender (e.g., 'user', 'assistant')")
    content: str = Field(..., description="Content of the message")

    @field_validator('role')
    @classmethod
    def validate_role(cls, v: str) -> str:
        """Validate that the role is one of the allowed values."""
        allowed_roles = {'user', 'assistant', 'system'}
        if v not in allowed_roles:
            raise ValueError(f"Role must be one of {allowed_roles}")
        return v

class ChatRequest(BaseModel):
    """Represents a chat completion request."""
    model: str = Field(..., description="Model identifier to use for completion")
    messages: List[Message] = Field(..., description="List of conversation messages")
    stream: bool = Field(default=True, description="Whether to stream the response")

class ChatResponse(BaseModel):
    """Represents a chat completion response."""
    id: Optional[str] = Field(None, description="Response identifier")
    choices: Optional[List[Dict]] = Field(None, description="List of completion choices")
    error: Optional[Dict] = Field(None, description="Error information if request failed")
    timings: Optional[Dict] = Field(None, description="Response time and token counts")

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Chat completion client')
    parser.add_argument(
        '--stream',
        type=str,
        choices=['yes', 'no'],
        default='yes',
        help='Enable or disable response streaming (default: yes)'
    )
    parser.add_argument(
        '--input',
        type=str,
        default=DEFAULT_CONTENT,
        help='Input content for the chat request'
    )
    return parser.parse_args()

def validate_endpoint_url(endpoint_url: str) -> bool:
    """
    Validate the endpoint URL format.
    
    Args:
        endpoint_url: The URL to validate
        
    Returns:
        bool: True if valid, False otherwise
    """
    try:
        result = urlparse(endpoint_url)
        return all([result.scheme, result.netloc])
    except Exception as e:
        logger.error(f"Invalid URL format: {e}")
        return False

def process_stream(response: requests.Response) -> Generator[str, None, None]:
    """
    Process streaming response and yield content chunks.
    
    Args:
        response: Streaming response from the server
        
    Yields:
        Decoded content from each chunk
    """
    for line in response.iter_lines():
        if line:
            try:
                json_str = line.decode('utf-8').removeprefix('data: ')
                if json_str.strip() == '[DONE]':
                    break
                chunk = json.loads(json_str)
                if content := chunk.get('choices', [{}])[0].get('delta', {}).get('content'):
                    yield content
            except json.JSONDecodeError as e:
                logger.error(f"Failed to decode JSON from chunk: {e}")
            except Exception as e:
                logger.error(f"Error processing stream chunk: {e}")

def send_chat_completion_request(
    endpoint_url: str,
    model: str,
    messages: List[Dict[str, str]],
    timeout: int = 300,
    stream: bool = True
) -> Union[ChatResponse, Generator[str, None, None]]:
    """
    Send a chat completion request to the specified endpoint.
    
    Args:
        endpoint_url: The URL of the chat completion endpoint
        model: The model identifier to use
        messages: List of conversation messages
        timeout: Request timeout in seconds
        stream: Whether to stream the response
        
    Returns:
        Either a ChatResponse object or a generator yielding streamed content
        
    Raises:
        ValueError: If the endpoint URL is invalid
        requests.exceptions.RequestException: If the request fails
    """
    if not validate_endpoint_url(endpoint_url):
        raise ValueError(f"Invalid endpoint URL: {endpoint_url}")
    
    try:
        # Create and validate request data using Pydantic
        request_data = ChatRequest(
            model=model,
            messages=[Message(**msg) for msg in messages],
            stream=stream
        )
        
        logger.info(f"Sending request to {endpoint_url} (streaming: {stream})")
        logger.debug(f"Request data: {request_data.model_dump_json()}")
        
        response = requests.post(
            endpoint_url,
            json=request_data.model_dump(),
            timeout=timeout,
            headers={"Content-Type": "application/json"},
            stream=stream
        )
        response.raise_for_status()
        
        if stream:
            return process_stream(response)
        else:
            # Parse response using Pydantic model
            print(response.json())
            chat_response = ChatResponse(**response.json())
            logger.info("Successfully received response")
            logger.debug(f"Response data: {chat_response.model_dump_json()}")
            return chat_response
        
    except requests.exceptions.Timeout:
        logger.error(f"Request timed out after {timeout} seconds")
        raise
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise

def main() -> None:
    """Main function to demonstrate usage."""
    args = parse_args()
    logger.info(f"Starting chat client (PID: {os.getpid()})")
    
    endpoint_url = "http://localhost:8080/v1/chat/completions"
    model = "deepseek-r1-variant"
    messages = [
        {
            "role": "user",
            "content": args.input
        }
    ]

    try:
        # Convert stream argument to boolean
        stream_enabled = args.stream.lower() == 'yes'
        
        # Get response (streaming or non-streaming)
        if stream_enabled:
            stream_generator = send_chat_completion_request(
                endpoint_url, model, messages, stream=True
            )
            print("Streaming response:")
            for content in stream_generator:
                print(content, end='', flush=True)
            print("\nStream completed")
        else:
            response = send_chat_completion_request(
                endpoint_url, model, messages, stream=False
            )
            print(json.dumps(response.model_dump(), indent=2))
        
        logger.info(f"Successfully completed {'streaming' if stream_enabled else 'non-streaming'} chat request")
    except Exception as e:
        logger.error(f"Failed to get chat completion: {e}")
        raise
    finally:
        logger.info("Chat client shutting down")

if __name__ == "__main__":
    main()