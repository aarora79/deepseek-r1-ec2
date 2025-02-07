import os
import chainlit as cl
import asyncio

from openai import AsyncClient

client = AsyncClient(base_url="http://localhost:8080/", api_key="lm-studio")

settings = {
    "temperature": 0.5,
    "max_tokens": 2048,
}


@cl.on_chat_start
async def start_chat():
    cl.user_session.set(
        "message_history",
        [
            {
                "role": "user",
                "content": "You are a helpful assistant, Answer the questions the user might have",
            }
        ],
    )

async def answer_as(name):
    message_history = cl.user_session.get("message_history")
    msg = cl.Message(author=name, content="")

    stream = await client.chat.completions.create(
        model=model_name,
        messages=message_history,
        stream=True,
        **settings,
    )
    async for part in stream:
        if token := part.choices[0].delta.content or "":
            await msg.stream_token(token)

    # Need to add the information that it was the author who answered but OpenAI only allows assistant.
    # simplified for the purpose of the demo.
    message_history.append({"role": "assistant", "content": msg.content})
    await msg.send()


@cl.on_message
async def main(message: cl.Message):
    message_history = cl.user_session.get("message_history")
    message_history.append({"role": "user", "content": message.content})

    await asyncio.gather( answer_as("assistant"))