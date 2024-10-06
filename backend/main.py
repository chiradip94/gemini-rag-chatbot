import asyncio
import os
import logging


choice = int(input("Type 1 for gemini LLM and 2 for Ollama: "))

if choice == 1:
    from .llm import LlmChatBot, WebSocketServer
    model = LlmChatBot()
    server = WebSocketServer("localhost", int(os.getenv("SERVER_PORT",8081)), model)
    asyncio.run(server.start_server())

elif choice == 2:
    from local_llm.llm import LlmChatBotLocal, WebSocketServer
    model = LlmChatBotLocal()
    server = WebSocketServer("localhost", int(os.getenv("SERVER_PORT",8081)), model)
    asyncio.run(server.start_server())
    