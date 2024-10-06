import asyncio
import os
import json
import websockets
import logging
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

logging.basicConfig(level = logging.INFO)

load_dotenv()

# tweet_prompt = PromptTemplate.from_template("You are a content creator. Write me a tweet about {topic}.")
# tweet_chain = LLMChain(llm=llm, prompt=tweet_prompt, verbose=True)

# resp = tweet_chain.run("Sibling rivalary")
# print(resp)

class LlmChatBot:
    def __init__(self) -> None:
        self.llm = self.get_llm()
        self.retriever = self.get_retriever(collection_name=os.getenv("COLLECTION_NAME","chatbot"))
        self.rag_chain = self.setup_rag_chain()
        self.store = {}

    def get_llm(self):
        GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", google_api_key=GOOGLE_API_KEY)
        return llm

    def get_retriever(self,collection_name):
        GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
        QDRANT_API_KEY= os.getenv("QDRANT_API_KEY")
        QDRANT_URL= os.getenv("QDRANT_URL")

        embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001", 
                                                  task_type="retrieval_document",
                                                  google_api_key=GOOGLE_API_KEY, 
                                                )
        client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
        )
        vectorstore = QdrantVectorStore(
            embedding=embedding, 
            client=client,
            collection_name=collection_name
        )
        return vectorstore.as_retriever()

    def setup_rag_chain(self):
        get_standalone_question_system_prompt = """Given a chat history and the latest user question \
        which might reference context in the chat history, formulate a standalone question \
        which can be understood without the chat history. Do NOT answer the question, \
        just reformulate it if needed and otherwise return it as is."""

        standalone_question_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", get_standalone_question_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        history_aware_retriever = create_history_aware_retriever(
            self.llm, self.retriever, standalone_question_prompt
        )

        q_and_a_system_prompt = """You are an assistant for question-answering tasks. \
        Use the following pieces of retrieved context to answer the question. \
        If you don't know the answer, just say that you don't know. Use \
        three sentences maximum and keep the answer concise.\n\n{context}"""

        q_and_a_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", q_and_a_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        question_answer_chain = create_stuff_documents_chain(self.llm, q_and_a_prompt)

        return create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    def get_session_history(self, session_id):
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
        return self.store[session_id]
    
    def get_answer(self, input_text, session_id):
        conversational_rag_chain = RunnableWithMessageHistory(
            self.rag_chain,
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )
        
        return conversational_rag_chain.invoke(
            {"input": input_text},
            config={"configurable": {"session_id": session_id}}
        )["answer"]


class WebSocketServer:
    def __init__(self, host, port, model):
        self.host = host
        self.port = port
        self.model = model
    
    async def handle_connection(self, websocket, path):
            """
            Handles a WebSocket connection by receiving messages from the client,
            processing them using the AI model, and sending back the response.

            Parameters:
            - websocket: The WebSocket connection object.
            - path: The path of the WebSocket connection.

            Returns:
            None
            """
            async for message in websocket:
                try:
                    data = json.loads(message)
                    session_id = data.get("session_id", "default_session")
                    input_text = data["input"]

                    answer = self.model.get_answer(input_text, session_id)
                    
                    response = {"answer": answer}
                    await websocket.send(json.dumps(response))
                except Exception as e:
                    logging.error(f"An error occurred: {e}")
                    error_response = {"error": str(e)}
                    await websocket.send(json.dumps(error_response))
    
    async def start_server(self):
            """
            Starts the server and listens for incoming connections.

            This method uses the `websockets.serve` function to create a WebSocket server
            and binds it to the specified `host` and `port`. It then waits for incoming
            connections and handles each connection using the `handle_connection` method.

            Note: This method runs indefinitely until the program is terminated.

            Parameters:
                self (object): The instance of the class.

            Returns:
                None
            """
            async with websockets.serve(self.handle_connection, self.host, self.port):
                await asyncio.Future()  # run forever


if __name__ == "__main__":
    model = LlmChatBot()
    server = WebSocketServer("localhost", int(os.getenv("SERVER_PORT",8081)), model)
    asyncio.run(server.start_server())