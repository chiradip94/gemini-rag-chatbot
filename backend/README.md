# Building an Intelligent Chatbot with Langchain: A Step-by-Step Guide

In today’s digital age, creating intelligent chatbots is not just a trend but a necessity for many businesses. With advancements in AI and natural language processing, we can build sophisticated conversational agents that can understand and respond to user queries effectively. In this blog post, we’ll walk through the implementation of a chatbot using Python, LangChain, and WebSocket technology.

## What You Will Learn

1. Setting up a chatbot class with LangChain
2. Utilizing a retriever for context-based responses
3. Creating a WebSocket server to handle real-time communication
4. Putting everything together for an interactive experience

## Prerequisites

To follow along, ensure you have the following installed:

- Python 3.7 or later
- Required libraries (`asyncio`, `websockets`, `langchain`, `qdrant-client`, and others)
- An API key for Google’s Generative AI and Qdrant

You can install the necessary libraries using pip:

```bash
pip install asyncio websockets langchain qdrant-client python-dotenv
```

### Step 1: Setting Up the Chatbot Class

Let’s start by defining our chatbot class, `LlmChatBot`. This class will handle the interaction with the language model (LLM) and the document retriever.

```python
class LlmChatBot:
    def __init__(self) -> None:
        self.llm = self.get_llm()
        self.retriever = self.get_retriever(collection_name=os.getenv("COLLECTION_NAME", "chatbot"))
        self.rag_chain = self.setup_rag_chain()
        self.store = {}
```

In the `__init__` method, we initialize the LLM and the retriever, and set up a retrieval-augmented generation (RAG) chain. 

### Step 2: Configuring the Language Model

We utilize Google's Generative AI model for our chatbot. Here’s how we set it up:

```python
def get_llm(self):
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=GOOGLE_API_KEY)
    return llm
```

This function retrieves the LLM using the Google API key stored in your environment variables.

### Step 3: Document Retrieval

Next, we need a method to retrieve relevant documents for the conversation. We’ll use Qdrant as our vector store.

```python
def get_retriever(self, collection_name):
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    QDRANT_URL = os.getenv("QDRANT_URL")

    embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001", 
                                              task_type="retrieval_document",
                                              google_api_key=GOOGLE_API_KEY)
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    vectorstore = QdrantVectorStore(embedding=embedding, client=client, collection_name=collection_name)
    return vectorstore.as_retriever()
```

This function initializes a Qdrant client and sets up the retriever, which will be used to fetch contextually relevant information during the conversation.

### Step 4: Setting Up the RAG Chain

Now, let’s create the RAG chain to combine retrieval with generative responses:

```python
def setup_rag_chain(self):
    get_standalone_question_system_prompt = """Given a chat history and the latest user question ...
    standalone_question_prompt = ChatPromptTemplate.from_messages([...])

    history_aware_retriever = create_history_aware_retriever(self.llm, self.retriever, standalone_question_prompt)
    
    q_and_a_system_prompt = """You are an assistant for question-answering tasks. ...
    q_and_a_prompt = ChatPromptTemplate.from_messages([...])
    
    question_answer_chain = create_stuff_documents_chain(self.llm, q_and_a_prompt)
    
    return create_retrieval_chain(history_aware_retriever, question_answer_chain)
```

This method combines both retrieval and question-answering prompts to create a seamless interaction experience.

### Step 5: Implementing WebSocket for Real-time Interaction

With our chatbot set up, it’s time to create a WebSocket server to handle real-time communication with users.

```python
class WebSocketServer:
    def __init__(self, host, port, model):
        self.host = host
        self.port = port
        self.model = model
    
    async def handle_connection(self, websocket, path):
        async for message in websocket:
            data = json.loads(message)
            session_id = data.get("session_id", "default_session")
            input_text = data["input"]

            answer = self.model.get_answer(input_text, session_id)
            response = {"answer": answer}
            await websocket.send(json.dumps(response))
```

This class manages WebSocket connections and processes incoming messages, generating responses using our chatbot model.

### Step 6: Starting the WebSocket Server

Finally, we need to start our WebSocket server. The following code will do that:

```python
if __name__ == "__main__":
    model = LlmChatBot()
    server = WebSocketServer("localhost", int(os.getenv("SERVER_PORT", 8081)), model)
    asyncio.run(server.start_server())
```

This snippet initializes the chatbot and starts the server, listening for incoming connections.

## Conclusion

Congratulations! You've successfully built a real-time chatbot using Python, LangChain, and WebSockets. You can extend this bot further by integrating more advanced features, such as user authentication, custom commands, or even deploying it to the cloud.

Feel free to experiment and enhance your chatbot's capabilities. The world of AI-driven conversational agents is vast and full of opportunities for innovation. Happy coding!