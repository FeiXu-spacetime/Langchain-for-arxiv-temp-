import os
import sys

import openai
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma

import API
# Sets the OpenAI API key as an environment variable for authentication purposes.
# This is necessary to make API requests to OpenAI services.
os.environ["OPENAI_API_KEY"] = API.APIKEY

# A flag to determine whether the data should be persisted (saved) on disk. 
# If True, the program will save the model's state to disk, allowing for faster subsequent queries
# by reusing the saved state instead of processing data from scratch.
PERSIST = False

# Initialize the query variable. If the script is run with command line arguments, 
# it takes the first argument as the initial query to process.
query = None
if len(sys.argv) > 1:
  query = sys.argv[1]

# If persistence is enabled and a saved state exists in the 'persist' directory,
# it reuses the existing index to avoid rebuilding it, which can save time.
# Otherwise, it proceeds to load data and create a new index.
if PERSIST and os.path.exists("persist"):
  print("Reusing index...\n")
  vectorstore = Chroma(persist_directory="persist", embedding_function=OpenAIEmbeddings())
  index = VectorStoreIndexWrapper(vectorstore=vectorstore)
else:
  # Depending on the requirement, loads data from a single text file or a directory of files.
  # Here, it's set to load from a single file, but an alternative method is commented out for directory loading.
  loader = TextLoader("./data/arxiv-cleaned-metadata.csv")  # Use this line if you only need data.txt
  #loader = DirectoryLoader("dataset/")
  if PERSIST:
    # If persistence is enabled, it creates a new index and specifies a directory to save it.
    # This allows for the index to be reused in future runs.
    index = VectorstoreIndexCreator(vectorstore_kwargs={"persist_directory":"persist"}).from_loaders([loader])
  else:
    # Creates a new index without saving it to disk.
    index = VectorstoreIndexCreator().from_loaders([loader])

# Initializes a conversational retrieval chain with an OpenAI GPT model and the created index.
# This setup enables efficient query processing and response generation based on the indexed data.
chain = ConversationalRetrievalChain.from_llm(
  llm=ChatOpenAI(model="gpt-3.5-turbo"),
  retriever=index.vectorstore.as_retriever(search_kwargs={"k": 1}),
)

# Initializes an empty list to keep track of the chat history. 
# This is useful for maintaining context in a conversation.
chat_history = []

# Main loop to handle input queries, process them, and generate responses.
while True:
  # If no query is provided (e.g., not passed as a command line argument), prompts the user for input.
  if not query:
    query = input("Prompt: ")
  # Allows the user to exit the loop and terminate the script with specific commands.
  if query in ['quit', 'q', 'exit']:
    sys.exit()
  # Processes the query using the conversational retrieval chain and prints the generated response.
  result = chain({"question": query, "chat_history": chat_history})
  print(result['answer'])

  # Updates the chat history with the current query and response. This ensures the context is
  # maintained for generating coherent responses in an ongoing conversation.
  chat_history.append((query, result['answer']))
  # Resets the query variable, allowing for new user inputs in the next iteration of the loop.
  query = None
