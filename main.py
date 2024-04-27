# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# !pip install PyPDF2
# !pip install python-dotenv
# !pip install langchain
# !pip install langchain-groq
# !pip install chromadb
# !pip install discord

import os
import sys
import logging
import PyPDF2
import json
import logging
import discord
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOllama
from langchain_groq import ChatGroq
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from dotenv import load_dotenv
from tqdm import tqdm
from discord.ext import commands

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)  # Set the desired logging level

# Create a console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)  # Set the desired logging level for the console

# Create a formatter and add it to the console handler
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
console_handler.setFormatter(formatter)

# Add the console handler to the root logger
logger.addHandler(console_handler)

load_dotenv()

groq_api_key = os.environ['GROQ_API_KEY']
discord_token = os.environ['DISCORD_TOKEN']
bot_channel_id = int(os.environ['BOT_CHANNEL_ID'])
custom_system_prompt = os.environ['CUSTOM_SYSTEM_PROMPT']

chat_history = []

llm_groq = ChatGroq(
    groq_api_key=groq_api_key,
    model_name='llama3-8b-8192'
)

def process_file(file_path):
    """
    Process a file (PDF or txt) and return its text content.
    
    Args:
        file_path (str): The path to the file.
        
    Returns:
        str: The text content of the file, or None if the file format is unsupported.
    """
    logging.info(f"Processing file: {file_path}")
    if file_path.endswith(".pdf"):
        # Read the PDF file
        pdf = PyPDF2.PdfReader(file_path)
        pdf_text = ""
        for page in pdf.pages:
            pdf_text += page.extract_text()
    elif file_path.endswith(".txt"):
        # Read the txt file
        with open(file_path, "r") as f:
            pdf_text = f.read()
    else:
        logging.warning(f"Unsupported file format: {file_path}")
        return None
    logging.info(f"Finished processing file: {file_path}")
    return pdf_text

def create_embeddings(data_dir, persist_directory, embeddings):
    """
    Create embeddings for the files in the data directory and store them in a Chroma vector store.
    
    Args:
        data_dir (str): The directory containing the PDF or txt files.
        persist_directory (str): The directory to store the Chroma database.
        embeddings (OllamaEmbeddings): The embeddings to use for creating the vector store.
    """
    file_extensions = [".pdf", ".txt"]  # Supported file extensions
    files = []

    logging.info(f"Searching for files with extensions: {', '.join(file_extensions)}")
    for file_name in os.listdir(data_dir):
        file_path = os.path.join(data_dir, file_name)
        if os.path.isfile(file_path) and any(file_name.endswith(ext) for ext in file_extensions):
            files.append(file_path)

    if not files:
        logging.warning("No PDF or txt files found in the data directory.")
        return

    logging.info(f"Found {len(files)} files to process.")
    for file_index, file in enumerate(files, start=1):
        logging.info(f"Processing file {file_index}/{len(files)}: `{os.path.basename(file)}`...")
        pdf_text = process_file(file)

        if pdf_text is None:
            logging.warning(f"Skipping file {file_index}/{len(files)} due to unsupported format.")
            continue

        # Split the text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_text(pdf_text)
        logging.info(f"Text split into {len(texts)} chunks")

        # Create metadata for each chunk
        metadatas = [{"source": f"{i}-pl"} for i in range(len(texts))]

        # Create a Chroma vector store with a persistent directory
        logging.info(f"Creating Chroma vector store for file {file_index}/{len(files)}...")
        docsearch = Chroma.from_texts(texts, embeddings, metadatas=metadatas, persist_directory=persist_directory)
        logging.info(f"Embedded {len(texts)} chunks for file: `{os.path.basename(file)}`")

    logging.info(f"Created Chroma vector store in directory: {persist_directory}")

def create_retrieval_chain(persist_directory):
    """
    Create a conversational retrieval chain using the Chroma vector store.
    
    Args:
        persist_directory (str): The directory where the Chroma database is stored.
        
    Returns:
        ConversationalRetrievalChain: The created retrieval chain.
    """
    # Load the Chroma vector store from the persistent directory
    docsearch = Chroma(persist_directory=persist_directory, embedding_function=OllamaEmbeddings(model="nomic-embed-text"))

    system_prompt = SystemMessagePromptTemplate.from_template(custom_system_prompt)
    human_prompt = HumanMessagePromptTemplate.from_template("{context}\n\nHuman: {question}")
    prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])

    # Create a chain that uses the Chroma vector store
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm_groq,
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": prompt},
    )
    return chain

def split_long_response(response, max_length=1800):
    """
    Split a long response into chunks of maximum length.

    Args:
        response (str): The response text to split.
        max_length (int): The maximum length of each chunk (default: 1800).

    Returns:
        List[str]: A list of response chunks.
    """
    chunks = []
    while len(response) > max_length:
        chunk = response[:max_length]
        last_newline = max(chunk.rfind('\n'), chunk.rfind('.'))
        if last_newline > 0:
            chunk = chunk[:last_newline + 1]
        chunks.append(chunk.strip())
        response = response[len(chunk):]
    if response:
        chunks.append(response.strip())
    return chunks

def main():
    data_dir = "data"  # Directory containing the PDF or txt files
    persist_directory = "db"  # Directory to store the Chroma database
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    # Check if the Chroma vector store already exists
    if not os.path.exists(persist_directory):
        create_embeddings(data_dir, persist_directory, embeddings)
    else:
        logging.info("Chroma vector store already exists. Skipping embedding creation.")
    global chain
    chain = create_retrieval_chain(persist_directory)
    logging.info("Processing files done. You can now ask questions!")

    intents = discord.Intents.default()
    intents.messages = True
    intents.guilds = True
    intents.message_content = True

    custom_activity = discord.CustomActivity(name=os.environ["CUSTOM_DISCORD_STATUS"][:128])
    bot = commands.Bot(command_prefix='!', intents=intents, activity=custom_activity)

    # Generate the OAuth URL
    permissions = discord.Permissions(
        send_messages=True,
        read_messages=True,
        embed_links=True,
        attach_files=True,
        read_message_history=True,
        external_emojis=True,
        add_reactions=True
    )
    #oauth_url = discord.utils.oauth_url(bot.user.id, permissions=permissions)
    #logging.info(f"OAuth URL: {oauth_url}")
    
    @bot.event
    async def on_ready():
        logging.info(f'Logged in as {bot.user.name} (ID: {bot.user.id})')

    @bot.event
    async def on_error(event, *args, **kwargs):
        logging.error(f"Error in event {event}: {sys.exc_info()}")

    @bot.event
    async def on_message(message):
        global chat_history
    
        if message.author == bot.user:
            return
    
        if bot.user in message.mentions and message.channel.id == bot_channel_id:
            user_input = message.content.replace(f'<@!{bot.user.id}>', '').strip()
            logging.info(f"User input: {user_input}")
    
            chat_history.append(("Human", user_input))
    
            query = {"question": user_input, "chat_history": chat_history}
            logging.info(f"Query to embeddings:\n{json.dumps(query, indent=2)}")
    
            docs = chain.retriever.invoke(query)
            logging.info(f"Retrieved documents:\n{json.dumps([doc.page_content for doc in docs], indent=2)}")
    
            try:
                res = chain.invoke(query)
                answer = res["answer"]
                source_documents = res["source_documents"]
    
                logging.info(f"Model answer: {answer}")
    
                chat_history.append(("Assistant", answer))
    
                # Split long response into chunks
                response_chunks = split_long_response(answer)
    
                for chunk in response_chunks:
                    await message.reply(chunk)
    
                if source_documents:
                    source_names = [doc.metadata["source"] for doc in source_documents]
                    logging.info(f"Sources: {', '.join(source_names)}")
                else:
                    await message.channel.send("No sources found")
                    logging.info("No sources found")
            except Exception as e:
                logging.error(f"Error: {e}")
    
    try:
        logging.info("Starting the bot...")
        bot.run(discord_token)
    except Exception as e:
        logging.error(f"Error while running the bot: {str(e)}")

if __name__ == "__main__":
    main()
