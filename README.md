# Discord Bot for RAG with Groq, Chroma using Llama 3

This Discord bot leverages advanced natural language processing and retrieval techniques to answer questions by searching through a database of indexed documents. It's built using a combination of Python libraries including `PyPDF2`, `python-dotenv`, `discord.py`, and several components from the `langchain` ecosystem.

## Features

- **Document Processing:** Converts PDF and text documents into searchable formats.
- **Advanced Query Handling:** Uses the `langchain` library to perform conversational queries and retrieve contextually relevant information.
- **Real-time Interaction:** Integrated within Discord to provide immediate responses to user queries.


Create a .env file in the root directory and update it with your Discord and GROQ API keys:

- DISCORD_TOKEN=your_discord_bot_token
- GROQ_API_KEY=your_groq_api_key
- BOT_CHANNEL_ID=your_discord_channel_id
- CUSTOM_SYSTEM_PROMPT=your_custom_prompt

Create a data directory, and put your files (text and/or PDF) in it.

To start the bot, run the following command:

python main.py

