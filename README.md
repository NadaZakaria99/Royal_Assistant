# Royal Assistant: Ancient Egyptian History Chatbot
The Royal Assistant is an interactive chatbot designed to provide educational information about ancient Egyptian kings and queens. It uses advanced natural language processing (NLP) techniques to generate responses in the style of historical royal figures. The chatbot is capable of detecting whether the query is about a male or female figure and responds with an appropriate voice (male or female) using text-to-speech (TTS) technology.

This project is ideal for educational purposes, museums, or anyone interested in ancient Egyptian history.

# Features
- **Historical Context:** Provides detailed responses about ancient Egyptian kings and queens.
- **Gender-Specific Voices:** Detects whether the query is about a male or female figure and responds with the appropriate voice.
- **Interactive Web Interface:** A user-friendly web interface for interacting with the chatbot.
- **Text-to-Speech:** Converts the chatbot's responses into audio for an immersive experience.
- **PDF Integration:** Loads and processes PDF documents containing historical information.

# Technologies Used
- **LangChain:** For building the conversational AI and retrieval-based question-answering system.
- **Hugging Face Embeddings:** For generating embeddings from the text.
- **FAISS:** For efficient similarity search and retrieval of documents.
- **Flask:** For building the web application.
- **pyttsx3:** For text-to-speech conversion.
- **NVIDIA API:** For leveraging the Llama 3.3 model for generating responses.

# Installation
- **Clone the Repository:**
   git clone https://github.com/your-username/royal-assistant.git
cd royal-assistant 
- **Install Dependencies:**
  pip install -r requirements.txt
- **Set Up API Key:**
  Replace the API_KEY in config.py with your NVIDIA API key.
- **Prepare PDF Documents:**
  Place your PDF documents containing historical information in the folder specified by PDF_FOLDER_PATH in config.py.
- **Run the Application:**
  python app.py
- **Access the Web Interface**
