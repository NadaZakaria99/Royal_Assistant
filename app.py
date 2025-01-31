# import os
# from langchain.llms import OpenAI    
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.vectorstores import FAISS
# from langchain.chains import RetrievalQA
# from langchain.document_loaders import PyPDFLoader, DirectoryLoader
# from langchain.text_splitter import CharacterTextSplitter
# from langchain.memory import ConversationBufferMemory
# from config import API_KEY, PDF_FOLDER_PATH

# from langchain.llms.base import LLM
# from typing import Any, List, Mapping, Optional
# from openai import OpenAI as OpenAIClient

# from flask import Flask, request, render_template_string, jsonify, send_file
# import pyttsx3
# import threading
# import uuid
# import re

# class LlamaLLM(LLM):
#     client: Any

#     def __init__(self):
#         super().__init__()
#         self.client = OpenAIClient(
#             base_url="https://integrate.api.nvidia.com/v1",
#             api_key=API_KEY 
#         )

#     def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
#         completion = self.client.chat.completions.create(
#             model="meta/llama-3.1-405b-instruct",
#             messages=[{"role": "user", "content": prompt}],
#             temperature=0.2,
#             top_p=0.7,
#             max_tokens=1024,
#         )
#         return completion.choices[0].message.content

#     @property
#     def _llm_type(self) -> str:
#         return "Llama 3.1"

# # Initialize the custom LLM
# llm = LlamaLLM()

# # Initialize the conversation memory
# memory = ConversationBufferMemory()

# # Load and process the PDF
# loader = DirectoryLoader(PDF_FOLDER_PATH, glob="**/*.pdf", loader_cls=PyPDFLoader)
# documents = loader.load()

# # Split the documents into chunks
# text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
# texts = text_splitter.split_documents(documents)

# # Initialize the embedding model
# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# # Create a vector store
# vectorstore = FAISS.from_documents(texts, embeddings)

# # Create a retrieval-based QA chain
# qa_chain = RetrievalQA.from_chain_type(
#     llm=llm,
#     chain_type="stuff",
#     retriever=vectorstore.as_retriever(),
#     memory=memory,
# )

# # Initialize pyttsx3
# engine = pyttsx3.init()

# def get_response(query: str) -> str:
#     # Retrieve the relevant information from the document
#     raw_response = qa_chain.invoke(query)

#     royal_prompt = f"""
#     Based on the following query, identify which king or queen is being addressed.
#     Then, respond to the query as if you were that royal person with politeness and authority.
#     Use a first-person perspective and be realistic, talk as a charismatic monarch.
#     Only use information from the provided context. Do not invent or add information not present in the source material.
#     If you don't have specific information about the query, respond in a way that's consistent with a royal personality, perhaps stating that such information is not within your royal knowledge or is kept private.

#     Respond in the language of the query. If the query is in Arabic, respond in Arabic. If the query is in English, respond in English. If the query is a mix of both, respond in a mix of both.

#     Context: {raw_response}

#     Previous conversation:
#     {memory.buffer}

#     Query: {query}

#     Response (as the identified king or queen):
#     """

#     # Get the final response from the LLM
#     final_response = llm._call(royal_prompt)

#     # Update the conversation memory
#     memory.save_context({'input': query}, {'output': final_response})

#     return final_response

# def determine_voice(text, llm):
#     prompt = f"""
#     Based on the following text, determine if it's spoken by a male or female character.
#     Respond with only 'male' or 'female'.
#     If you can't determine the gender, respond with 'neutral'.

#     Text: {text}

#     Gender (male/female/neutral):
#     """
    
#     response = llm._call(prompt).strip().lower()
    
#     if response == 'male':
#         return 'male'
#     elif response == 'female':
#         return 'female'
#     else:
#         return 'neutral'

# app = Flask(__name__)

# html_content = """
# <!DOCTYPE html>
# <html lang="en">
# <head>
#     <meta charset="UTF-8">
#     <meta name="viewport" content="width=device-width, initial-scale=1.0">
#     <title>Royal Assistant</title>
#     <style>
#         body {
#             font-family: Arial, sans-serif;
#             line-height: 1.6;
#             margin: 0;
#             padding: 20px;
#             background-color: #f0f0f0;
#         }
#         .container {
#             max-width: 800px;
#             margin: 0 auto;
#             background-color: #ffffff;
#             padding: 20px;
#             border-radius: 8px;
#             box-shadow: 0 0 10px rgba(0,0,0,0.1);
#         }
#         h1 {
#             text-align: center;
#             color: #333;
#         }
#         #prompt-input {
#             width: 100%;
#             padding: 10px;
#             margin-bottom: 10px;
#             border: 1px solid #ddd;
#             border-radius: 4px;
#             box-sizing: border-box;
#         }
#         #send-button, #clear-button {
#             display: inline-block;
#             width: 48%;
#             padding: 10px;
#             background-color: #4CAF50;
#             color: white;
#             border: none;
#             border-radius: 4px;
#             cursor: pointer;
#             margin-top: 10px;
#         }
#         #clear-button {
#             background-color: #f44336;
#         }
#         #send-button:hover {
#             background-color: #45a049;
#         }
#         #clear-button:hover {
#             background-color: #e53935;
#         }
#         #response-output {
#             width: 100%;
#             height: 300px;
#             padding: 10px;
#             margin-top: 20px;
#             border: 1px solid #ddd;
#             border-radius: 4px;
#             background-color: #f9f9f9;
#             overflow-y: auto;
#             box-sizing: border-box;
#         }
#         .audio-player {
#             margin-top: 10px;
#             display: none;
#         }
#         .toggle-audio {
#             background-color: #008CBA;
#             border: none;
#             color: white;
#             padding: 5px 10px;
#             text-align: center;
#             text-decoration: none;
#             display: inline-block;
#             font-size: 12px;
#             margin: 4px 2px;
#             cursor: pointer;
#             border-radius: 4px;
#         }
#     </style>
# </head>
# <body>
#     <div class="container">
#         <h1>Royal Assistant</h1>
#         <input type="text" id="prompt-input" placeholder="Enter your prompt here...">
#         <button id="send-button">Send</button>
#         <button id="clear-button">Clear</button>
#         <div id="response-output"></div>
#     </div>

#     <script>
#         document.getElementById('send-button').addEventListener('click', function() {
#             var prompt = document.getElementById('prompt-input').value;
#             var responseOutput = document.getElementById('response-output');

#             fetch('/get_response', {
#                 method: 'POST',
#                 headers: {
#                     'Content-Type': 'application/json'
#                 },
#                 body: JSON.stringify({ prompt: prompt })
#             })
#             .then(response => response.json())
#             .then(data => {
#                 var responseHtml = '<p><strong>You:</strong> ' + prompt + '</p>';
#                 responseHtml += '<p><strong>Assistant:</strong> ' + data.response + '</p>';
#                 responseHtml += '<button class="toggle-audio" onclick="toggleAudio(this)">Show Audio</button>';
#                 responseHtml += '<audio class="audio-player" controls><source src="' + data.audio_url + '" type="audio/wav"></audio>';
                
#                 responseOutput.innerHTML += responseHtml;

#                 // Clear the input field
#                 document.getElementById('prompt-input').value = '';

#                 // Scroll to the bottom of the response output
#                 responseOutput.scrollTop = responseOutput.scrollHeight;
#             })
#             .catch(error => console.error('Error:', error));
#         });

#         document.getElementById('clear-button').addEventListener('click', function() {
#             document.getElementById('response-output').innerHTML = '';
#         });

#         // Allow sending the prompt by pressing Enter
#         document.getElementById('prompt-input').addEventListener('keypress', function(e) {
#             if (e.key === 'Enter') {
#                 document.getElementById('send-button').click();
#             }
#         });

#         function toggleAudio(button) {
#             var audioPlayer = button.nextElementSibling;
#             if (audioPlayer.style.display === "none" || audioPlayer.style.display === "") {
#                 audioPlayer.style.display = "block";
#                 button.textContent = "Hide Audio";
#             } else {
#                 audioPlayer.style.display = "none";
#                 button.textContent = "Show Audio";
#             }
#         }
#     </script>
# </body>
# </html>
# """

# @app.route('/')
# def index():
#     return render_template_string(html_content)

# @app.route('/get_response', methods=['POST'])
# def get_response_route():
#     data = request.get_json()
#     prompt = data['prompt']
#     response = get_response(prompt)
    
#     # Determine voice based on the LLM's interpretation of the response content
#     voice = determine_voice(response, llm)
    
#     # Set voice
#     voices = engine.getProperty('voices')
#     if len(voices) >= 2:
#         if voice == 'male':
#             engine.setProperty('voice', voices[0].id)
#         elif voice == 'female':
#             engine.setProperty('voice', voices[1].id)
#         else:
#             # If neutral or not enough voices, use the default voice
#             pass
#     else:
#         print("Warning: Not enough voices available for gender-specific selection.")
    
#     # Generate speech
#     audio_filename = f"temp_audio_{uuid.uuid4()}.wav"
#     audio_path = os.path.join('static', audio_filename)
#     engine.save_to_file(response, audio_path)
#     engine.runAndWait()
    
#     return jsonify({
#         'response': response,
#         'audio_url': f"/static/{audio_filename}"
#     })

# @app.route('/static/<path:filename>')
# def serve_static(filename):
#     return send_file(os.path.join('static', filename))

# def run_app():
#     if not os.path.exists('static'):
#         os.makedirs('static')
#     app.run(debug=True, use_reloader=False, host="0.0.0.0")

# # Run the Flask app in a separate thread
# if __name__ == "__main__":
#     threading.Thread(target=run_app).start()

# main.py (your main application file)
# main.py
import os
from langchain_community.llms import OpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from config import API_KEY, PDF_FOLDER_PATH
from langchain.llms.base import LLM
from typing import Any, List, Mapping, Optional
from openai import OpenAI as OpenAIClient
from flask import Flask, request, render_template_string, jsonify, send_file
import pyttsx3
import threading
import uuid

class LlamaLLM(LLM):
    client: Any = None

    def __init__(self):
        super().__init__()
        self.client = OpenAIClient(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=API_KEY
        )

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        completion = self.client.chat.completions.create(
            model="meta/llama-3.3-70b-instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            top_p=0.7,
            max_tokens=1024,
        )
        return completion.choices[0].message.content

    @property
    def _llm_type(self) -> str:
        return "Llama 3.3"

# Initialize components
llm = LlamaLLM()
memory = ConversationBufferMemory()

# Load and process PDFs
loader = DirectoryLoader(PDF_FOLDER_PATH, glob="**/*.pdf", loader_cls=PyPDFLoader)
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(texts, embeddings)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
    memory=memory,
)

engine = pyttsx3.init()

def get_response(query: str) -> str:
    raw_response = qa_chain.invoke(query)

    royal_prompt = f"""
    Respond as a historical royal figure mentioned in the query. 
    Use first-person perspective and be gender-specific.
    Respond in the query's language. Be authoritative but polite.
    Use only context information. If unsure, respond as a monarch would.

    Context: {raw_response}
    Previous conversation: {memory.buffer}
    Query: {query}

    Royal Response:
    """

    final_response = llm._call(royal_prompt)
    memory.save_context({'input': query}, {'output': final_response})
    return final_response

def determine_voice(text: str, llm: LlamaLLM) -> str:
    prompt = f"""Analyze if the speaker in this text is female. 
    Respond ONLY with 'female' for queens/goddesses, 'male' for kings/pharaohs, 
    or 'neutral' if uncertain.

    Text: {text}
    Gender:"""
    
    try:
        response = llm._call(prompt).strip().lower()
        if 'female' in response:
            return 'female'
        elif 'male' in response:
            return 'male'
        return 'neutral'
    except Exception as e:
        print(f"Voice detection error: {e}")
        return 'neutral'

app = Flask(__name__)

html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Royal Assistant</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f0f0f0;
        }
        .container {
            max-width: 800px;
            margin: 0 auto;
            background-color: #ffffff;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }
        h1 {
            text-align: center;
            color: #333;
        }
        #prompt-input {
            width: 100%;
            padding: 10px;
            margin-bottom: 10px;
            border: 1px solid #ddd;
            border-radius: 4px;
            box-sizing: border-box;
        }
        #send-button, #clear-button {
            display: inline-block;
            width: 48%;
            padding: 10px;
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            margin-top: 10px;
        }
        #clear-button {
            background-color: #f44336;
        }
        #send-button:hover {
            background-color: #45a049;
        }
        #clear-button:hover {
            background-color: #e53935;
        }
        #response-output {
            width: 100%;
            height: 300px;
            padding: 10px;
            margin-top: 20px;
            border: 1px solid #ddd;
            border-radius: 4px;
            background-color: #f9f9f9;
            overflow-y: auto;
            box-sizing: border-box;
        }
        .audio-player {
            margin-top: 10px;
            display: none;
        }
        .toggle-audio {
            background-color: #008CBA;
            border: none;
            color: white;
            padding: 5px 10px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 12px;
            margin: 4px 2px;
            cursor: pointer;
            border-radius: 4px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Royal Assistant</h1>
        <input type="text" id="prompt-input" placeholder="Enter your prompt here...">
        <button id="send-button">Send</button>
        <button id="clear-button">Clear</button>
        <div id="response-output"></div>
    </div>

    <script>
        document.getElementById('send-button').addEventListener('click', function() {
            var prompt = document.getElementById('prompt-input').value;
            var responseOutput = document.getElementById('response-output');

            fetch('/get_response', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ prompt: prompt })
            })
            .then(response => response.json())
            .then(data => {
                var responseHtml = '<p><strong>You:</strong> ' + prompt + '</p>';
                responseHtml += '<p><strong>Assistant:</strong> ' + data.response + '</p>';
                responseHtml += '<button class="toggle-audio" onclick="toggleAudio(this)">Show Audio</button>';
                responseHtml += '<audio class="audio-player" controls><source src="' + data.audio_url + '" type="audio/wav"></audio>';
                
                responseOutput.innerHTML += responseHtml;

                // Clear the input field
                document.getElementById('prompt-input').value = '';

                // Scroll to the bottom of the response output
                responseOutput.scrollTop = responseOutput.scrollHeight;
            })
            .catch(error => console.error('Error:', error));
        });

        document.getElementById('clear-button').addEventListener('click', function() {
            document.getElementById('response-output').innerHTML = '';
        });

        // Allow sending the prompt by pressing Enter
        document.getElementById('prompt-input').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                document.getElementById('send-button').click();
            }
        });

        function toggleAudio(button) {
            var audioPlayer = button.nextElementSibling;
            if (audioPlayer.style.display === "none" || audioPlayer.style.display === "") {
                audioPlayer.style.display = "block";
                button.textContent = "Hide Audio";
            } else {
                audioPlayer.style.display = "none";
                button.textContent = "Show Audio";
            }
        }
    </script>
</body>
</html>
"""
@app.route('/')  # <-- Add this route handler
def index():
    return render_template_string(html_content)
@app.route('/get_response', methods=['POST'])
def get_response_route():
    data = request.get_json()
    prompt = data['prompt']
    response = get_response(prompt)
    
    # Determine voice using enhanced detection
    voice = determine_voice(response, llm)
    print(f"Selected voice: {voice}")

    # Configure TTS engine
    try:
        voices = engine.getProperty('voices')
        print(f"Available voices: {[v.name for v in voices]}")
        
        if len(voices) >= 2:
            if voice == 'female':
                # Find first available female voice
                female_voices = [v for v in voices if 'female' in v.name.lower() or 'woman' in v.name.lower()]
                if female_voices:
                    engine.setProperty('voice', female_voices[0].id)
                else:
                    engine.setProperty('voice', voices[1].id)
            else:
                engine.setProperty('voice', voices[0].id)
    except Exception as e:
        print(f"Voice config error: {e}")

    # Generate audio
    audio_filename = f"temp_audio_{uuid.uuid4()}.wav"
    audio_path = os.path.join('static', audio_filename)
    engine.save_to_file(response, audio_path)
    engine.runAndWait()
    
    return jsonify({
        'response': response,
        'audio_url': f"/static/{audio_filename}"
    })

def run_app():
    if not os.path.exists('static'):
        os.makedirs('static')
    app.run(debug=True, use_reloader=False, host="0.0.0.0")

if __name__ == "__main__":
    threading.Thread(target=run_app).start()