# Ali-chat

This repository contains a chatbot that utilizes the Gemini API and Retrieval-Augmented Generation (RAG) to answer questions about me based on my resume.
## Features
- Uses Gemini API for intelligent responses
- Implements RAG for accurate and contextual answers based on my resume
- Interactive chatbot interface built with Gradio
- FAISS for efficient document retrieval

## Repository Structure
```
├── README.md              # Project documentation
├── app.py                  # Main application file for running the chatbot
├── doc_texts.npy          # Processed document embeddings
├── faiss_index.bin        # FAISS index for document retrieval
├── generate_index.py      # Script to process documents and generate FAISS index
├── requirements.txt        # List of dependencies
```

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/sidmodz/Ali-chat.git
   cd chatbot-resume
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Generate the FAISS index (if not already generated):
   ```bash
   python generate_index.py
   ```
2. Run the chatbot:
   ```bash
   python app.py
   ```
3. Access the chatbot interface via the link provided by Gradio.

## How It Works
1. The `generate_index.py` script processes your documents (resume in my case) and stores the document embeddings.
2. `app.py` loads the FAISS index and interacts with the Gemini API to provide responses.
3. Users can ask questions about my resume, and the chatbot retrieves relevant context before generating answers.

## Contributions
Feel free to fork this repository, submit issues, or suggest improvements.




