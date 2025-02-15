import os
import faiss
import numpy as np
import google.generativeai as genai
from langchain import PromptTemplate
from langchain.memory import ConversationBufferMemory
import gradio as gr

API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("Missing API key! Set GOOGLE_API_KEY as an environment variable.")


def get_embedding(text):
    """Generate an embedding for a given text using Gemini."""
    response = genai.embed_content(
        model="models/embedding-001",
        content=text,
        task_type="retrieval_document"
    )
    return np.array(response["embedding"])

class JobApplicationChatbot:
    def __init__(self, api_key):
        if not api_key:
            raise ValueError("Missing API key! Set GOOGLE_API_KEY.")

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-pro")
        self.memory = ConversationBufferMemory(memory_key="history")
        self.personal_info = {}

        # Load FAISS index and document texts
        try:
            self.index = faiss.read_index("faiss_index.bin")
            self.doc_texts = np.load("doc_texts.npy", allow_pickle=True)
        except FileNotFoundError:
            raise FileNotFoundError("FAISS index or document texts file not found!")

        self.prompt = PromptTemplate(
            input_variables=["personal_info", "history", "context", "input"],
            template="""
            You are an final year AI student named Ali, helping recruiters to learn about yourself in a positive way.
            Answer questions only based on the provided information.

            Relevant Context:
            {context}

            Personal Information:
            {personal_info}

            Conversation History:
            {history}

            Recruiter's Question: {input}

            Provide a professional and concise response.
            """
        )

    def get_relevant_context(self, query, k=11):
        """Retrieve relevant context using FAISS"""
        if self.index is None or len(self.doc_texts) == 0:
            return ""
        
        try:
            query_embedding = np.array([get_embedding(query)])
            D, I = self.index.search(query_embedding, k)
            relevant_docs = [self.doc_texts[i] for i in I[0]]
            return "\n".join(relevant_docs)
        except Exception as e:
            print(f"Error retrieving context: {str(e)}")
            return ""

    def respond(self, message):
        if not message.strip():
            return "Please enter a valid question."

        # Get relevant context
        context = self.get_relevant_context(message)

        personal_info_str = "\n".join(f"{k}: {v}" for k, v in self.personal_info.items())

        formatted_prompt = self.prompt.format(
            personal_info=personal_info_str,
            history=self.memory.buffer,
            context=context,
            input=message
        )

        try:
            response = self.model.generate_content(formatted_prompt)
            response_text = response.text
            self.memory.save_context({"input": message}, {"output": response_text})
            return response_text
        except Exception as e:
            return f"Error processing request: {str(e)}"

    def update_personal_info(self, new_info):
        self.personal_info.update(new_info)

# Initialize chatbot
chatbot = JobApplicationChatbot(API_KEY)

# Load personal details
chatbot.update_personal_info({
    "Name": "Ali Mokrani",
    "Email": "ali.mokrani@g.enp.edu.dz",
    "Phone": "+213779778331",
    "LinkedIn": "https://www.linkedin.com/in/ali-mokrani",
    "GitHub": "https://github.com/sidmodz",
    "Resume": 'https://drive.google.com/file/d/1E84kMrVnJ8WiB9i0ay9cXM2VJcpLiapf/view?usp=sharing'
})

def chat_response(message, history):
    try:
        if not message.strip():
            return "Please enter a valid question."
            
        return chatbot.respond(message)
            
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Create Gradio Chat Interface
demo = gr.ChatInterface(
    fn=chat_response,
    title="🤖 Ali Chatbot",
    description="""
                            Welcome! I am an AI assistant that can answer all of your questions about Ali!
    
    """,
    examples=[["Present yourself."],
        ["Tell me about your experience."],
        ["What projects have you worked on?"],
        ["What are your technical skills?"],
        ["Do you have any certifications?"],
        ["Where can I see your portfolio?"]
    ],
    theme=gr.themes.Soft()
)

if __name__ == "__main__":
    demo.launch(share=True)
