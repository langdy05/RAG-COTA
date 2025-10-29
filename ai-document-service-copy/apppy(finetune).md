# NOTE:
THIS IS FOR USING THE FINE TUNED MODEL ON GOOGLE CLOUD.
---

import os
import json
import base64
import re
import random
import textwrap
import logging
import pypdf
from flask import Flask, request, jsonify
from flask_cors import CORS
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig
from google.api_core.exceptions import GoogleAPICallError
from google.oauth2 import service_account

# --- Global Config ---
RAG_CHUNKS = []
DOCUMENT_DIR = None

def get_document_dir():
    global DOCUMENT_DIR
    
    # Render deployment detection
    if os.path.exists('/opt/render/project/src'):
        render_paths = [
            os.getenv("DOCUMENT_DIR"),
            "/opt/render/project/src/documents/documents",
            "./documents/documents",
            "documents/documents"
        ]
        for path in render_paths:
            if path and os.path.isdir(path):
                logging.info(f"Render: Using path {path}")
                DOCUMENT_DIR = path
                return path
        return None
    
    # Local development - try multiple paths
    local_paths = [
        os.getenv("DOCUMENT_DIR"),
        "./documents/documents",
        "documents/documents",
        "AI-DOCUMENT-SERVICE/documents/documents",
        "/Users/macbookpro/Desktop/ai rag testing/ai-document-service/documents/documents",
    ]
    
    for path in local_paths:
        if path and os.path.isdir(path):
            logging.info(f"Local: Using path {path}")
            DOCUMENT_DIR = path
            return path
    
    # Fallback search from current directory
    cwd = os.getcwd()
    for root, dirs, _ in os.walk(cwd):
        if 'documents' in dirs:
            docs_path = os.path.join(root, 'documents', 'documents')
            if os.path.isdir(docs_path):
                logging.info(f"Found documents at: {docs_path}")
                DOCUMENT_DIR = docs_path
                return docs_path
    
    logging.warning("No documents directory found")
    return None

# Initialize document directory BEFORE logging
get_document_dir()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
PROJECT_ID = os.getenv("GCP_PROJECT_ID")
LOCATION = os.getenv("GCP_LOCATION")
MODEL_NAME = os.getenv("GCP_ENDPOINT_ID")
SA_KEY_B64 = os.getenv("GCP_SA_KEY_B64")
app = Flask(__name__)
CORS(app, resources={r"/ask": {"origins": ["http://localhost:5500", "*"]}})
llm_model = None

def clean_text(text):
    text = text.replace('\t', ' ')
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()

def chunk_text(text, chunk_size=768, overlap=150):
    chunks, i, n = [], 0, len(text)
    while i < n:
        chunks.append(text[i:i + chunk_size])
        i += chunk_size - overlap
    return chunks

def load_and_chunk_documents():
    global RAG_CHUNKS
    
    # Use a local variable first, only set global at the end if needed
    current_docs_dir = DOCUMENT_DIR
    
    if not current_docs_dir or not os.path.isdir(current_docs_dir):
        logging.warning(f"Documents directory not found: {current_docs_dir}")
        logging.info(f"Current working dir: {os.getcwd()}")
        logging.info(f"Root contents: {os.listdir('.')}")
        
        # Final check for documents/documents in current dir
        docs_root = os.path.join(os.getcwd(), 'documents', 'documents')
        if os.path.isdir(docs_root):
            logging.info(f"Found documents/documents at: {docs_root}")
            current_docs_dir = docs_root
        
        # If still no valid directory, exit
        if not current_docs_dir or not os.path.isdir(current_docs_dir):
            logging.error("No valid documents directory found. Cannot proceed.")
            return
    
    logging.info(f"Loading documents from: {current_docs_dir}")
    file_count = 0
    
    for root, _, files in os.walk(current_docs_dir):
        for filename in files:
            if filename.lower().endswith(('.txt', '.md', '.markdown', '.pdf')):
                filepath = os.path.join(root, filename)
                text = ""
                try:
                    if filename.lower().endswith(('.txt', '.md', '.markdown')):
                        with open(filepath, 'r', encoding='utf-8') as f:
                            text = f.read()
                    elif filename.lower().endswith('.pdf'):
                        with open(filepath, 'rb') as f:
                            reader = pypdf.PdfReader(f)
                            for page in reader.pages:
                                text += page.extract_text() or ""
                    text = clean_text(text)
                    if text.strip():
                        rel_path = os.path.relpath(filepath, current_docs_dir)
                        tagged_chunks = [f"Source: {rel_path}\\n\\n{chunk}" for chunk in chunk_text(text)]
                        RAG_CHUNKS.extend(tagged_chunks)
                        file_count += 1
                        logging.info(f"Loaded {rel_path} ({len(tagged_chunks)} chunks)")
                except Exception as e:
                    logging.error(f"Error loading {filepath}: {e}")
    
    logging.info(f"RAG system ready with {len(RAG_CHUNKS)} chunks from {file_count} files")

def retrieve_context(question, num_chunks=8):
    if not RAG_CHUNKS:
        return ""
    question_tokens = set(re.findall(r'\b\w+\b', question.lower()))
    scored = []
    for chunk in RAG_CHUNKS:
        score = len(question_tokens.intersection(set(re.findall(r'\b\w+\b', chunk.lower()))))
        if score > 1:
            scored.append((score, chunk))
    scored.sort(key=lambda x: (x[0], random.random()), reverse=True)
    return "\n---\n".join(chunk for _, chunk in scored[:num_chunks])

def _initialize_llm_model():
    global llm_model
    if llm_model:
        return llm_model
    load_and_chunk_documents()
    try:
        credentials = None
        if SA_KEY_B64:
            key_json_bytes = base64.b64decode(SA_KEY_B64)
            key_json = json.loads(key_json_bytes)
            credentials = service_account.Credentials.from_service_account_info(key_json)
        vertexai.init(project=PROJECT_ID, location=LOCATION, credentials=credentials)
        llm_model = GenerativeModel(model_name=MODEL_NAME)
        logging.info(f"Initialized Vertex AI model: {MODEL_NAME}")
        return llm_model
    except Exception as e:
        logging.error(f"Vertex AI init error: {e}")
        raise RuntimeError("Failed to initialize Vertex AI model.")

@app.route('/')
def home():
    return jsonify({
        "message": "NEXUS AI RAG Server is running.",
        "docs_path": DOCUMENT_DIR,
        "docs_exists": DOCUMENT_DIR and os.path.isdir(DOCUMENT_DIR),
        "chunks_loaded": len(RAG_CHUNKS),
        "environment": "Render" if os.path.exists('/opt/render/project/src') else "Local"
    }), 200

@app.route('/debug')
def debug():
    info = {
        "DOCUMENT_DIR": DOCUMENT_DIR,
        "docs_exists": DOCUMENT_DIR and os.path.isdir(DOCUMENT_DIR),
        "cwd": os.getcwd(),
        "is_render": os.path.exists('/opt/render/project/src'),
        "root_contents": os.listdir('.') if os.path.exists('.') else [],
        "documents_folder": os.path.exists('documents'),
        "nested_documents": os.path.exists('documents/documents') if os.path.exists('documents') else False
    }
    
    if os.path.exists('documents'):
        try:
            info["documents_contents"] = os.listdir('documents')[:10]
            if os.path.exists('documents/documents'):
                info["nested_docs_contents"] = os.listdir('documents/documents')[:10]
        except Exception as e:
            info["documents_error"] = str(e)
    
    return jsonify(info)

@app.route('/ask', methods=['POST'])
def ask_llm():
    try:
        data = request.get_json()
        question = data.get('question')
        if not question:
            return jsonify({"error": "Missing 'question' field."}), 400
        
        model = _initialize_llm_model()
        retrieved_context = retrieve_context(question)

        if retrieved_context:
            prompt = textwrap.dedent(f"""
You are NEXUS AI, a sophisticated Legal AI Assistant specializing in the Cambodian Law.
                Your answers must be based only on the provided CONTEXT.
                
                **Core Directives:**
                1.  **Expertise:** Act as an expert on the Cambodian Law respond only to questions and prompts about laws, legal terms and legal topics only. For any other topic, you must respond with: 'I am NEXUS AI, specialized only in the Cambodian Law. I cannot provide information on other topics.'
                2.  **Context is King:** Your answers must be derived *strictly* from the provided CONTEXT. Do not use any external knowledge.
                3.  **Handle Insufficient Context:** If the CONTEXT does not contain the information to answer the question, you must respond with: 'I currently do not have a detailed understanding of this topic. However, I am able to give you a brief overview of this topic.'

                **Response Style and Structure:**
                - **Be Explanatory:** Do not give short, direct answers. Explain the 'why' and 'how' behind each point as if explaining to the user to understand the topic. But, strictly stick to the information provided in the context. 
                - **Summarize First:** Begin your response with a concise summary that directly addresses the user's question.
                - **Provide Detailed Elaboration:** Following the summary, offer a detailed breakdown. Use bullet points or numbered lists to structure complex information. Explain key legal terms and concepts in simple language.
                - **Cite Evidence:** Support your explanations by referencing the CONTEXT. You can quote key phrases or mention the source document (e.g., "According to the constitution/commercial law it states...").
                - **Professional Tone:** Maintain a helpful, educational, and professional tone befitting a legal assistant.
                - **Detailed and Hierarchical Formatting:** Always provide comprehensive and detailed answers. Structure all responses using clear hierarchical markdown syntax to organize the information logically.
                    * Use a main heading for the overall topic (\`# Title\`).
                    * Use sub-headings for major sections (\`## Section\`).
                    * Use further sub-headings for specific details (\`### Detail\`).
                    * Use **bold text** to emphasize key terms official names dates or important figures. For example use bold for terms like **Law on Military Service** or for dates like **enforced starting in 2026**.
                    * Use standard bulleted lists (\`*\`) for clarity when presenting multiple points.
                    * To break down a single bullet point further embed sub-points directly within the main point's sentence. Use hyphens for these inline items to keep them on the same indentation level. Example: \`* Key benefits include: - enhanced discipline - valuable life skills and - priority in future employment.\`
                    * Conclude with a summary where appropriate.
                - **Language Use:** Respond in English if the question is in English. Respond in Khmer if the question is in Khmer.
                - **Scope Warning:** If a query is completely outside your specialization of Cambodian law for example asking about French or US law you must clearly state this limitation and advise the user to consult an appropriate source.
                - **Interaction Flow:**
                    **Initial Greeting:** If the user's first input is a greeting provide a single brief professional acknowledgment and introduction. From then on for the following questions and prompt DO NOT introduce yourself, just answer the user directly. 
                    **Follow-Up Questions:** Maintain the context of the conversation for follow-up questions. If a new question is unrelated to the previous one then treat it as a fresh inquiry.

                **Disclaimer Rule:**
                - Only for questions that ask for advice, legal opinions, or document drafting, you must end your entire response with the disclaimer: 'Consult qualified lawyers for personalized advice.' For all other informational questions, do not include this disclaimer.

                ---
                
                CONTEXT:
                {retrieved_context}
                
                Question: {question}
            """)
        else:
            prompt = f"You are NEXUS AI, specialized in Cambodian Law. No specific knowledge on this question. Question: {question}"

        response = model.generate_content(
            contents=[prompt],
            generation_config=GenerationConfig(temperature=0.2, max_output_tokens=2048)
        )
        if hasattr(response, "text") and response.text:
            return jsonify({
                "question": question, 
                "answer": response.text, 
                "context_found": bool(retrieved_context),
                "total_chunks": len(RAG_CHUNKS)
            })
        return jsonify({"error": "Model returned no output."}), 500
    except GoogleAPICallError as e:
        return jsonify({"error": f"Vertex AI failed: {str(e)}"}), 500
    except Exception as e:
        logging.exception("Unexpected /ask error")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.getenv('PORT', 8080)))