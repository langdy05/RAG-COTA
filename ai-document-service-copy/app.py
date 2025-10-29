import os
import re
import textwrap
import logging
import pypdf
import subprocess  # Added for downloading/extracting
import os.path     # Added for checking if DB folder exists
import sys         # Added for sys.argv and sys.exit
from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
import chromadb
import hashlib
import json
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException

# --- Configure logging at the very top ---
# Ensure logging is set up before any messages are emitted
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(process)d - %(levelname)s - %(message)s')

# --- Global Config & Clients ---
# Initialized as None, will be set by initialize_clients()
db_client = None
doc_collection = None
llm_model = None
EMBEDDING_MODEL_NAME = None

# Get the database path from an environment variable.
# Defaults to './chroma_db' for local, uses DB_MOUNT_PATH on Render if set.
DB_PATH = os.getenv("DB_MOUNT_PATH", "./chroma_db")
logging.info(f"Database path set to: {DB_PATH}") # Log DB path early

# --- Gemini API Configuration ---
# Configure API key early. Critical failure if missing/invalid.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logging.critical("Missing GEMINI_API_KEY environment variable. Application cannot start.")
    sys.exit("Startup Error: Missing GEMINI_API_KEY") # Use exit code for clarity
try:
    logging.info("Configuring Gemini API...")
    genai.configure(api_key=GEMINI_API_KEY)
    logging.info("Gemini API configured successfully.")
except Exception as e:
    logging.critical(f"Failed to configure Gemini API: {e}. Application cannot start.")
    sys.exit(f"Startup Error: Failed to configure Gemini API: {e}")


# --- Function Definitions ---

def initialize_clients():
    """Initializes global clients (DB, LLM, Embedding Model Name). Called once per worker."""
    global db_client, doc_collection, llm_model, EMBEDDING_MODEL_NAME
    
    # Avoid re-initialization if already done in this process
    if llm_model is not None:
         logging.warning("initialize_clients called again, but clients seem already initialized. Skipping.")
         return

    try:
        logging.info(f"Initializing clients using database path: {DB_PATH}")
        # Add robustness: Check if DB_PATH actually exists before initializing client
        if not os.path.isdir(DB_PATH):
             logging.critical(f"Database directory '{DB_PATH}' does not exist. Cannot initialize Chroma client.")
             raise RuntimeError(f"Database directory '{DB_PATH}' missing after setup.")

        db_client = chromadb.PersistentClient(path=DB_PATH)
        # Ensure collection exists or is created
        doc_collection = db_client.get_or_create_collection(name="cambodian_law")
        llm_model = genai.GenerativeModel("gemini-2.5-flash-lite") # Using your specified model
        EMBEDDING_MODEL_NAME = "models/text-embedding-004"
        # Check connection by getting count, handle potential DB errors
        db_count = doc_collection.count()
        logging.info(f"Database and models initialized successfully by worker process. Total chunks in DB: {db_count}")
    except Exception as e:
        logging.critical(f"CRITICAL: Failed to initialize ChromaDB or Gemini models: {e}")
        # Re-raise the exception to signal fatal error during worker startup
        raise RuntimeError(f"Failed to initialize clients in worker: {e}") from e

def check_and_download_db():
    """
    Checks if the database folder exists. If not, downloads and extracts it from R2.
    Returns True if the database folder is ready, False otherwise. Called once per worker.
    """
    DB_DOWNLOAD_URL = os.getenv("DB_DOWNLOAD_URL")

    if os.path.isdir(DB_PATH):
        logging.info(f"Database folder found at {DB_PATH}. Setup complete for this worker.")
        return True

    logging.warning(f"Database folder not found at {DB_PATH} for this worker.")
    if not DB_DOWNLOAD_URL:
        logging.critical("DB_DOWNLOAD_URL environment variable is not set. Worker cannot download database.")
        return False

    logging.info(f"Attempting to download database from {DB_DOWNLOAD_URL}...")
    # Use a worker-specific temporary filename to potentially avoid conflicts if multiple workers start simultaneously
    # Though Render typically starts them sequentially or isolates filesystems
    worker_pid = os.getpid()
    tar_path = f"chroma_db_{worker_pid}.tar.gz"
    
    try:
        download_timeout = 180 # 3 minutes
        logging.info(f"Starting download to {tar_path} with timeout {download_timeout}s...")
        subprocess.run(
            ["curl", "-f", "-L", DB_DOWNLOAD_URL, "-o", tar_path, "--connect-timeout", "30"],
            check=True, timeout=download_timeout
        )
        
        if not os.path.exists(tar_path) or os.path.getsize(tar_path) == 0:
             logging.critical(f"Download command completed but the file {tar_path} is missing or empty.")
             return False

        logging.info(f"Download complete ({os.path.getsize(tar_path)} bytes). Extracting database...")
        # Extract to the current directory, should create the DB_PATH folder
        subprocess.run(["tar", "-xzf", tar_path], check=True)
        
        if not os.path.isdir(DB_PATH):
            logging.critical(f"Extraction command completed but directory '{DB_PATH}' was not created.")
            try: subprocess.run(["rm", tar_path], check=False)
            except: pass
            return False

        logging.info(f"Database successfully extracted to {DB_PATH}. Cleaning up {tar_path}...")
        subprocess.run(["rm", tar_path], check=True)
        logging.info("Cleanup complete. Database ready for this worker.")
        return True

    except subprocess.TimeoutExpired:
        logging.critical(f"Database download timed out after {download_timeout} seconds.")
    except subprocess.CalledProcessError as cpe:
         logging.critical(f"A command failed during DB setup: {cpe}")
    except Exception as e:
        logging.critical(f"An unexpected error occurred during database download/extraction: {e}", exc_info=True)
    finally:
        # Ensure cleanup happens even on error
        if os.path.exists(tar_path):
            try:
                logging.info(f"Cleaning up potentially incomplete download: {tar_path}")
                subprocess.run(["rm", tar_path], check=False)
            except Exception as cleanup_err:
                logging.error(f"Error during cleanup of {tar_path}: {cleanup_err}")
                
    return False # Return False if any exception occurred


# --- LOCAL BUILD FUNCTIONS (Only used via 'python app.py build') ---

def get_document_dir():
    """Finds the local 'documents/documents' folder for indexing."""
    local_paths = ["./documents/documents", "documents/documents"]
    for path in local_paths:
        abs_path = os.path.abspath(path)
        if os.path.isdir(abs_path):
            logging.info(f"Local build: Using document path {abs_path}")
            return abs_path
    logging.error("Local build: Could not find the 'documents/documents' directory.")
    return None

def clean_text(text):
    """Removes extra whitespace from text."""
    if not isinstance(text, str): return ""
    text = text.replace('\t', ' ')
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()

def chunk_text(text, chunk_size=768, overlap=150):
    """Splits text into overlapping chunks."""
    if not isinstance(text, str): return []
    chunks, i, n = [], 0, len(text)
    effective_overlap = min(overlap, chunk_size - 1) if chunk_size > 0 else 0
    step = max(1, chunk_size - effective_overlap)
    while i < n:
        chunks.append(text[i:min(i + chunk_size, n)])
        i += step
    return chunks

def get_file_hash(content):
    """Calculates the SHA256 hash of a file's content."""
    # We use the raw text content to ensure consistency
    return hashlib.sha256(content.encode('utf-8')).hexdigest()

def load_and_chunk_documents():
    """
    Runs LOCALLY via 'python app.py build'.
    Performs a "smart sync":
    1. Deletes files from DB that are no longer on disk.
    2. Skips files that are unchanged (using a hash).
    3. Deletes and re-adds files that have been modified.
    4. Adds new files.
    """
    global db_client, doc_collection, EMBEDDING_MODEL_NAME

    if not doc_collection or not EMBEDDING_MODEL_NAME:
        logging.critical("Local build error: Clients not initialized. Run build init first.")
        sys.exit(1)

    current_docs_dir = get_document_dir()
    if not current_docs_dir:
        logging.critical("Local build error: Document directory not found.")
        sys.exit(1)

    logging.info(f"--- Starting local document sync from: {current_docs_dir} ---")
    
    # --- STEP 1: Get all files currently on disk ---
    logging.info("Scanning local document directory...")
    disk_files = {} # Stores {rel_path: abs_path}
    try:
        for root, _, files in os.walk(current_docs_dir):
            for filename in files:
                if filename.lower().endswith(('.txt', '.md', '.markdown', '.pdf')):
                    filepath = os.path.join(root, filename)
                    rel_path = os.path.relpath(filepath, current_docs_dir)
                    disk_files[rel_path] = filepath
        logging.info(f"Found {len(disk_files)} valid source files on disk.")
    except Exception as e:
        logging.error(f"Error scanning disk files: {e}. Aborting build.", exc_info=True)
        sys.exit(1)
        
    # --- STEP 2: Get all file records currently in the database ---
    logging.info("Fetching existing document records from database...")
    db_files = {} # Stores {rel_path: file_hash}
    try:
        if doc_collection.count() > 0:
            # Get metadata for all entries
            all_db_entries = doc_collection.get(include=["metadatas"])
            all_metadatas = all_db_entries.get('metadatas', [])
            
            # Create a set of unique source files and their hashes in the DB
            for meta in all_metadatas:
                if meta and 'source' in meta and meta.get('source') not in db_files:
                    db_files[meta['source']] = meta.get('file_hash', None)
            logging.info(f"Found {len(db_files)} unique source files in the database.")
        else:
            logging.info("Database is empty. No sync/delete needed.")
    except Exception as e:
        logging.error(f"Error fetching existing DB records: {e}. Aborting.", exc_info=True)
        sys.exit(1)

    # --- STEP 3: Sync Deletions (In DB, not on disk) ---
    files_to_delete = set(db_files.keys()) - set(disk_files.keys())
    if files_to_delete:
        logging.warning(f"Found {len(files_to_delete)} files in DB to delete (removed from disk).")
        for rel_path in files_to_delete:
            logging.info(f"Deleting chunks for '{rel_path}'...")
            try:
                # This 'where' filter is more efficient than getting all IDs
                doc_collection.delete(where={"source": rel_path})
            except Exception as e:
                logging.error(f"Failed to delete chunks for '{rel_path}': {e}")
        logging.info("Orphaned file deletion complete.")
    else:
        logging.info("No orphaned files to delete.")

    # --- STEP 4: Sync Additions/Modifications (On disk) ---
    logging.info("Checking disk files for additions or modifications...")
    files_processed = 0
    chunks_added = 0

    for rel_path, filepath in disk_files.items():
        text = ""
        try:
            # --- Read file content first ---
            if filepath.lower().endswith(('.txt', '.md', '.markdown')):
                with open(filepath, 'r', encoding='utf-8-sig') as f: text = f.read()
            elif filepath.lower().endswith('.pdf'):
                with open(filepath, 'rb') as f:
                    reader = pypdf.PdfReader(f)
                    for page in reader.pages:
                        page_text = page.extract_text(); text += (page_text + "\n") if page_text else ""

            if not text.strip():
                logging.warning(f"No text extracted from '{rel_path}'. Skipping."); continue
            
            # --- Compare Hashes ---
            current_hash = get_file_hash(text)
            old_hash = db_files.get(rel_path)

            if current_hash == old_hash:
                logging.debug(f"Skipping '{rel_path}', content is unchanged.")
                continue
            
            # --- Process File (it's new or modified) ---
            if old_hash:
                logging.info(f"File '{rel_path}' has changed. Deleting old chunks...")
                try:
                    doc_collection.delete(where={"source": rel_path})
                except Exception as e:
                    logging.error(f"Failed to delete old chunks for modified file '{rel_path}': {e}. Skipping."); continue
            else:
                logging.info(f"Processing new file: '{rel_path}'...")

            cleaned_text = clean_text(text)
            chunks = chunk_text(cleaned_text)
            if not chunks: logging.warning(f"Zero chunks created for '{rel_path}'. Skipping."); continue

            logging.info(f"Generating {len(chunks)} embeddings for '{rel_path}'...")
            try:
                embeddings_response = genai.embed_content(
                    model=EMBEDDING_MODEL_NAME, 
                    content=chunks,
                    task_type="RETRIEVAL_DOCUMENT" # This is critical!
                )
                embeddings = embeddings_response['embedding']
            except Exception as embed_error: logging.error(f"Embedding failed for '{rel_path}': {embed_error}. Skipping."); continue

            ids = [f"{rel_path}_{i}" for i in range(len(chunks))]
            # --- [NEW] Add the hash to the metadata ---
            metadatas=[{"source": rel_path, "file_hash": current_hash} for _ in chunks]
            
            try:
                doc_collection.add(embeddings=embeddings, documents=chunks, metadatas=metadatas, ids=ids)
                logging.info(f"Successfully indexed '{rel_path}' ({len(chunks)} chunks).")
                files_processed += 1; chunks_added += len(chunks)
            except Exception as db_add_error: logging.error(f"Failed to add chunks for '{rel_path}' to DB: {db_add_error}")

        except Exception as file_proc_error: 
            logging.error(f"Error processing file '{filepath}': {file_proc_error}")

    try: total_chunks = doc_collection.count()
    except: total_chunks = "Error"
    logging.info("--- LOCAL DOCUMENT SYNC COMPLETE ---")
    logging.info(f"Processed (added/updated) {files_processed} files. Added {chunks_added} new chunks.")
    logging.info(f"Total chunks now in database: {total_chunks}")


# --- SERVER FUNCTIONS ---

def retrieve_context(question, num_chunks=15):
    """Performs semantic search on ChromaDB."""
    if not doc_collection or not EMBEDDING_MODEL_NAME:
        logging.error("retrieve_context error: Clients not initialized.")
        return ""
    try:
        if doc_collection.count() == 0: logging.warning("retrieve_context: No documents in DB."); return ""
    except Exception as count_err: logging.error(f"retrieve_context: DB count failed: {count_err}"); return ""

    try:
        logging.debug(f"Generating embedding for question (first 100): {question[:100]}...")
        question_embedding = genai.embed_content(model=EMBEDDING_MODEL_NAME, content=question, task_type="RETRIEVAL_QUERY")['embedding']
        logging.debug("Querying DB...")
        results = doc_collection.query(query_embeddings=[question_embedding], n_results=num_chunks)
        logging.debug(f"Query returned {len(results.get('ids', [[]])[0])} results.")

        if not results or not results.get('ids') or not results['ids'][0]:
            logging.warning("No relevant documents found for the question."); return ""
        retrieved_docs = results.get('documents', [[]])[0]
        metadatas = results.get('metadatas', [[]])[0]
        if len(retrieved_docs) != len(metadatas): logging.error("Mismatch in query results."); return ""

        context_chunks = [f"Source: {meta.get('source', 'Unknown')}\n\n{doc}" for doc, meta in zip(retrieved_docs, metadatas) if meta]
        logging.info(f"Retrieved {len(context_chunks)} context chunks. First chunk (first 200 chars): {context_chunks[0][:200] if context_chunks else 'N/A'}")
        return "\n---\n".join(context_chunks)
    except Exception as e:
        logging.error(f"Error during context retrieval: {e}", exc_info=True); return ""


# --- Flask App Definition & Routes ---
# Define app object BEFORE routes and startup logic
app = Flask(__name__)
CORS(app, resources={r"/ask": {"origins": ["http://localhost:5500", "http://127.0.0.1:5500", "*"]}})

@app.route('/')
def home():
    return "Hello from RAG-COTA!"
    """Basic health check / info endpoint."""
    db_count_str = "N/A"
    if doc_collection:
        try: db_count_str = str(doc_collection.count())
        except Exception as e: db_count_str = f"Error ({e})"
    return jsonify({
        "message": "COTA AI RAG Server is running (Optimized - R2 Remote DB).",
        "db_path": DB_PATH, "db_folder_exists": os.path.isdir(DB_PATH),
        "chunks_indexed_in_db": db_count_str,
    }), 200
@app.route('/debug')
def debug():
    """Endpoint for debugging info."""
    db_count_str = "N/A"
    if doc_collection:
        try: db_count_str = str(doc_collection.count())
        except Exception as e: db_count_str = f"Error ({e})"
    info = {
        "DB_PATH": DB_PATH, "db_folder_exists": os.path.isdir(DB_PATH),
        "db_chunks": db_count_str,
        "llm_model_initialized": llm_model is not None,
        "embedding_model_name": EMBEDDING_MODEL_NAME or "N/A",
        "cwd": os.getcwd(), "is_render": "RENDER" in os.environ,
        "env_db_download_url": os.getenv("DB_DOWNLOAD_URL", "Not Set"),
    }
    return jsonify(info)

@app.route('/ask', methods=['POST'])
def ask_llm():
    """Main RAG endpoint."""
    logging.info("--- Received request to /ask ---")
    if not llm_model or not doc_collection: # Check both essential clients
        logging.error("/ask error: LLM model or DB collection not initialized.")
        return jsonify({"error": "Service Unavailable: AI components not ready."}), 503
        
    try:
        data = request.get_json()
        if not data: logging.error("/ask: Invalid JSON."); return jsonify({"error": "Bad Request: Invalid JSON."}), 400
        question = data.get('question')
        history = data.get('history', [])

        if not isinstance(history, list):
            logging.error("/ask: Invalid 'history' format, must be a list."); 
            return jsonify({"error": "Bad Request: Invalid 'history' format."}), 400
        
        if not question or not isinstance(question, str) or not question.strip():
            logging.error("/ask: Missing/invalid 'question'."); return jsonify({"error": "Bad Request: Invalid 'question'."}), 400
        
        logging.info(f"Q (first 100): {question[:100]}...")

        # --- CRITICAL FIX: Only use USER turns for the RAG query ---
        # This prevents the model's own (potentially off-topic) answers from polluting the search.
        history_for_rag = " ".join([
            turn.get('content', '') for turn in history[-4:] 
            if turn.get('content') and turn.get('role', 'user') == 'user'
        ])
        rag_query = f"{history_for_rag} {question}".strip()
        logging.info(f"Contextualized RAG Query (first 100): {rag_query[:100]}...")
        
        # --- [NEW] EFFICIENT QUERY TRANSLATION STEP ---
        # Detect language locally to avoid an extra API call for English queries.

        translated_rag_query = rag_query # Default to the original query

        try:
            detected_lang = detect(rag_query)
            logging.info(f"Detected query language locally: '{detected_lang}'")

            # Only call the translation API if the language is NOT English
            if detected_lang != 'en':
                logging.info("Query is not English. Calling LLM to translate for retrieval...")
                translate_prompt = textwrap.dedent(f"""
                    Translate the following text to English for a database query.
                    Respond *only* with the translated text, nothing else.

                    Input: "{rag_query}"
                    Output:
                """)

                response = llm_model.generate_content(
                    translate_prompt,
                    generation_config=genai.types.GenerationConfig(temperature=0.0)
                )

                translated_rag_query = response.text.strip()
                logging.info(f"Query translated to: {translated_rag_query[:100]}...")

        except LangDetectException:
            # This can happen on very short or ambiguous text (e.g., "123")
            logging.warning("Language detection failed (ambiguous text?). Assuming English.")
            # We assume English and just use the original query, which is fine.
        except Exception as e:
            logging.error(f"Error during query translation: {e}. Using original query.")

        # --- END NEW TRANSLATION STEP ---
        
        translated_rag_query = rag_query # Default in case translation fails

        try:
            logging.debug("Detecting language and translating query to English for retrieval...")
            
            # Use the LLM to perform detection and translation in one shot.
            translate_prompt = textwrap.dedent(f"""
                Analyze the following text: "{rag_query}"
                
                Respond in JSON format with two keys:
                1. "language": The detected IETF language code (e.g., "en", "km").
                2. "translated_query": If the language is "en", return the original text. If the language is *not* "en", translate the text to English.
                
                Example 1 (Khmer):
                Input: "តើរដ្ឋធម្មនុញ្ញជាអ្វី?"
                Output: {{"language": "km", "translated_query": "What is the constitution?"}}
                
                Example 2 (English):
                Input: "What is the constitution?"
                Output: {{"language": "en", "translated_query": "What is the constitution?"}}
            """)
            
            response = llm_model.generate_content(
                translate_prompt,
                generation_config=genai.types.GenerationConfig(temperature=0.0) # Deterministic
            )
            
            response_text = response.text.strip()
            
            # Use regex to find the JSON block, in case the model adds backticks
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                translation_data = json.loads(json_match.group(0))
                detected_lang = translation_data.get("language")
                query_for_retrieval = translation_data.get("translated_query")
                
                if detected_lang != "en" and query_for_retrieval:
                    logging.info(f"Query language detected as '{detected_lang}'. Translated to English for retrieval: {query_for_retrieval[:100]}...")
                    translated_rag_query = query_for_retrieval
                else:
                    logging.info("Query language detected as 'en'. Using original query for retrieval.")
            else:
                logging.warning(f"Could not parse translation JSON from LLM response: {response_text}. Using original query.")
                
        except Exception as translate_err:
            logging.error(f"Error during query translation: {translate_err}. Using original query.", exc_info=True)
        # --- END NEW TRANSLATION STEP ---

        logging.info("Retrieving context...")
        # --- MODIFIED: Use the *translated* query for retrieval ---
        retrieved_context = retrieve_context(translated_rag_query)
        
        context_found = bool(retrieved_context)
        if not context_found: logging.warning("No relevant context found.")
        else: logging.info(f"Context retrieved ({len(retrieved_context)} chars).")

        formatted_history = "\n".join(
            [f"{turn.get('role', 'user').title()}: {turn.get('content', '')}" for turn in history]
        )

        # Construct prompt
        if context_found:
            # --- Using the Mark V Prompt with enhanced formatting ---
            prompt = textwrap.dedent(f"""
              You are COTA AI, a sophisticated Tourism AI Assistant specializing in **Cambodia Tourism Information**.
                    Your answers must be based *only* on the provided CONTEXT.

                    **--- RESPONSE RULES (In Order of Priority) ---**

                    **RULE 1: LANGUAGE AND TRANSLATION (HIGHEST PRIORITY)**
                    - You MUST respond in the *same language* as the "New Question".
                    - **IF the "New Question" is in KHMER:**
                        - You **MUST** assume the topic is "Cambodia Tourism".
                        - You **MUST** search the *entire* CONTEXT (which may be in English) for the answer.
                        - If the only relevant context is in **English**, you **MUST** use it to synthesize an answer and write the final answer in **KHMER**.
                        - You are **FORBIDDEN** from using the "off-topic" response (Rule 3) or the "insufficient context" response (Rule 2) if the relevant answer exists in English.

                    **RULE 2: INSUFFICIENT CONTEXT (FALLBACK)**
                    - **IF, AND ONLY IF,** after searching the *entire* context, there is *genuinely no relevant information* to answer the question...
                    - ...THEN, and *only* then, respond in the question's language with: 'I currently do not have detailed information on this topic. However, I can give you a brief overview of popular tourist attractions in Cambodia.'

                    **RULE 3: OFF-TOPIC (FOR NON-KHMER QUESTIONS)**
                    - This rule applies **ONLY IF** the "New Question" is in **ENGLISH** (or another language, but not Khmer).
                    - If the English question is *NOT* about Cambodia Tourism, respond *only* with: 'I am COTA AI, specialized only in Cambodia Tourism Information. I cannot provide information on other topics.'

                    **RULE 4: RESPONSE FORMATTING & STYLE (CRITICAL)**
                    - **Structure:**
                        1. Start with a **direct, concise answer** to the user's question.
                        2. Follow this with a **detailed elaboration**, organized logically.
                    - **Markdown:** You **MUST** use hierarchical Markdown to structure all answers for professionalism and readability.
                        * Use a main heading for the overall topic (e.g., `# Angkor Wat Temple Complex`).
                        * Use sub-headings for major sections (e.g., `## Visitor Guidelines`).
                        * Use further sub-headings for specific details (e.g., `### Ticket Prices & Hours`).
                        * Use `**Bold Text**` to emphasize key terms, attraction names, locations, or important travel tips.
                        * Use standard bulleted lists (`*` or `-`) for clarity and indented, nested lists (`  *`) for sub-points.
                    - **Tone & Diction:**
                        * Maintain a helpful, enthusiastic, and professional tone.
                        * Do not use or mention the term "provided context"; use "the current information I have".
                    - **Citations:** Support your explanations by referencing the CONTEXT (e.g., "According to the Ministry of Tourism..." or "Official guidelines state...").
                    - **Disclaimer:** **Only** for questions that ask for personalized travel planning, bookings, or safety advice, you must end your entire response with the disclaimer: 'Always verify the latest travel advisories and consult official sources before planning your trip.'
                    - **Flow:** Do not introduce yourself after the first greeting.

                ---
                CHAT HISTORY:
                {formatted_history}
                ---
                CONTEXT:
                {retrieved_context}
                ---
                New Question: {question}
            """)
            # --- END MODIFIED ---
        else:
            # --- MODIFIED: Added CHAT HISTORY block ---
            prompt = textwrap.dedent(f"""
                You are NEXUS AI, a sophisticated Legal AI Assistant specializing in the Cambodian Law.
                **Core Directives:**
                1.  **Expertise:** For topics outside Cambodian Law, respond: 'I am NEXUS AI, specialized only in the Cambodian Law. I cannot provide information on other topics.'
                
                ---
                CHAT HISTORY:
                {formatted_history}
                ---
                **Handle Insufficient Context:** No relevant documents were found for the new question, even after translating it. Respond ONLY with: 'I do not have sufficient information currently to provide a detailed answer on this specific topic.' Do not answer from general knowledge.
                ---
                New Question: {question}
            """)
            # --- END MODIFIED ---

        # Generate content
        logging.info("Generating content with LLM...")
        try:
            response = llm_model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(temperature=0.2, max_output_tokens=2048)
            )
            logging.info("LLM generation successful.")
        except Exception as llm_error:
            logging.error(f"LLM generation failed: {llm_error}", exc_info=True)
            return jsonify({"error": "Internal Server Error: Failed to generate response."}), 500

        # Extract answer
        answer_text = None
        try:
            if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                 answer_text = "".join(part.text for part in response.candidates[0].content.parts if hasattr(part, 'text'))
            elif hasattr(response, "text"): answer_text = response.text # Fallback
            if answer_text: answer_text = answer_text.strip()
            if not answer_text: # Handle empty response case
                 logging.error(f"LLM response was empty or structure invalid. Response: {response}")
                 answer_text = "Error: The AI generated an empty or invalid response."
        except Exception as extract_error:
             logging.error(f"Error extracting LLM response text: {extract_error}. Response: {response}", exc_info=True)
             answer_text = "Error: Failed to process the AI's response."

        logging.info(f"Answer extracted ({len(answer_text)} chars). Sending JSON...")
        db_count = 0
        try: db_count = doc_collection.count() if doc_collection else 0
        except: pass # Ignore count error in final response
        
        return jsonify({
            "question": question, "answer": answer_text,
            "context_found": context_found, "total_chunks_in_db": db_count
        })

    except Exception as e:
        logging.exception("!!! Unexpected error in /ask route !!!")
        return jsonify({"error": "Internal Server Error: An unexpected issue occurred."}), 500


# --- SERVER STARTUP (Runs ONCE PER WORKER when Gunicorn imports 'app') ---
# This block executes outside the __main__ check, so Gunicorn runs it.
try:
    # Only run download/init logic if NOT in build mode
    # Check sys.argv safely
    is_build_mode = len(sys.argv) > 1 and sys.argv[1] == 'build'
    
    if not is_build_mode:
        logging.info("--- Worker process starting: Initializing DB and Clients ---")
        database_ready = check_and_download_db()
        if not database_ready:
            logging.critical("Database setup failed. Worker cannot initialize.")
            sys.exit("Worker Error: Database setup failed.") # Exit worker
        
        initialize_clients() # Initialize globals for this worker
        logging.info("--- Worker initialization complete ---")
    else:
        # This case handles 'python app.py build' where we don't need server init
        logging.info("Detected build mode, skipping server initialization in this process.")

except Exception as e:
    # Catch any error during the top-level initialization block
    logging.critical(f"Fatal error during worker initialization: {e}", exc_info=True)
    # Exit the worker process if any initialization step fails
    sys.exit(f"Worker Error: Initialization failed: {e}")


# --- Main Execution Block (for local 'python app.py build' or 'python app.py') ---
if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'server'

    if mode == 'build':
        # --- LOCAL BUILD MODE ---
        logging.info("--- RUNNING IN LOCAL BUILD MODE (Main Block) ---")
        try:
            # Initialize minimal clients needed *only* for building
            logging.info(f"Initializing local DB client for build at: {DB_PATH}")
            db_client = chromadb.PersistentClient(path=DB_PATH)
            doc_collection = db_client.get_or_create_collection(name="cambodian_law")
            EMBEDDING_MODEL_NAME = "models/text-embedding-004"
            logging.info(f"Local build clients initialized. DB has {doc_collection.count()} chunks.")
        except Exception as e:
            logging.critical(f"Failed to initialize local clients for build: {e}", exc_info=True)
            sys.exit(1)
        
        load_and_chunk_documents() # Run the indexing function
        
        logging.info("--- LOCAL BUILD COMPLETE ---")
        logging.info(f"Database build finished in the '{DB_PATH}' folder.")
        logging.info("You can now compress this folder for upload:")
        logging.info(f"  tar -czvf chroma_db.tar.gz '{os.path.basename(DB_PATH)}' -C '{os.path.dirname(DB_PATH) or '.'}'")


    elif mode == 'server':
         # --- LOCAL SERVER MODE (when running `python app.py`) ---
         # Gunicorn bypasses this __main__ block entirely.
         # The initialization logic for Gunicorn workers is handled above.
         logging.warning("--- Running Flask development server locally (Use Gunicorn for production) ---")
         
         # Ensure initialization happened (it should have via the top-level block)
         if not llm_model or not doc_collection:
              logging.critical("Clients were not initialized by the top-level server block. Cannot start local server.")
              # Attempt re-initialization for local dev server case
              logging.info("Attempting initialization for local Flask server...")
              try:
                  db_ready = check_and_download_db()
                  if db_ready: initialize_clients()
                  else: raise RuntimeError("Local DB setup failed.")
              except Exception as e:
                  logging.critical(f"Initialization failed for local server: {e}")
                  sys.exit(1)

         port = int(os.getenv('PORT', 8080))
         use_debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
         logging.info(f"Starting Flask development server on http://0.0.0.0:{port} (Debug: {use_debug})")
         app.run(debug=use_debug, host='0.0.0.0', port=port, use_reloader=False if use_debug else True) # Disable reloader if debug is on

    else:
        logging.error(f"Unknown command: '{mode}'. Use 'build' or run without arguments for server mode.")
        sys.exit(1)