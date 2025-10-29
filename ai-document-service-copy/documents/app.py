import os
from flask import Flask, request, jsonify
from langchain_google_vertexai import ChatVertexAI

# --- Basic Flask App Setup ---
app = Flask(__name__)

# --- Lazy Initialization for Vertex AI Client ---

# 1. Initialize llm as None.
# The client will only be created when the first request comes in.
# This prevents the app from crashing on startup if credentials or config are wrong.
llm = None

def get_llm():
    """
    Initializes and returns the Vertex AI client.
    Uses a global variable to ensure it's only created once per server process.
    """
    global llm
    if llm is None:
        print("INFO: Initializing Vertex AI client for the first time...")
        
        # 2. Read configuration from environment variables.
        # This is safer than hardcoding values in your code.
        project_id = os.environ.get("GCP_PROJECT_ID")
        location = os.environ.get("GCP_LOCATION")
        model_name = os.environ.get("ENDPOINT_MODEL_NAME") # e.g., "gemini-1.5-pro-preview-0409"

        if not all([project_id, location, model_name]):
            raise ValueError("Missing required environment variables: GCP_PROJECT_ID, GCP_LOCATION, ENDPOINT_MODEL_NAME")

        # 3. Create the client instance.
        # This is the line that was previously causing the startup crash. Now it's
        # safely wrapped inside a function that's only called during a request.
        llm = ChatVertexAI(
            project=project_id,
            location=location,
            model_name=model_name,
        )
        print("INFO: Vertex AI client initialized successfully.")
    return llm

# --- API Endpoints ---

@app.route("/")
def health_check():
    """A simple endpoint to confirm the server is running."""
    return "Server is up and running!"

@app.route("/ask", methods=["POST"])
def ask_ai():
    """
    Endpoint to receive a question and get an answer from the AI model.
    Expects JSON input like: {"question": "Why is the sky blue?"}
    """
    data = request.get_json()
    if not data or "question" not in data:
        return jsonify({"error": "Missing 'question' in request body"}), 400

    question = data["question"]

    try:
        # 4. Get the LLM client (it will be initialized on the first call).
        client = get_llm()

        # 5. Use the client to get a response.
        response = client.invoke(question)
        
        # Assuming the response object has a 'content' attribute.
        # Adjust '.content' if your LangChain version returns a different structure.
        return jsonify({"answer": response.content})

    except Exception as e:
        # If initialization or invocation fails, return a detailed error.
        # This makes debugging much easier than a server crash.
        error_message = f"Failed to communicate with Vertex AI. Details: {str(e)}"
        print(f"ERROR: {error_message}")
        return jsonify({"error": error_message}), 500

# This part is useful for local testing but Gunicorn will handle it on Render.
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))