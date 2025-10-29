# AI Document Service
This is the backend service for my RAG AI application.

### When adding new documents
poetry run python app.py build
tar -czvf chroma_db.tar.gz ./chroma_db
-> Upload to cloudflare

To Delete old database if needed: rm -rf ./chroma_db

### Running server:
poetry run python app.py

