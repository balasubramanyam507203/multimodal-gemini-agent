# Multimodal-Gemini-Agent 🚀

A local, multimodal AI agent built using Google Gemini + MongoDB for retrieval, enabling PDF & image question-answering using a RAG (retrieval-augmented generation) pipeline.

---

## ✨ Features

- **📄 PDF Ingestion**  
  Automatically reads PDFs, splits them into text chunks, embeds them, and stores them in MongoDB.

- **🧠 Multimodal AI Reasoning**  
  Supports text + image queries. Ask questions about documents, charts, diagrams, screenshots, etc.

- **🔍 Retrieval-Augmented Generation (RAG)**  
  Uses embedding-based search to fetch relevant chunks before sending context to Gemini for accurate answers.

- **🔒 Secure Configuration**  
  API keys and database credentials stay in `.env` (not committed to GitHub).

---

## 📁 Project Structure
'''multimodal-gemini-agent/
├── config.py # Loads API keys + MongoDB connection
├── ingest.py # PDF ingestion + embedding
├── agents.py # Retrieval + multimodal agent logic
├── main.py # CLI interface for asking questions
├── data/
│ ├── docs/ # Place your PDF files here
│ └── images/ # Optional: charts/images for analysis
├── .gitignore
└── README.md'''


