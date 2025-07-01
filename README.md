# Invoice Reimbursement System

## Project Overview

The Invoice Reimbursement System is an AI-powered platform designed to automate the analysis and management of employee invoice reimbursements. It leverages advanced language models and vector search to evaluate invoices against company policies, provide detailed reasoning, and enable natural language querying of stored invoice data.

## Installation Instructions

1. **Clone the Repository**

   ```sh
   git clone <your-repo-url>
   cd <project-directory>
   ```

2. **Set Up Python Environment**

   - Ensure you have Python 3.8+ installed.
   - (Recommended) Create a virtual environment:
     ```sh
     python -m venv venv
     source venv/bin/activate  # On Windows: venv\Scripts\activate
     ```

3. **Install Dependencies**

   ```sh
   pip install -r requirements.txt
   ```

4. **Start the FastAPI Server**
   ```sh
   uvicorn main:app --reload
   ```

## Usage Guide

### 1. Analyze Invoice Endpoint

- **POST** `/analyze_invoice/`
- **Parameters:**
  - `policy_pdf`: UploadFile (the company policy document in PDF format)
  - `invoices_zip`: UploadFile (a ZIP file containing invoice PDFs)
  - `employee_name`: str (name of the employee)
- **Response:**
  - JSON message indicating the result of the analysis and storage.

### 2. Query Chatbot Endpoint

- **GET** `/query_chatbot/`
- **Parameters:**
  - `query`: str (your natural language query)
  - `employee_name`: str (employee name to filter, optional)
  - `date`: str (date to filter, optional)
  - `status`: str (status to filter, optional)
- **Response:**
  - JSON with the chatbot's response based on stored invoice analyses.

## Technical Details

- **Libraries Used:**

  - `FastAPI`: For building the REST API.
  - `langchain`: For prompt management, LLM integration, and document processing.
  - `langchain_ollama`: For connecting to local LLMs (e.g., Gemma 2B).
  - `langchain_huggingface`: For embedding generation using models like `all-MiniLM-L6-v2`.
  - `langchain_chroma`: For vector store management using ChromaDB.
  - `PDFPlumberLoader`, `TextLoader`: For extracting text from PDFs and text files.

- **LLM & Embedding Model Choices:**

  - **LLM:** Gemma 2B via Ollama for invoice analysis and chatbot responses.
  - **Embeddings:** all-MiniLM-L6-v2 for semantic search and vector storage.

- **Vector Store Integration:**

  - ChromaDB is used to persist and search invoice analyses as vector embeddings, enabling efficient retrieval for chatbot queries.

- **Overall Architecture:**
  - Invoices and policies are processed and analyzed by the LLM.
  - Results are parsed, embedded, and stored in ChromaDB with relevant metadata.
  - The chatbot endpoint retrieves and summarizes relevant invoices using semantic search and LLM reasoning.

## Prompt Design

- Prompts for invoice analysis are crafted to:
  - Clearly instruct the LLM to evaluate invoices strictly according to policy.
  - Request structured output (status, reason, name, date) for easy parsing.
- Chatbot prompts:
  - Provide the LLM with relevant invoice analyses and the user's query.
  - Instruct the LLM to answer in clear markdown, referencing retrieved invoices.

## Challenges & Solutions

- **Challenge:** Ensuring consistent, structured LLM output for reliable parsing.
  - **Solution:** Carefully designed prompts and robust parsing logic.
- **Challenge:** Filtering and searching invoices with multiple metadata fields in ChromaDB.
  - **Solution:** Used ChromaDB's `$and` operator for multi-field filtering.
- **Challenge:** Handling file encodings and PDF extraction edge cases.
  - **Solution:** Explicit encoding specification and use of robust PDF/text loaders.

---

For further questions or contributions, please open an issue or pull request.
