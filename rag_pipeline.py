from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# Load embedding model (must be same as insertion script)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Load persisted Chroma DB
vectorstore = Chroma(
    collection_name="invoice_reimbursements",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)

# Load Ollama LLM
llm = OllamaLLM(model="gemma:2b")


def rag_invoice_query(user_query, employee_name=None, date=None, status=None):
    # Perform similarity search without filters
    results = vectorstore.similarity_search(
        query=user_query,
        k=5
    )

    if not results:
        return "No matching invoices found."

    # Format retrieved docs for prompt
    context_text = "\n\n".join([
        f"Invoice:\n{doc.page_content}\n\nMetadata: {doc.metadata}"
        for doc in results
    ])

    # Build prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an AI assistant for the HR department. Use the retrieved invoice analyses to answer the user's query in clear markdown format."),
        ("user", "Query: {query}\n\nRelevant Invoices:\n{docs}")
    ])

    parser = StrOutputParser()
    chain = prompt | llm | parser

    # Get LLM response
    response = chain.invoke({"query": user_query, "docs": context_text})

    return response


result = rag_invoice_query(
    user_query="List all invoices fully reimbursed for Sushma",
    employee_name="Sushma",
    status="Fully Reimbursed"
)

print(result)
