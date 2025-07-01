from langchain_community.document_loaders import PDFPlumberLoader, TextLoader
from langchain_ollama import OllamaLLM
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document

def analyze_invoice_and_store(policy_path, invoice_path, employee_name):
    # Load PDF invoices
    loader = PDFPlumberLoader(invoice_path)
    docs = loader.load()
    invoice_text = "\n".join([doc.page_content for doc in docs])

    # Load policy document
    policy_loader = TextLoader(policy_path, encoding="utf-8")
    policy = policy_loader.load()

    # Initialize LLM
    llm = OllamaLLM(model="gemma:2b")

    # Prompt template
    prompt = ChatPromptTemplate.from_messages([
        ('system', "You are an HR invoice reimbursement assistant."
         "Based on the **Company Reimbursement Policy** provided you have to evaluate whether the entered **Employee Invoice** can be:"
         "Fully Reimbursed"
         "Partially Reimbursed"
         "Declined"
         "Provide detailed reason and explain your reasoning strictly as per policy."
         "**Policy:** {policy_text} "
         "**Invoice:** {invoice_text} "
         "Provide your response as:"
         "Status: <Fully Reimbursed / Partially Reimbursed / Declined>  "
         "Reason: <detailed reason>"
         "Name: <name from invoice>"
         "Date: <date from invoice>"
         )
    ])

    parser = StrOutputParser()
    chain = prompt | llm | parser

    # Run analysis
    response = chain.invoke({
        'policy_text': policy[0].page_content,
        'invoice_text': invoice_text
    })

    # Parse LLM output
    def parse_response(response_text):
        result = {
            "status": None,
            "name": None,
            "date": None,
            "reason": None
        }
        lines = response_text.strip().split("\n")
        for line in lines:
            if line.lower().startswith("status:"):
                result["status"] = line.split(":", 1)[1].strip()
            elif line.lower().startswith("name:"):
                result["name"] = line.split(":", 1)[1].strip()
            elif line.lower().startswith("date:"):
                result["date"] = line.split(":", 1)[1].strip()
            elif line.lower().startswith("reason:"):
                result["reason"] = line.split(":", 1)[1].strip()
        return result

    parsed_result = parse_response(response)

    # Prepare embedding text
    embedding_text = f"""
    Invoice:
    {invoice_text}

    Analysis:
    Status: {parsed_result['status']}
    Reason: {parsed_result['reason']}
    """

    # Metadata directly from LLM
    metadata = {
        "invoice_id": f"INV-{parsed_result['name']}-{parsed_result['date']}",
        "employee_name": parsed_result["name"],
        "date": parsed_result["date"],
        "status": parsed_result["status"],
        "reason": parsed_result["reason"]
    }

    # Load embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Connect to Chroma vector store
    vectorstore = Chroma(
        collection_name="invoice_reimbursements",
        embedding_function=embeddings,
        persist_directory="./chroma_db"
    )

    # Create Document
    doc = Document(
        page_content=embedding_text,
        metadata=metadata
    )

    # Add to Chroma
    vectorstore.add_documents([doc])

    return f"Invoice for {metadata['employee_name']} stored successfully with status: {metadata['status']}"
