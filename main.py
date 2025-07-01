from fastapi import FastAPI, UploadFile, File, Form
from invoice_analyzer import analyze_invoice_and_store
from rag_pipeline import rag_invoice_query

app = FastAPI()

@app.post("/analyze_invoice/")
async def analyze_invoice(
    policy_pdf: UploadFile = File(...),
    invoices_zip: UploadFile = File(...),
    employee_name: str = Form(...)
):
    result = analyze_invoice_and_store(policy_pdf.file, invoices_zip.file, employee_name)
    return {"message": result}

@app.get("/query_chatbot/")
async def query_chatbot(
    query: str,
    employee_name: str,
    date: str,
    status: str
):
    result = rag_invoice_query(
        user_query=query,
        employee_name=employee_name,
        date=date,
        status=status
    )
    return {"response": result}
