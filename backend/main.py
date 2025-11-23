from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import shutil
import os
import uuid
import logging
from typing import List
from .core import analyze_contract

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Contract Intelligence API", version="1.0.0")

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for dev; restrict in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure uploads directory exists
os.makedirs("uploads", exist_ok=True)

# Ensure uploads directory exists
os.makedirs("uploads", exist_ok=True)

# Mount static files
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

@app.get("/")
async def root():
    return FileResponse("frontend/index.html")

@app.post("/api/analyze")
async def analyze(
    obligations_file: UploadFile = File(...),
    contract_file: UploadFile = File(...)
):
    session_id = str(uuid.uuid4())
    logger.info(f"Starting analysis for session {session_id}")
    
    try:
        # Ensure uploads directory exists
        os.makedirs("uploads", exist_ok=True)
        
        # Save contract file to disk for preview
        contract_path = f"uploads/{session_id}_{contract_file.filename}"
        with open(contract_path, "wb") as f:
            contract_content = await contract_file.read()
            f.write(contract_content)
            
        # Reset cursor for reading content in memory
        await contract_file.seek(0)
        
        # Read files into memory for processing
        ob_content = await obligations_file.read()
        # contract_content is already read
        
        # Run analysis
        results, full_text = analyze_contract(
            ob_content, 
            obligations_file.filename, 
            contract_content, 
            contract_file.filename, 
            session_id
        )
        
        return JSONResponse(content={
            "status": "success", 
            "results": results,
            "contract_url": f"/uploads/{session_id}_{contract_file.filename}",
            "full_text": full_text
        })
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500, 
            content={"status": "error", "message": str(e)}
        )
