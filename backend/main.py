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
    """
    Main analysis endpoint with enhanced features (caching + batch processing).
    Uses text-embedding-3-small by default for performance.
    """
    from .core_enhanced import analyze_contract_enhanced
    
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
        
        # Run enhanced analysis (with caching and batch processing)
        results, full_text, cache_stats = analyze_contract_enhanced(
            ob_content, 
            obligations_file.filename, 
            contract_content, 
            contract_file.filename, 
            session_id,
            use_batch=True  # Enable batch processing by default
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

@app.post("/api/analyze/enhanced")
async def analyze_enhanced(
    obligations_file: UploadFile = File(...),
    contract_file: UploadFile = File(...),
    use_batch: bool = True
):
    """
    Enhanced analysis endpoint with caching and batch processing.
    
    Args:
        obligations_file: Obligations file (Excel/CSV)
        contract_file: Contract file (PDF/DOCX/TXT/Excel)
        use_batch: Enable batch processing for parallel analysis
    """
    from .core_enhanced import analyze_contract_enhanced
    
    session_id = str(uuid.uuid4())
    logger.info(f"Starting enhanced analysis for session {session_id} (batch={use_batch})")
    
    try:
        # Ensure uploads directory exists
        os.makedirs("uploads", exist_ok=True)
        
        # Save contract file to disk for preview
        contract_path = f"uploads/{session_id}_{contract_file.filename}"
        with open(contract_path, "wb") as f:
            contract_content = await contract_file.read()
            f.write(contract_content)
            
        # Reset cursor
        await contract_file.seek(0)
        
        # Read files into memory
        ob_content = await obligations_file.read()
        
        # Run enhanced analysis
        results, full_text, cache_stats = analyze_contract_enhanced(
            ob_content, 
            obligations_file.filename, 
            contract_content, 
            contract_file.filename, 
            session_id,
            use_batch=use_batch
        )
        
        return JSONResponse(content={
            "status": "success", 
            "results": results,
            "contract_url": f"/uploads/{session_id}_{contract_file.filename}",
            "full_text": full_text,
            "cache_stats": cache_stats,
            "batch_processing_used": use_batch
        })
        
    except Exception as e:
        logger.error(f"Enhanced analysis failed: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500, 
            content={"status": "error", "message": str(e)}
        )

@app.get("/api/cache/stats")
async def get_cache_stats():
    """Get cache statistics."""
    from .cache import get_cache
    
    cache = get_cache()
    stats = cache.get_stats()
    
    return JSONResponse(content={
        "status": "success",
        "cache_stats": stats
    })

@app.post("/api/cache/clear")
async def clear_cache():
    """Clear all cached results."""
    from .cache import get_cache
    
    cache = get_cache()
    cache.clear()
    
    return JSONResponse(content={
        "status": "success",
        "message": "Cache cleared successfully"
    })
