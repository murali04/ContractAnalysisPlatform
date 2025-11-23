"""
Enhanced core functions with caching, batch processing, and configurable embeddings.
This module extends core.py without modifying it.
"""
import os
import logging
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

# Import from core
from backend.core import (
    query_rag, generate_dynamic_keywords, chunk_text, 
    build_vector_store, translate_to_english
)
from backend.cache import get_cache, hash_contract

load_dotenv()
logger = logging.getLogger(__name__)

# Configuration
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
USE_CACHE = os.getenv("USE_CACHE", "true").lower() == "true"
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "5"))

# Enhanced embedder (can use text-embedding-3-large for legal domain)
enhanced_embedder = None

def get_enhanced_embedder():
    """Get embedder with configurable model."""
    global enhanced_embedder
    if enhanced_embedder is None:
        enhanced_embedder = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        logger.info(f"Initialized embedder with model: {EMBEDDING_MODEL}")
    return enhanced_embedder

def query_rag_with_cache(vs, obligation: str, auto_keywords: Dict, contract_hash: str, top_k: int = 6) -> Dict[str, Any]:
    """
    Query RAG with caching support.
    
    Args:
        vs: Vector store
        obligation: Obligation text
        auto_keywords: Keywords dictionary
        contract_hash: Hash of contract content
        top_k: Number of chunks to retrieve
        
    Returns:
        Analysis result
    """
    cache = get_cache()
    
    # Check cache if enabled
    if USE_CACHE:
        cached_result = cache.get(obligation, contract_hash)
        if cached_result:
            return cached_result
    
    # Perform analysis
    result = query_rag(vs, obligation, auto_keywords, top_k)
    
    # Store in cache if enabled
    if USE_CACHE:
        cache.set(obligation, contract_hash, result)
    
    return result

def batch_analyze_obligations(
    vs, 
    obligations: List[str], 
    auto_keywords: Dict, 
    contract_hash: str,
    top_k: int = 6,
    max_workers: int = None
) -> List[Dict[str, Any]]:
    """
    Analyze multiple obligations in parallel.
    
    Args:
        vs: Vector store
        obligations: List of obligation texts
        auto_keywords: Keywords dictionary
        contract_hash: Hash of contract content
        top_k: Number of chunks to retrieve
        max_workers: Number of parallel workers (default: BATCH_SIZE)
        
    Returns:
        List of analysis results in same order as obligations
    """
    if max_workers is None:
        max_workers = BATCH_SIZE
    
    results = [None] * len(obligations)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_index = {
            executor.submit(
                query_rag_with_cache, 
                vs, 
                ob, 
                auto_keywords, 
                contract_hash, 
                top_k
            ): i
            for i, ob in enumerate(obligations)
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            try:
                result = future.result()
                results[index] = result
                logger.info(f"Completed analysis for obligation {index + 1}/{len(obligations)}")
            except Exception as e:
                logger.error(f"Error analyzing obligation {index + 1}: {e}")
                results[index] = {
                    "obligation": obligations[index],
                    "is_present": "No",
                    "reason": f"Error during analysis: {str(e)}",
                    "confidence": 0.0,
                    "similarity_score": 0.0,
                    "keyword_hits": [],
                    "page": None,
                    "line": None,
                    "supporting_clauses": [],
                    "suggestion": "Unable to analyze due to error."
                }
    
    return results

def analyze_contract_enhanced(
    obligations_file_bytes, 
    obligations_filename, 
    contract_file_bytes, 
    contract_filename, 
    session_id,
    use_batch: bool = True
):
    """
    Enhanced contract analysis with caching and batch processing.
    
    Args:
        obligations_file_bytes: Bytes of obligations file
        obligations_filename: Name of obligations file
        contract_file_bytes: Bytes of contract file
        contract_filename: Name of contract file
        session_id: Session identifier
        use_batch: Whether to use batch processing
        
    Returns:
        Tuple of (results, full_text, cache_stats)
    """
    import pandas as pd
    import io
    from backend.core import (
        extract_text_from_pdf, extract_text_from_docx,
        extract_text_from_excel, extract_text_from_txt,
        get_user_vector_path, detect_language
    )
    
    # 1. Load Obligations
    if obligations_filename.endswith(".csv"):
        df_ob = pd.read_csv(io.BytesIO(obligations_file_bytes)).dropna(how="all")
    else:
        df_ob = pd.read_excel(io.BytesIO(obligations_file_bytes)).dropna(how="all")
    
    df_ob.columns = [str(c).strip() for c in df_ob.columns]
    df_ob["Language"] = df_ob.iloc[:, 0].astype(str).apply(lambda x: detect_language(x))
    df_ob["Obligation_English"] = df_ob.iloc[:, 0].astype(str).apply(lambda x: translate_to_english(x).strip())
    df_ob = df_ob[df_ob["Obligation_English"].str.strip() != ""].reset_index(drop=True)
    obligations = df_ob["Obligation_English"].tolist()
    
    # 2. Extract Contract Text
    if contract_filename.endswith(".pdf"):
        records = extract_text_from_pdf(contract_file_bytes)
    elif contract_filename.endswith(".docx"):
        records = extract_text_from_docx(contract_file_bytes)
    elif contract_filename.endswith(".xlsx"):
        records = extract_text_from_excel(contract_file_bytes)
    else:
        records = extract_text_from_txt(contract_file_bytes)
    
    # Translate contract text
    for rec in records:
        rec["text"] = translate_to_english(rec["text"]).strip()
    
    # 3. Build Vector Store
    docs = chunk_text(records)
    vector_path = get_user_vector_path(session_id)
    vs = build_vector_store(docs, vector_path)
    
    # 4. Generate Keywords
    auto_keywords = generate_dynamic_keywords(obligations)
    
    # 5. Generate contract hash for caching
    full_text = "\\n\\n".join([r["text"] for r in records])
    contract_hash_val = hash_contract(full_text)
    
    # 6. Run Analysis (batch or sequential)
    if use_batch and len(obligations) > 1:
        logger.info(f"Using batch processing for {len(obligations)} obligations")
        results = batch_analyze_obligations(vs, obligations, auto_keywords, contract_hash_val)
    else:
        logger.info(f"Using sequential processing for {len(obligations)} obligations")
        results = []
        for ob in obligations:
            results.append(query_rag_with_cache(vs, ob, auto_keywords, contract_hash_val))
    
    # 7. Get cache stats
    cache_stats = get_cache().get_stats() if USE_CACHE else None
    
    return results, full_text, cache_stats
