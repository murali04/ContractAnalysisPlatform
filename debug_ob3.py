import os
import sys
import logging
from dotenv import load_dotenv

# Add backend to path
sys.path.append(os.path.join(os.getcwd(), 'backend'))

from backend.core import build_vector_store, query_rag, chunk_text, generate_dynamic_keywords, embedder

# Setup logging
logging.basicConfig(level=logging.INFO)
load_dotenv()

# Exact Text from User Request
ob3 = "Vendor shall undertake all necessary modifications the product to full remedy any infringement, without degradation of the product."
contract3 = "Vendor shall retain the exclusive right, at its sole discretion and employing all reasonable commercial efforts, to implement such modifications, alterations, or adjustments to the affected portion of the service as may be necessary to render that portion non-infringing under applicable law. Furthermore, vendor may, in lieu of such modification, secure and maintain - at vendor sole cost and expense - all rights, licenses, consents, or permissions required to ensure that subscriber may continue to access, utilize, and benefit from the affected portion of the services without interruption. Such actions may be taken by vendor whenever it deems them reasonable necessary to mitigate or resolve any claim or threat of infringement."

def debug_ob3():
    print("--- Debugging Obligation 3 ---")
    
    # 1. Prepare Contract Records
    records = [{"page": 1, "line": 1, "text": contract3}]
    
    # 2. Chunk and Build Vector Store
    vs_path = "temp_debug_vs"
    docs = chunk_text(records)
    vs = build_vector_store(docs, vs_path)
    
    # 3. Generate Keywords
    auto_keywords = generate_dynamic_keywords([ob3])
    
    # 4. Run Analysis
    result = query_rag(vs, ob3, auto_keywords)
    
    print(f"\nObligation: {ob3}")
    print(f"Status: {result['is_present']}")
    print(f"Confidence: {result['confidence']}")
    print(f"Similarity Score: {result['similarity_score']}")
    print(f"Reason: {result['reason']}")
    print(f"Suggestion: {result.get('suggestion')}")
    
    # Cleanup
    import shutil
    if os.path.exists(vs_path):
        shutil.rmtree(vs_path)

if __name__ == "__main__":
    debug_ob3()
