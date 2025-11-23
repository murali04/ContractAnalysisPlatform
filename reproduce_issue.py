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

# Define Inputs
obligations = [
    "vendor agrees to return all confidential information and data to bank in format acceptable to bank.",
    "Vendor does not have to indemnify the Bank if the infringement is solely from Bank's unauthorized modification or use of the product.",
    "Vendor shall undertake all necessary modifications the product to full remedy any infringement, without degradation of the product.",
    "The Licensee must guarantee that Licensed Products do not infringe patents or copyrights; if they do, the Licensee must (i) secure continued use rights, (i) replace with non-infringing products."
]

contract_clauses = [
    """The receiving Party shall follow the disclosing party's reasonable instructions regarding the handling and disposal and disclosing party's confidential information XYZ Inc (vendor) shall also comply with the destruction requirements in schedule AE after the Transition Services period. such instructions may include or return or destroying all confidential information (including electronic or proper copies) and providing an officer signed certification of compliance. However, return or destruction is not required for confidential information that must be retained for a) valid business purpose b) submission to government authority c) disaster recovery requirement d) technology the receiving party is permitted to retain under section 27.0. The receiving party may also retain one copy if required by applicable law ( to be used only for legal compliance). Section 16.6 does not apply where XYZ. inc needs the confidential information to operate it's systems, however, once the banks confidential informationis no longer needed for system operation, vendor must comply with disposal obligations.""",
    
    """Vendor shall defend customer and its affiliates against any claim to the extent it alleges that a product, fix or services deliverable made available by ABC for a fee and used within the scope of the license misappropriates a trade secret or infringes a patent, copyright, trademark, or other proprietary right of a third party. ABC's obligations under this section apply solely to clams of direct infringement. For covered products, ABC's defense and payment obligations are further subject to customer copyright commitment under the product terms. Notwithstanding the foregoing, ABC shall have no obligation with respect to any claim arising solely form: a) a modification of any product, fix, or deliverable by customer, or b) any combination of such product, fix, or deliverable with a non-ABC product, unless: 1) The modification or combination is required under the applicable license 1. The modification or combination is required by ABC product documentation 2. The combination of an ABC production with a non-ABC product constitutes an essential or inventive element of the patent claim, or 3. The modification or combination is mutually agreed in writing by the parties ABC's defense obligations apply only to customers or its affiliates using the P Products, fixes or deliverables as permitted in a supplemental agreement. Further, ABC shall have no obligation where a claim is asserted against an affiliate that is no using products, fixes or deliverables.""",
    
    """Vendor shall retain the exclusive right, at its sole discretion and employing all reasonable commercial efforts, to implement such modifications, alterations, or adjustments to the affected portion of the service as may be necessary to render that portion non-infringing under applicable law. Furthermore, vendor may, in lieu of such modification, secure and maintain - at vendor sole cost and expense - all rights, licenses, consents, or permissions required to ensure that subscriber may continue to access, utilize, and benefit from the affected portion of the services without interruption. Such actions may be taken by vendor whenever it deems them reasonable necessary to mitigate or resolve any claim or threat of infringement.""",
    
    """The Licensee hereby warrants and represents that all Licensed Products shall be free from any infringement or misappropriation of third-party intellectual property rights, including but not limited to patents, copyrights, trademarks, or trade secrets. In the event any such claim or proceeding arises, the Licensee shall, at its sole cost and expense, promptly procure the necessary rights or licenses to continue lawful use of the Licensed Products, substitute them with non-infringing equivalents of comparable quality, or reimburse purchasers in full for the affected goods. Furthermore, the Licensee shall indemnify and hold harmless the Licensor from all losses, damages, and reasonable legal fees incurred in defending any such infringement action, regardless of its outcome."""
]

# Expected Results
expected = ["No", "No", "Yes", "No"]

def run_reproduction():
    print("--- Starting Reproduction ---")
    
    # 1. Prepare Contract Records
    records = [{"page": 1, "line": i+1, "text": clause} for i, clause in enumerate(contract_clauses)]
    
    # 2. Chunk and Build Vector Store
    # Note: Using a temporary path for the vector store
    vs_path = "temp_repro_vs"
    docs = chunk_text(records)
    vs = build_vector_store(docs, vs_path)
    
    # 3. Generate Keywords
    auto_keywords = generate_dynamic_keywords(obligations)
    
    # 4. Run Analysis
    print(f"{'Obligation':<10} | {'Expected':<10} | {'Actual':<10} | {'Confidence':<10} | {'Reason'}")
    print("-" * 100)
    
    for i, ob in enumerate(obligations):
        result = query_rag(vs, ob, auto_keywords)
        status = result["is_present"]
        conf = result["confidence"]
        reason = result["reason"][:50] + "..." if len(result["reason"]) > 50 else result["reason"]
        
        print(f"Ob {i+1:<7} | {expected[i]:<10} | {status:<10} | {conf:<10} | {reason}")

    # Cleanup
    import shutil
    if os.path.exists(vs_path):
        shutil.rmtree(vs_path)

if __name__ == "__main__":
    run_reproduction()
