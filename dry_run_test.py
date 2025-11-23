import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Exact inputs from user
test_cases = [
    {
        "id": 1,
        "obligation": "vendor agrees to return all confidential information and data to bank in format acceptable to bank.",
        "contract": """The receiving Party shall follow the disclosing party's reasonable instructions regarding the handling and disposal and disclosing party's confidential information XYZ Inc (vendor) shall also comply with the destruction requirements in schedule AE after the Transition Services period. such instructions may include or return or destroying all confidential information (including electronic or proper copies) and providing an officer signed certification of compliance. However, return or destruction is not required for confidential information that must be retained for a) valid business purpose b) submission to government authority c) disaster recovery requirement d) technology the receiving party is permitted to retain under section 27.0. The receiving party may also retain one copy if required by applicable law ( to be used only for legal compliance). Section 16.6 does not apply where XYZ. inc needs the confidential information to operate it's systems, however, once the banks confidential informationis no longer needed for system operation, vendor must comply with disposal obligations.""",
        "expected": "No"
    },
    {
        "id": 3,
        "obligation": "Vendor shall undertake all necessary modifications the product to full remedy any infringement, without degradation of the product.",
        "contract": """Vendor shall retain the exclusive right, at its sole discretion and employing all reasonable commercial efforts, to implement such modifications, alterations, or adjustments to the affected portion of the service as may be necessary to render that portion non-infringing under applicable law. Furthermore, vendor may, in lieu of such modification, secure and maintain - at vendor sole cost and expense - all rights, licenses, consents, or permissions required to ensure that subscriber may continue to access, utilize, and benefit from the affected portion of the services without interruption. Such actions may be taken by vendor whenever it deems them reasonable necessary to mitigate or resolve any claim or threat of infringement.""",
        "expected": "Yes"
    },
    {
        "id": 4,
        "obligation": "The Licensee must guarantee that Licensed Products do not infringe patents or copyrights; if they do, the Licensee must (i) secure continued use rights, (i) replace with non-infringing products.",
        "contract": """The Licensee hereby warrants and represents that all Licensed Products shall be free from any infringement or misappropriation of third-party intellectual property rights, including but not limited to patents, copyrights, trademarks, or trade secrets. In the event any such claim or proceeding arises, the Licensee shall, at its sole cost and expense, promptly procure the necessary rights or licenses to continue lawful use of the Licensed Products, substitute them with non-infringing equivalents of comparable quality, or reimburse purchasers in full for the affected goods. Furthermore, the Licensee shall indemnify and hold harmless the Licensor from all losses, damages, and reasonable legal fees incurred in defending any such infringement action, regardless of its outcome.""",
        "expected": "No"
    }
]

prompt_template = """
You are a multilingual contract compliance analyst. Translate internally to English.
Analyze the obligation and the relevant clauses. Return valid JSON ONLY.

Task: Determine if the 'Obligation' is fully present and agreed to in the 'Relevant Clauses'.

Guidelines:
1. **Strict Remedy Matching**: If the Obligation specifies required remedies (e.g., "must fix or replace"), and the Contract adds an alternative that allows the vendor to simply "refund" or "reimburse" and walk away (terminating usage), this is a NON-COMPLIANCE. Return "No".
2. **Reasonable Efforts**: If the Obligation requires a result (e.g., "remedy infringement"), and the Contract commits to "reasonable commercial efforts" to achieve that result, this is generally ACCEPTABLE. Return "Yes".
3. **Guarantees**: If the Obligation demands a guarantee, but the Contract limits the remedy to a refund (undermining the guarantee of continued use), Return "No".

Return JSON:
{{
  "is_present": "Yes" or "No",
  "reason": "short 1-2 sentence rationale"
}}

Obligation:
{obligation}

Relevant Clauses:
{contract}
"""

print("=" * 80)
print("DRY RUN TEST - Direct LLM Call")
print("=" * 80)

for tc in test_cases:
    print(f"\n{'='*80}")
    print(f"OBLIGATION {tc['id']}: {tc['obligation'][:80]}...")
    print(f"EXPECTED: {tc['expected']}")
    print(f"{'='*80}")
    
    prompt = prompt_template.format(obligation=tc['obligation'], contract=tc['contract'])
    
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a concise contract compliance expert who replies in JSON. You only output Yes or No."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=400
        )
        
        res_text = resp.choices[0].message.content.strip()
        print(f"\nRAW LLM RESPONSE:\n{res_text}")
        
        # Parse
        if "```json" in res_text:
            res_text = res_text.split("```json")[1].split("```")[0]
        elif "```" in res_text:
            res_text = res_text.split("```")[1]
        
        parsed = json.loads(res_text)
        llm_status = parsed.get("is_present", "No").strip()
        
        # Normalize
        if llm_status.lower() == "yes":
            llm_status = "Yes"
        elif llm_status.lower() == "no":
            llm_status = "No"
        
        if llm_status not in ["Yes", "No"]:
            llm_status = "No"
        
        reason = parsed.get("reason", "")
        
        print(f"\nPARSED STATUS: {llm_status}")
        print(f"REASON: {reason}")
        
        if llm_status == tc['expected']:
            print(f"✅ PASS")
        else:
            print(f"❌ FAIL - Expected {tc['expected']}, got {llm_status}")
        
    except Exception as e:
        print(f"❌ ERROR: {e}")

print(f"\n{'='*80}")
print("DRY RUN COMPLETE")
print("=" * 80)
