import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Test Obligation 3 with updated prompt
obligation = "Vendor shall undertake all necessary modifications the product to full remedy any infringement, without degradation of the product."
contract = """Vendor shall retain the exclusive right, at its sole discretion and employing all reasonable commercial efforts, to implement such modifications, alterations, or adjustments to the affected portion of the service as may be necessary to render that portion non-infringing under applicable law. Furthermore, vendor may, in lieu of such modification, secure and maintain - at vendor sole cost and expense - all rights, licenses, consents, or permissions required to ensure that subscriber may continue to access, utilize, and benefit from the affected portion of the services without interruption. Such actions may be taken by vendor whenever it deems them reasonable necessary to mitigate or resolve any claim or threat of infringement."""

prompt = f"""
You are a multilingual contract compliance analyst. Translate internally to English.
Analyze the obligation and the relevant clauses. Return valid JSON ONLY.

Task: Determine if the 'Obligation' is fully present and agreed to in the 'Relevant Clauses'.

Guidelines:
1. **Reasonable Efforts to Achieve Result**: If the Obligation requires a specific result (e.g., "remedy infringement", "fix the product"), and the Contract commits to "reasonable commercial efforts" to achieve that result, this is ACCEPTABLE. Return "Yes". The contract may also offer alternatives "in lieu of" the primary remedy (e.g., securing licenses instead of fixing) as long as these alternatives achieve the same end result (non-infringement, continued use).

2. **Strict Remedy Matching for Guarantees**: If the Obligation demands a GUARANTEE with specific remedies (e.g., "must fix or replace"), and the Contract adds a REFUND/REIMBURSE option that allows termination of use (walking away), this undermines the guarantee. Return "No".

3. **Refund as Escape Clause**: If the Obligation requires continued use/access (implied by "guarantee", "must secure rights", "must replace"), but the Contract allows the vendor to simply refund money and terminate, this is NON-COMPLIANCE. Return "No".

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

if "```json" in res_text:
    res_text = res_text.split("```json")[1].split("```")[0]
elif "```" in res_text:
    res_text = res_text.split("```")[1]

parsed = json.loads(res_text)
status = parsed.get("is_present", "No").strip()

if status.lower() == "yes":
    status = "Yes"
elif status.lower() == "no":
    status = "No"

reason = parsed.get("reason", "")

print("="*80)
print("VERIFICATION TEST - Obligation 3")
print("="*80)
print(f"Expected: Yes")
print(f"Actual: {status}")
print(f"Reason: {reason}")
print("="*80)
if status == "Yes":
    print("✅ SUCCESS - Prompt fix verified!")
else:
    print("❌ FAILED - Prompt still needs adjustment")
print("="*80)
