import json

def test_parsing():
    print("--- Testing Parsing Logic ---")
    
    test_cases = [
        '{"is_present": "Yes", "reason": "Compliant"}',
        '{"is_present": "yes", "reason": "Compliant"}',
        '{"is_present": "YES", "reason": "Compliant"}',
        '{"is_present": " No ", "reason": "Not compliant"}'
    ]
    
    for json_str in test_cases:
        print(f"\nInput: {json_str}")
        parsed = json.loads(json_str)
        llm_status = parsed.get("is_present", "No")
        
        # Current Logic
        original_status = llm_status
        if llm_status not in ["Yes", "No"]:
            llm_status = "No"
            
        print(f"Parsed: '{original_status}' -> Final: '{llm_status}'")
        
        if original_status.lower() == "yes" and llm_status == "No":
            print("!!! BUG DETECTED: 'yes' became 'No' !!!")

if __name__ == "__main__":
    test_parsing()
