# AI-Powered Contract Analysis System

## 📋 Table of Contents
- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Technical Approach](#technical-approach)
- [Key Challenges & Solutions](#key-challenges--solutions)
- [LLM Prompt Engineering](#llm-prompt-engineering)
- [Setup & Usage](#setup--usage)
- [API Reference](#api-reference)

---

## Overview

This project implements an **AI-powered contract compliance analysis system** that uses **Retrieval-Augmented Generation (RAG)** and **Large Language Models (LLMs)** to automatically verify if contract clauses meet specified obligations.

### Key Features
- ✅ **Semantic Analysis**: Uses cosine similarity for confidence scoring
- ✅ **Multilingual Support**: Automatic translation to English
- ✅ **RAG-based Retrieval**: FAISS vector store for efficient clause matching
- ✅ **LLM Reasoning**: GPT-4o-mini for nuanced compliance decisions
- ✅ **Strict Yes/No Output**: Binary compliance status (no "Partial")

---

## System Architecture

```
┌─────────────────┐
│   User Input    │
│  - Obligations  │
│  - Contract     │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│         Text Extraction Layer           │
│  - PDF/DOCX/Excel/TXT support          │
│  - Language detection & translation     │
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│         Chunking & Embedding            │
│  - RecursiveCharacterTextSplitter       │
│  - Chunk size: 2000, Overlap: 200      │
│  - OpenAI text-embedding-3-small       │
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│         FAISS Vector Store              │
│  - Stores contract clause embeddings    │
│  - Enables similarity search            │
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│         RAG Query Pipeline              │
│  1. Embed obligation                    │
│  2. Retrieve top-k similar clauses      │
│  3. Calculate cosine similarity         │
│  4. Extract keywords (KeyBERT + spaCy) │
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│         LLM Analysis (GPT-4o-mini)      │
│  - Semantic compliance check            │
│  - Reason generation                    │
│  - Suggestion generation (if No)        │
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│         Final Output                    │
│  - Status: Yes/No                       │
│  - Confidence: Cosine similarity %      │
│  - Reason: LLM explanation              │
│  - Suggestion: Remediation advice       │
└─────────────────────────────────────────┘
```

---

## Technical Approach

### 1. **Confidence Scoring: Cosine Similarity Only**

**Decision**: Use pure cosine similarity between obligation and contract clause embeddings.

**Why?**
- Directly measures semantic similarity
- Transparent and explainable
- Avoids complex hybrid formulas

**Implementation**:
```python
confidence = round(best_score * 100, 1)  # best_score is cosine similarity
```

### 2. **Chunking Strategy**

**Configuration**:
- **Chunk Size**: 2000 characters
- **Overlap**: 200 characters

**Why larger chunks?**
- Complex obligations span multiple sentences
- More context improves semantic matching
- Reduces fragmentation of related clauses

### 3. **Retrieval Strategy**

**Top-k Retrieval**: Fetch 6 most similar chunks

**Why?**
- Provides multiple perspectives
- Handles cases where obligation matches across clauses
- Gives LLM sufficient context for reasoning

### 4. **LLM for Final Decision**

**Model**: GPT-4o-mini  
**Temperature**: 0.1 (low for consistency)

**Why LLM?**
- Handles nuanced language (e.g., "in lieu of", "reasonable efforts")
- Understands legal implications
- Generates human-readable explanations

---

## Key Challenges & Solutions

### Challenge 1: **Obligation 3 Incorrectly Marked as "No"**

**Scenario**:
- **Obligation**: "Vendor shall undertake all necessary modifications to remedy infringement"
- **Contract**: "Vendor will use reasonable commercial efforts to modify OR secure licenses"
- **Expected**: Yes
- **Initial Result**: No

**Root Cause**:
The LLM interpreted "in lieu of" alternatives (securing licenses) as non-compliance, not recognizing that both paths achieve the same result (non-infringement).

**Solution**:
Strengthened prompt guideline #1:

```
1. **Reasonable Efforts to Achieve Result**: If the Obligation requires a 
   specific result (e.g., "remedy infringement"), and the Contract commits 
   to "reasonable commercial efforts" to achieve that result, this is 
   ACCEPTABLE. Return "Yes". The contract may also offer alternatives 
   "in lieu of" the primary remedy (e.g., securing licenses instead of 
   fixing) as long as these alternatives achieve the same end result 
   (non-infringement, continued use).
```

**Verification**: Dry run test confirmed Obligation 3 now returns "Yes"

---

### Challenge 2: **Low Similarity Threshold Overriding LLM**

**Issue**: 
Code had a fallback check:
```python
if best_score < 0.3:
    final_status = "No"
```

This overrode the LLM's "Yes" decision when vocabulary differed (e.g., "remedy" vs "implement modifications").

**Solution**: 
Removed the threshold check to fully trust semantic analysis:
```python
# Removed strict similarity threshold to allow LLM reasoning to prevail
final_status = llm_status
```

---

### Challenge 3: **Case Sensitivity Bug**

**Issue**: 
LLM returned `"yes"` (lowercase), but validation checked for `"Yes"` (title case):
```python
if llm_status not in ["Yes", "No"]:
    llm_status = "No"  # Incorrectly defaulted to No
```

**Solution**: 
Added normalization:
```python
llm_status = parsed.get("is_present", "No").strip()

if llm_status.lower() == "yes":
    llm_status = "Yes"
elif llm_status.lower() == "no":
    llm_status = "No"
```

---

### Challenge 4: **Incorrect Suggestions**

**Issue**: 
Prompt didn't ask for suggestions, causing `null` or contradictory messages.

**Solution**: 
1. Added `suggestion` field to prompt
2. Added logic to ensure proper suggestions:
```python
if llm_status == "Yes":
    llm_suggestion = None
elif not llm_suggestion or llm_suggestion == "null":
    llm_suggestion = "Consider adding explicit language to address this obligation."
```

---

## LLM Prompt Engineering

### Final Prompt Structure

```
You are a multilingual contract compliance analyst.

Task: Determine if the 'Obligation' is fully present in the 'Relevant Clauses'.

Guidelines:
1. **Reasonable Efforts to Achieve Result**: If the Obligation requires a 
   result, and the Contract commits to "reasonable commercial efforts" to 
   achieve it, this is ACCEPTABLE. Alternatives "in lieu of" the primary 
   remedy are acceptable if they achieve the same end result.

2. **Strict Remedy Matching for Guarantees**: If the Obligation demands a 
   GUARANTEE with specific remedies, and the Contract adds a REFUND option 
   that allows termination, this undermines the guarantee. Return "No".

3. **Refund as Escape Clause**: If the Obligation requires continued use, 
   but the Contract allows refund and termination, this is NON-COMPLIANCE.

Return JSON:
{
  "is_present": "Yes" or "No",
  "reason": "short 1-2 sentence rationale",
  "suggestion": "If 'No', provide specific clause suggestion. If 'Yes', null."
}

Obligation: [...]
Relevant Clauses: [...]
```

### Prompt Evolution

| Version | Issue | Fix |
|---------|-------|-----|
| v1 | Used hybrid score (similarity + keywords + LLM boost) | Switched to cosine similarity only |
| v2 | Had "Partial" status | Enforced strict Yes/No |
| v3 | Hard-coded examples ("fix vs reimburse") | Generalized to principle-based guidelines |
| v4 | Didn't handle "in lieu of" alternatives | Strengthened guideline #1 |
| v5 | Missing suggestion field | Added suggestion with proper logic |

---

## Setup & Usage

### Installation

```bash
# Install dependencies
pip install -r backend/requirements.txt

# Set OpenAI API key
export OPENAI_API_KEY="your-key-here"
```

### Running the Backend

```bash
uvicorn backend.main:app --reload
```

### API Endpoint

**POST** `/api/analyze`

**Request**:
```json
{
  "obligations_file": "base64_encoded_file",
  "contract_file": "base64_encoded_file"
}
```

**Response**:
```json
{
  "results": [
    {
      "obligation": "Vendor must remedy infringement",
      "is_present": "Yes",
      "confidence": 87.3,
      "similarity_score": 0.873,
      "reason": "Contract commits to reasonable efforts to remedy",
      "suggestion": null,
      "supporting_clauses": ["..."]
    }
  ]
}
```

---

## API Reference

### Core Functions

#### `query_rag(vs, obligation, auto_keywords, top_k=6)`
Performs RAG-based analysis for a single obligation.

**Parameters**:
- `vs`: FAISS vector store
- `obligation`: Obligation text
- `auto_keywords`: Extracted keywords
- `top_k`: Number of chunks to retrieve

**Returns**: Dictionary with status, confidence, reason, suggestion

#### `analyze_contract(obligations_file, contract_file, session_id)`
End-to-end analysis pipeline.

**Parameters**:
- `obligations_file`: Bytes of obligations file
- `contract_file`: Bytes of contract file
- `session_id`: Unique session identifier

**Returns**: List of analysis results

---

## Technologies Used

- **LLM**: OpenAI GPT-4o-mini
- **Embeddings**: OpenAI text-embedding-3-small
- **Vector Store**: FAISS (Facebook AI Similarity Search)
- **Keyword Extraction**: KeyBERT + spaCy
- **Translation**: googletrans + OpenAI fallback
- **Backend**: FastAPI + Uvicorn
- **Frontend**: Streamlit (optional)

---

## Future Enhancements

- [ ] Support for more LLM providers (Anthropic, Gemini)
- [ ] Fine-tuned embeddings for legal domain
- [ ] Caching for repeated obligations
- [ ] Batch processing for large contracts
- [ ] Confidence calibration based on historical data

---

## License

MIT License

---

## Contributors

Developed as part of an AI-powered contract intelligence project.
