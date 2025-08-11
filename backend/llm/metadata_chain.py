# backend/llm/metadata_chain.py

import json
import os
import re
from typing import Any, Dict, Final, Optional

from langchain_core.prompts import PromptTemplate

# Primary: Langfuse OpenAI drop-in -> Together (preferred for robust tracing)
try:
    from backend.graphs.monitor import get_traced_openai_together_client
except Exception:
    get_traced_openai_together_client = None  # type: ignore

# Fallback: LangChain Together wrapper
try:
    from langchain_together import Together
except Exception:
    Together = None  # type: ignore

_PROMPT: Final = """
You are an expert at extracting educational module/course metadata from academic documents. 

INSTRUCTIONS:
1. Carefully read the entire document text
2. Extract the 8 metadata fields defined below with maximum detail
3. Make reasonable inferences when information is implicit
4. Return ONLY a single JSON object with no additional text
5. Use null only when absolutely no information can be determined

FIELD DEFINITIONS AND EXTRACTION STRATEGIES:

"Module Name": The official title/name of the course, module, or program
- Look for: Headers, titles, "Module Handbook", course names in headers
- Examples: "EDMAPd Development of Academic Practice", "Basic Radiotherapy"

"ECTS Credits": European Credit Transfer System credits - CRITICAL TO FIND!
- Look for: "ECTS", "CP", "credit points", "credits", "20 credit", numbers + "credit"
- Search thoroughly: module descriptions, credit allocations, program specifications
- Examples: "20", "5", "120", "60 credits"
- EDMAPd is typically 20 credits - use this if you see EDMAPd mentioned
- Return as string

"Instructor Name(s)": Teaching staff, lecturers, convenors, directors
- Look for: "Module Convenor", "Programme Director", "Dr", "Prof", email addresses
- Include ALL staff mentioned in teaching roles
- Return as array: ["Dr Tony Churchill", "Angela Buckingham", "Dr Jo Cordy"]

"Course Duration": PROVIDE MAXIMUM DETAIL about time span, schedule, format
- Look for and combine ALL temporal information:
  * Semester length: "4 semesters", "1 year", "2 years"  
  * Teaching schedule: "intensive blocks", "face-to-face", "online sessions"
  * Specific dates: "June 16th", "September 17th", exact dates mentioned
  * Session format: "two taught days", "four essential dates", "workshops"
  * Timeline: "across two academic years", "over 3 months"
- Create detailed description combining all temporal elements
- Examples: 
  * "4 teaching days across 9 months: Day 1 (June 16, face-to-face), Day 2 (June 17, online), Day 3 (September 17, face-to-face), Day 4 (September 18, online), spanning two academic years"
  * "4 semesters full-time with practical sessions and seminars"
  * "1 semester with weekly lectures, tutorials, and final intensive week"

"Prerequisites": Required prior knowledge, modules, qualifications
- Look for: "Prerequisites", "Requirements", "builds on", "second of two modules"
- Make inferences: If "second module" mentioned, infer first module as prerequisite
- Examples: "EDMAPi (Introduction to Academic Practice)", "Basic Radiotherapy", "degree in Physics"

"Language of Instruction": Teaching language
- Look for: "Language of instruction", university context, document language
- Make inferences from institutional context (UK = English, etc.)
- Examples: "English", "German", "English and German"

"Course Objective": Learning outcomes, aims - provide comprehensive summary
- Look for: "Learning objectives", "Aims", "students will be able to"
- Summarize ALL key objectives mentioned
- Examples: "Develop academic teaching practice, provide evidence for PSF Descriptor 2, design inclusive curricula, demonstrate effective teaching methods"

"Assessment Methods": How students are evaluated - list ALL methods found
- Look for: "Assessment", "Examination", "graded", various assessment types
- Return as array with ALL methods: ["individual reflective report", "Professional Dialogue", "peer observation", "written examination"]

DOCUMENT TEXT:
{text}

Return only the JSON object with maximum detail:
"""

PROMPT_TMPL = PromptTemplate.from_template(_PROMPT)

def _call_llm(model: str, prompt: str) -> str:
    """
    Prefer Langfuse OpenAI drop-in client configured for Together.ai for best observability.
    Fallback to langchain_together if the drop-in is unavailable.
    """
    if not os.getenv("TOGETHER_API_KEY"):
        raise EnvironmentError("TOGETHER_API_KEY missing")

    # Preferred: Langfuse OpenAI drop-in (Together)
    if get_traced_openai_together_client is not None:
        client = get_traced_openai_together_client()
        if client is not None:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an expert at extracting detailed educational metadata. Always provide maximum detail and make reasonable inferences. Respond with valid JSON only."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,  # Higher for better detailed extraction
                top_p=0.95,
                max_tokens=4096,  # Significantly increased for detailed responses
            )
            return resp.choices[0].message.content or ""

    # Fallback path via langchain_together
    if Together is None:
        raise RuntimeError("Neither Langfuse OpenAI drop-in nor langchain_together is available")
    llm = Together(model=model, max_tokens=2048, temperature=0.3, top_p=0.95)
    return llm.invoke(prompt)

def _extract_json(raw: str) -> Dict[str, Any]:
    # Try to find JSON in the response
    m = re.search(r'\{[\s\S]*\}', raw)
    if not m:
        return _get_empty_metadata()
    
    try:
        result = json.loads(m.group(0))
        # Ensure all required fields exist
        template = _get_empty_metadata()
        for key in template:
            if key not in result:
                result[key] = None
        return result
    except json.JSONDecodeError:
        return _get_empty_metadata()

def _get_empty_metadata() -> Dict[str, Any]:
    return {
        "Module Name": None,
        "ECTS Credits": None,
        "Instructor Name(s)": None,
        "Course Duration": None,
        "Prerequisites": None,
        "Language of Instruction": None,
        "Course Objective": None,
        "Assessment Methods": None,
    }

class _MetaChain:
    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # Use much more text for comprehensive extraction - 12000 chars
        # Include beginning, multiple middle sections, and end
        full_text = inputs["text"]
        
        if len(full_text) <= 12000:
            txt = full_text
        else:
            # Beginning (first 5000 chars for headers, intro, module info)
            beginning = full_text[:5000]
            
            # First middle section (for detailed module descriptions)
            mid1_start = len(full_text) // 4
            mid1 = full_text[mid1_start:mid1_start + 3000]
            
            # Second middle section (for roadmap, timeline details)
            mid2_start = len(full_text) // 2
            mid2 = full_text[mid2_start:mid2_start + 2000]
            
            # End section (for assessment details, appendices)
            end = full_text[-2000:]
            
            txt = f"{beginning}\n\n[SECTION 2]\n{mid1}\n\n[SECTION 3]\n{mid2}\n\n[FINAL SECTION]\n{end}"
        
        prompt = PROMPT_TMPL.format(text=txt)
        raw = _call_llm(inputs.get("llm_model", ""), prompt)
        return _extract_json(raw)

metadata_chain: Final = _MetaChain()
