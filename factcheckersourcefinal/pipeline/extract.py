import ollama
import json
import re

EXTRACT_PROMPT = """You are a fact-checking assistant. Your job is to read a news article and extract all falsifiable factual claims that could be verified against academic research.

Rules:
- Only extract specific, concrete factual claims (statistics, causal claims, scientific assertions)
- Do NOT extract opinions, predictions, or value judgements
- Do NOT extract claims about named individuals' personal actions
- Return ONLY a JSON array of strings, no other text

Example output:
["Claim one here", "Claim two here", "Claim three here"]

Article:
{article}

Return only the JSON array:"""

def extract_claims(article_text):
    print("Extracting claims from article...")

    prompt = EXTRACT_PROMPT.format(article=article_text)

    response = ollama.chat(
        
        model="mistral",
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.1}
    )

    raw = response["message"]["content"].strip()

    # try to parse the JSON array from the response
    try:
        claims = json.loads(raw)
        if isinstance(claims, list):
            return [str(c) for c in claims]
    except json.JSONDecodeError:
        pass

    # if that fails, try to find a JSON array anywhere in the response
    match = re.search(r'\[.*?\]', raw, re.DOTALL)
    if match:
        try:
            claims = json.loads(match.group())
            if isinstance(claims, list):
                return [str(c) for c in claims]
        except json.JSONDecodeError:
            pass

    # last resort: split by newlines and clean up
    print("  Warning: could not parse JSON, falling back to line splitting")
    lines = [l.strip().strip('"').strip("'").strip("-").strip() for l in raw.split("\n")]
    return [l for l in lines if len(l) > 20]

