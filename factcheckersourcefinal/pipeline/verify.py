import ollama
import json
import re
from pipeline.retrieve import retrieve, format_context
from pipeline.config import ADAPTER_PATH
from pathlib import Path

VERIFY_PROMPT = """You are an academic fact-checker. You will be given a claim and a set of extracts from peer-reviewed academic papers. Your job is to assess whether the claim is supported, contradicted, or unverifiable based on the provided evidence.

Claim: {claim}

Academic Evidence:
{context}

Respond with ONLY a JSON object in exactly this format:
{{
    "verdict": "SUPPORTED" or "CONTRADICTED" or "UNVERIFIABLE",
    "confidence": a number between 0.0 and 1.0,
    "explanation": "2-3 sentence explanation citing the sources",
    "supporting_sources": [list of source numbers that support the claim, e.g. [1, 2]],
    "contradicting_sources": [list of source numbers that contradict the claim, e.g. [3]]
}}

Rules:
- SUPPORTED: ONLY if the evidence directly and explicitly addresses the specific claim
- CONTRADICTED: the evidence directly and explicitly opposes the claim
- UNVERIFIABLE: use this if the papers are about a related but different topic
- UNVERIFIABLE: use this if the papers discuss the general area but not the specific claim
- UNVERIFIABLE: use this if you are stretching to connect the evidence to the claim
- Never use indirect or tangential evidence to reach SUPPORTED
- The bar for SUPPORTED is high — the paper must directly address the claim
- When in doubt, always return UNVERIFIABLE
Return only the JSON object:"""

def query_ollama(prompt, model="mistral"):
    print(f"  [MODEL] Using: {model}")
    if model == "academicheck":
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.1},
            format="json"
        )
    else:
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.1}
        )
    return response["message"]["content"].strip()

def parse_verdict(raw):
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    match = re.search(r'\{.*?\}', raw, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    return None

def verify_claim(claim, top_k=5, model="mistral"):
    retrieved = retrieve(claim, top_k=top_k)
    context = format_context(retrieved)
    prompt = VERIFY_PROMPT.format(claim=claim, context=context)

    raw = query_ollama(prompt, model=model)
    parsed = parse_verdict(raw)

    result = {
        "verdict": "UNVERIFIABLE",
        "confidence": 0.5,
        "explanation": "Verdict based on retrieved academic evidence.",
        "supporting_sources": [],
        "contradicting_sources": []
    }

    if parsed:
        result = parsed

    result["claim"] = claim
    result["sources"] = retrieved

    if not result.get("explanation"):
        result["explanation"] = "Verdict based on retrieved academic evidence."
    if result.get("confidence") is None:
        result["confidence"] = 0.5

    return result

def verify_article(article_text, model="mistral"):
    from pipeline.extract import extract_claims
    claims = extract_claims(article_text)

    print(f"\nVerifying {len(claims)} claims using {model}...\n")
    results = []
    for i, claim in enumerate(claims, 1):
        print(f"  [{i}/{len(claims)}] {claim[:70]}...")
        result = verify_claim(claim, model=model)
        results.append(result)
        print(f"  Verdict: {result.get('verdict')} (confidence: {result.get('confidence')})\n")

    return results

def print_report(results):
    print("\n" + "="*60)
    print("FACT-CHECK REPORT")
    print("="*60)

    for i, r in enumerate(results, 1):
        verdict = r.get("verdict", "UNKNOWN")
        confidence = r.get("confidence", 0)
        explanation = r.get("explanation", "")
        claim = r.get("claim", "")
        sources = r.get("sources", [])

        verdict_symbol = {"SUPPORTED": "✅", "CONTRADICTED": "❌", "UNVERIFIABLE": "⚠️"}.get(verdict, "❓")

        print(f"\n{i}. {claim}")
        print(f"   {verdict_symbol} {verdict} (confidence: {confidence})")
        print(f"   {explanation}")

        if sources:
            print(f"   Top source: {sources[0]['title']} ({sources[0]['year']})")

    supported = sum(1 for r in results if r.get("verdict") == "SUPPORTED")
    contradicted = sum(1 for r in results if r.get("verdict") == "CONTRADICTED")
    unverifiable = sum(1 for r in results if r.get("verdict") == "UNVERIFIABLE")

    print(f"\n{'='*60}")
    print(f"SUMMARY: {supported} supported | {contradicted} contradicted | {unverifiable} unverifiable")
    print("="*60)