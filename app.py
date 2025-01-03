import openai
from langchain_community.document_loaders import WikipediaLoader
from langchain_google_community import GoogleSearchAPIWrapper
import json
import os
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify
import logging
from pydantic import BaseModel
from diff_match_patch import diff_match_patch

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

client = openai.OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)
model = "gpt-4o-mini"

app = Flask(__name__)

class Claim(BaseModel):
    original_text: str
    claim_text: str

class Claims(BaseModel):
    list: list[Claim]

# Functions for core steps
def chunk_text(input_text):
    """Splits text into paragraphs or logical chunks."""
    logging.debug("Chunking text")
    return input_text.split("\n\n")  # Simple split by double newline for paragraphs


def extract_claims(paragraph):
    """Extracts factual claims from a paragraph using LLM."""
    logging.debug(f"Extracting claims from paragraph: {paragraph}")
    prompt = f"Extract factual claims from the following text:\n\n{paragraph}\n\nOutput each claim as a JSON object with 'claim_text' and 'original_text' fields. The 'original_text' should match exactly the part of the input it was derived from."
    response = client.beta.chat.completions.parse(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model=model,
        max_tokens=500,
        response_format=Claims
    )
    claims = response.choices[0].message.parsed
    if not claims:
        logging.debug("No claims extracted")
        return []

    logging.debug(f"Extracted claims: {claims}")
    return claims.list


def retrieve_evidence(claim):
    """Retrieve evidence for a claim from multiple sources."""
    logging.debug(f"Retrieving evidence for claim: {claim}")
    # Wikipedia retrieval
    # wiki_results = WikipediaLoader(query=claim, load_max_docs=3).load()

    # Google Search retrieval
    google_results = GoogleSearchAPIWrapper(k=3).results(query=claim, num_results=3)
    logging.debug(f"Retrieved Google results: {google_results}")

    return {
        # "wikipedia": wiki_results,
        "google": [result["snippet"] for result in google_results]
    }


def classify_evidence(claim, evidence):
    """Classify evidence snippets as Supporting, Contradicting, or Neutral."""
    logging.debug(f"Classifying evidence for claim: {claim}")
    results = []
    for source, snippets in evidence.items():
        for snippet in snippets:
            prompt = f"Claim: '''{claim}'''\nEvidence: '''{snippet}'''\n\nDoes the evidence support, contradict, or is neutral to the claim? Result must contain 1 of 3 options: 'Supporting', 'Contradicting', 'Neutral'."
            response = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model=model,
                max_tokens=100
            )
            classification = response.choices[0].message.content.strip()
            logging.debug(f"Classified evidence: {snippet} as {classification}")
            results.append({
                "source": source,
                "snippet": snippet,
                "classification": classification
            })
    return results


def aggregate_results(classifications):
    """Aggregate classifications to determine chunk-level validity."""
    logging.debug("Aggregating classification results")
    supporting = sum(1 for c in classifications if c["classification"] == "Supporting")
    contradicting = sum(1 for c in classifications if c["classification"] == "Contradicting")
    neutral = sum(1 for c in classifications if c["classification"] == "Neutral")
    aggregate = {
        "supporting": supporting,
        "contradicting": contradicting,
        "neutral": neutral,
        "flagged": contradicting > supporting
    }
    logging.debug(f"Aggregated results: {aggregate}")
    return aggregate


def generate_revision(paragraph, classifications):
    """Propose a revised paragraph if necessary."""
    logging.debug(f"Generating revision for paragraph: {paragraph}")
    supporting_snippets = [c["snippet"] for c in classifications if c["classification"] == "Supporting"]
    contradicting_snippets = [c["snippet"] for c in classifications if c["classification"] == "Contradicting"]

    if not contradicting_snippets:
        return None, None

    prompt = (
        f"Original Paragraph:\n{paragraph}\n\n"
        f"Supporting Evidence:\n{supporting_snippets}\n\n"
        f"Contradicting Evidence:\n{contradicting_snippets}\n\n"
        f"Revise the paragraph to address the contradictions and ensure factual accuracy."
    )
    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model=model,
        max_tokens=500
    )
    revised_paragraph = response.choices[0].message.content.strip()
    logging.debug(f"Generated revised paragraph: {revised_paragraph}")

    # Compute differences
    dmp = diff_match_patch()
    diffs = dmp.diff_main(paragraph, revised_paragraph)
    dmp.diff_cleanupSemantic(diffs)
    diff_html = dmp.diff_prettyHtml(diffs)

    return revised_paragraph, diff_html


# Main function
def verify_text(input_text):
    """Main workflow to verify and revise input text."""
    logging.debug("Starting text verification")
    paragraphs = chunk_text(input_text)
    results = []

    for i, paragraph in enumerate(paragraphs):
        claims = extract_claims(paragraph)
        claims_results = []
        for claim_obj in claims:
            claim = claim_obj.claim_text
            original_text = claim_obj.original_text
            evidence = retrieve_evidence(claim)
            classified_evidence = classify_evidence(claim, evidence)
            claim_aggregate = aggregate_results(classified_evidence)
            claims_results.append({
                "claim": claim,
                "original_text": original_text,
                "classifications": classified_evidence,
                "aggregate": claim_aggregate
            })

        flagged = any(c["aggregate"]["flagged"] for c in claims_results)
        revised_paragraph, paragraph_diff = generate_revision(paragraph, [classification for c in claims_results for classification in c["classifications"]]) if flagged else None

        results.append({
            "paragraph": paragraph,
            "claims_results": claims_results,
            "revised_paragraph": revised_paragraph,
            "paragraph_diff": paragraph_diff
        })

    logging.debug("Completed text verification")
    return results

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/verify', methods=['POST'])
def verify():
    input_text = request.json.get('text')
    if not input_text:
        logging.error("No input text provided")
        return jsonify({"error": "No input text provided."}), 400

    logging.debug("Received text for verification")
    verification_results = verify_text(input_text)

    text = input_text
    paragraphs = []

    for result in verification_results:
        for claim_result in result['claims_results']:
            highlights = []
            if claim_result['aggregate']['flagged']:
                highlights.append({
                    "claim": claim_result['claim'],
                    "original_text": claim_result['original_text'],
                    "classifications": claim_result['classifications']
                })

        paragraphs.append({
            "paragraph": result['paragraph'],
            "paragraph_diff": result['paragraph_diff'],
            "revised_paragraph": result['revised_paragraph'],
            "highlights": highlights
        })

    logging.debug("Returning verification results")
    return jsonify({
        "text": text,
        "paragraphs": paragraphs
    })

# Example Usage
if __name__ == "__main__":
    # input_text = """In 2020, the global market cap was $50 trillion. The population of the Earth is 8 billion people. Einstein was born in 1905."""
    # output = verify_text(input_text)
    # print(json.dumps(output, indent=4))
    app.run(debug=True, port=9000)