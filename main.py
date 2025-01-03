import openai
from langchain_community.document_loaders import WikipediaLoader
from langchain_google_community import GoogleSearchAPIWrapper
import json
import os
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify

load_dotenv()

client = openai.OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)
model = "gpt-4o-mini"

app = Flask(__name__)

# Functions for core steps
def chunk_text(input_text):
    """Splits text into paragraphs or logical chunks."""
    return input_text.split("\n\n")  # Simple split by double newline for paragraphs


def extract_claims(paragraph):
    """Extracts factual claims from a paragraph using LLM."""
    prompt = f"Extract factual claims from the following text:\n\n{paragraph}\n\nOutput each claim as a separate line."
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
    return response.choices[0].message.content.strip().split("\n")


def retrieve_evidence(claim):
    """Retrieve evidence for a claim from multiple sources."""
    # Wikipedia retrieval
    # wiki_results = WikipediaLoader(query=claim, load_max_docs=3).load()

    # Google Search retrieval
    google_results = GoogleSearchAPIWrapper(k=3).results(query=claim, num_results=3)

    return {
        # "wikipedia": wiki_results,
        "google": [result["snippet"] for result in google_results]
    }


def classify_evidence(claim, evidence):
    """Classify evidence snippets as Supporting, Contradicting, or Neutral."""
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
            results.append({
                "source": source,
                "snippet": snippet,
                "classification": classification
            })
    return results


def aggregate_results(classifications):
    """Aggregate classifications to determine chunk-level validity."""
    supporting = sum(1 for c in classifications if c["classification"] == "Supporting")
    contradicting = sum(1 for c in classifications if c["classification"] == "Contradicting")
    neutral = sum(1 for c in classifications if c["classification"] == "Neutral")
    return {
        "supporting": supporting,
        "contradicting": contradicting,
        "neutral": neutral,
        "flagged": contradicting > supporting
    }


def generate_revision(paragraph, classifications):
    """Propose a revised paragraph if necessary."""
    supporting_snippets = [c["snippet"] for c in classifications if c["classification"] == "Supporting"]
    contradicting_snippets = [c["snippet"] for c in classifications if c["classification"] == "Contradicting"]

    if contradicting_snippets:
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
        return response.choices[0].message.content.strip()
    return None


# Main function
def verify_text(input_text):
    """Main workflow to verify and revise input text."""
    paragraphs = chunk_text(input_text)
    results = []

    for i, paragraph in enumerate(paragraphs):
        claims = extract_claims(paragraph)
        claims_results = []
        for claim in claims:
            evidence = retrieve_evidence(claim)
            classified_evidence = classify_evidence(claim, evidence)
            claim_aggregate = aggregate_results(classified_evidence)
            claims_results.append({
                "claim": claim,
                "classifications": classified_evidence,
                "aggregate": claim_aggregate
            })

        flagged = any(c["aggregate"]["flagged"] for c in claims_results)
        revised_paragraph = generate_revision(paragraph, [c["classifications"] for c in claims_results]) if flagged else None

        results.append({
            "paragraph": paragraph,
            "claims_results": claims_results,
            "revised_paragraph": revised_paragraph
        })

    return results

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/verify', methods=['POST'])
def verify():
    input_text = request.json.get('text')
    if not input_text:
        return jsonify({"error": "No input text provided."}), 400

    verification_results = verify_text(input_text)

    highlighted_text = input_text
    highlights = []

    for result in verification_results:
        for claim_result in result['claims_results']:
            if claim_result['aggregate']['flagged']:
                highlights.append({
                    "claim": claim_result['claim'],
                    "classifications": claim_result['classifications']
                })

    return jsonify({
        "text": highlighted_text,
        "highlights": highlights
    })

# Example Usage
if __name__ == "__main__":
    # input_text = """In 2020, the global market cap was $50 trillion. The population of the Earth is 8 billion people. Einstein was born in 1905."""
    # output = verify_text(input_text)
    # print(json.dumps(output, indent=4))
    app.run(debug=True, port=9000)
