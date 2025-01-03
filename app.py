import openai
from langchain_community.document_loaders import WikipediaLoader
from langchain_google_community import GoogleSearchAPIWrapper
import os
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify
import logging
from pydantic import BaseModel
from enum import Enum
from diff_match_patch import diff_match_patch

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

client = openai.OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)
model = "gpt-4o-mini"
system_prompt = (f"You are a highly objective scientist and researcher with expertise in evaluating claims "
                 f"against evidence. Your goal is to assess relationships between claims and provided evidence, "
                 f"relying only on factual information and verifiable sources"
                 f"Your analysis should avoid assumptions, focusing only on the evidence provided and its credibility. "
                 f"Respond concisely and clearly in the specified output format, ensuring objectivity and "
                 f"adherence to the criteria.")

app = Flask(__name__)


class Claim(BaseModel):
    original_text: str
    claim_text: str


class Claims(BaseModel):
    list: list[Claim]


class DecisionClaimEnum(str, Enum):
    supporting = 'Supporting'
    contradicting = 'Contradicting'
    neutral = 'Neutral'


class ClassifiedClaim(BaseModel):
    decision: DecisionClaimEnum
    source_credibility: float
    key_evidence: str
    credibility_justification: str


class ClassifiedClaims(BaseModel):
    list: list[ClassifiedClaim]


class RevisedParagraph(BaseModel):
    revised_paragraph: str
    explanation: str


# Functions for core steps
def chunk_text(input_text):
    """Splits text into paragraphs or logical chunks."""
    logging.info("Chunking text")
    return input_text.split("\n\n")  # Simple split by double newline for paragraphs


def extract_claims(paragraph):
    """Extracts factual claims from a paragraph using LLM."""
    logging.info(f"Extracting claims from paragraph: {paragraph}")
    prompt = (
        f"Extract all factual claims from the given text. A factual claim is any statement that asserts something "
        f"as a fact and can be verified\n"
        f"Guidelines:"
        f"1. The 'claim_text' should distill the factual assertion while preserving its meaning. Remove unrelated phrases or context."
        f"2. The 'original_text' must exactly match the corresponding part of the input text from which the claim was derived."
        f"3. Do not include opinions, speculative statements, or rhetorical questions as claims."
        f"4. If there is ambiguity in the claim, provide the most concise and accurate interpretation."
        f"5. If there is no claims, output empty list"
        f"6. Ensure claims are extracted accurately, even if they are embedded in longer sentences.\n\n'''${paragraph}'''")
    response = client.beta.chat.completions.parse(
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model=model,
        response_format=Claims
    )
    claims = response.choices[0].message.parsed
    if not claims:
        logging.info("No claims extracted")
        return []

    logging.info(f"Extracted claims: {claims}")
    return claims.list


def retrieve_evidence(claim):
    """Retrieve evidence for a claim from multiple sources."""
    results = []
    logging.info(f"Retrieving evidence for claim: {claim}")

    try:
        wiki_results = WikipediaLoader(query=claim, load_max_docs=3).load()
        results.extend([{'snippet': result.page_content, 'short_snippet': result.metadata['summary'],
                         'source': result.metadata['source']} for result in wiki_results])
    except Exception as e:
        logging.error(f"Error retrieving wiki results: {e}")

    try:
        google_results = GoogleSearchAPIWrapper(k=3).results(query=claim, num_results=5)
        logging.info(f"Retrieved Google results: {google_results}")
        results.extend(
            [{'snippet': result["snippet"], 'short_snippet': result["snippet"], 'source': result['link']} for result in
             google_results])
    except Exception as e:
        logging.error(f"Error retrieving google results: {e}")

    return results


def classify_evidence(claim, evidence):
    """Classify evidence snippets as Supporting, Contradicting, or Neutral."""
    logging.info(f"Classifying evidence for claim: {claim}")
    prompt = f"Claim: '''{claim}'''"
    for item in evidence:
        prompt += f"\n\nEvidence: '''{item['snippet']}'''\nSource: '''{item['source']}'''"

    response = client.beta.chat.completions.parse(
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": (f"Evaluate the relationship between the given claim and the provided evidence. "
                            f"For each claim, determine whether the evidence supports, contradicts, or is neutral to the claim.\n"
                            f"Criteria for Decision:"
                            f"- {DecisionClaimEnum.supporting}: Evidence explicitly affirms the claim with no ambiguity."
                            f"- {DecisionClaimEnum.contradicting}: Evidence explicitly denies or disputes the claim with no ambiguity."
                            f"- {DecisionClaimEnum.neutral}: Evidence is unrelated, ambiguous, or insufficient to affirm or deny the claim.\n"
                            f"Additionally, assess the credibility of the source using the following scale:"
                            f"- 1.0: Peer-reviewed resources (e.g., white papers, academic journals)"
                            f"- 0.8: Crowd-sourced resources with robust moderation (e.g., Wikipedia)"
                            f"- 0.6: Reputable news outlets and well-known publications"
                            f"- Below 0.6: Sources with questionable credibility or lack of verification\n"
                            f"The result should include:"
                            f"1. Decision: '{DecisionClaimEnum.supporting}', '{DecisionClaimEnum.contradicting}', or '{DecisionClaimEnum.neutral}'"
                            f"2. Source Credibility: A float value between 0 and 1, as defined above"
                            f"3. Extract key phrases from the evidence that support the decision."
                            f"4. Provide a brief justification for the credibility score.\n"
                            f"Additional Step:"
                            f"- Relevance: Assess whether the evidence is contextually relevant to the claim before "
                            f"determining the relationship. If irrelevant, the decision should default to '{DecisionClaimEnum.neutral}'.\n\n"
                            f"CLAIMS:\n{prompt}")
            }
        ],
        model=model,
        response_format=ClassifiedClaims
    )

    claims = response.choices[0].message.parsed
    if not claims:
        logging.info("No claims extracted")
        return []

    logging.info(f"Classified evidence: {claims}")
    return sorted([{
        "source": claim_evidence['source'],
        "snippet": claim_evidence['snippet'],
        "short_snippet": claim_evidence['short_snippet'],
        "classification": claim_classification.decision,
        "credibility": claim_classification.source_credibility,
        "evidence": claim_classification.key_evidence,
        "credibility_justification": claim_classification.credibility_justification
    } for (claim_evidence, claim_classification) in zip(evidence, claims.list)
        if claim_classification.source_credibility > 0.5],
        key=lambda x: x["credibility"],
        reverse=True)


def aggregate_results(classifications):
    """Aggregate classifications to determine chunk-level validity."""
    logging.info("Aggregating classification results")
    supporting = sum(1 for c in classifications if c["classification"] == DecisionClaimEnum.supporting)
    contradicting = sum(1 for c in classifications if c["classification"] == DecisionClaimEnum.contradicting)
    neutral = sum(1 for c in classifications if c["classification"] == DecisionClaimEnum.neutral)
    aggregate = {
        "supporting": supporting,
        "contradicting": contradicting,
        "neutral": neutral,
        "flagged": contradicting > supporting
    }
    logging.info(f"Aggregated results: {aggregate}")
    return aggregate


def generate_revision(paragraph, claims_results):
    """Propose a revised paragraph if necessary."""
    logging.info(f"Generating revision for paragraph: {paragraph}")

    flagged_claims = [c for c in claims_results if c["aggregate"]["flagged"]]

    if not flagged_claims:
        return None, None

    claims_to_improve = ""

    for claim in flagged_claims:
        supporting_snippets = "\n".join(
            [f"- Credibility: {c['credibility']} - {c['snippet']}" for c in claim['classifications'] if
             c["classification"] == DecisionClaimEnum.supporting])
        contradicting_snippets = "\n".join(
            [f"- Credibility: {c['credibility']} {c['snippet']}" for c in claim['classifications'] if
             c["classification"] == DecisionClaimEnum.contradicting])
        if not contradicting_snippets:
            continue

        claims_to_improve += (f"Claim: {claim['claim']}\n"
                              f"Supporting Evidence:\n{supporting_snippets}\n"
                              f"Contradicting Evidence:\n{contradicting_snippets}\n\n")

    if claims_to_improve == "":
        return None, None

    prompt = (
        f"Original Paragraph: \n{paragraph}\n\n"
        f"Flagged Claims:\n{claims_to_improve}\n\n"
        f"Task: \n Revise the paragraph to address flagged contradictions in the claims and ensure factual accuracy. "
        f"Make only the necessary changes to resolve inaccuracies, preserving the original meaning and tone of the "
        f"paragraph.\n\n"
        f"Instructions:"
        f"1. Revise only the flagged claims listed under 'Flagged Claims' to ensure they are factually correct. "
        f"Do not edit unrelated content."
        f"2. Make minimal changes: adjust only the flagged claims and directly related phrasing to fix contradictions "
        f"while leaving the rest of the paragraph untouched."
        f"3. Do not modify style, grammar, or expand context unless necessary for factual correction."
        f"4. If a contradiction cannot be resolved due to insufficient information, state explicitly that the claim requires clarification or additional evidence."
        f"5. Provide a short explanation for each revision, describing how the change ensures factual accuracy."
    )
    response = client.beta.chat.completions.parse(
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model=model,
        response_format=RevisedParagraph
    )
    revised_paragraph = response.choices[0].message.parsed
    if not revised_paragraph:
        logging.info("Did not revise paragraph")
        return None, None

    logging.info(f"Generated revised paragraph: {revised_paragraph}")
    dmp = diff_match_patch()
    diffs = dmp.diff_main(paragraph, revised_paragraph.revised_paragraph)
    diff_html = dmp.diff_prettyHtml(diffs)

    return revised_paragraph, diff_html


def verify_text(input_text):
    """Main workflow to verify and revise input text."""
    logging.info("Starting text verification")
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
        revised_paragraph, paragraph_diff = generate_revision(paragraph, claims_results) if flagged else (None, None)
        if (revised_paragraph is None) or (paragraph_diff is None):
            results.append({
                "paragraph": paragraph,
                "flagged": flagged,
                "claims_results": claims_results,
            })
        else:
            results.append({
                "paragraph": paragraph,
                "flagged": flagged,
                "claims_results": claims_results,
                "revised_paragraph": revised_paragraph.revised_paragraph,
                "change_explanation": revised_paragraph.explanation,
                "paragraph_diff": paragraph_diff
            })

    logging.info("Completed text verification")
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

    logging.info("Received text for verification")
    verification_results = verify_text(input_text)

    text = input_text
    paragraphs = []

    for result in verification_results:
        highlights = []
        for claim_result in result['claims_results']:
            if claim_result['aggregate']['flagged']:
                highlights.append({
                    "claim": claim_result['claim'],
                    "original_text": claim_result['original_text'],
                    "classifications": claim_result['classifications']
                })

        if result['flagged']:
            paragraphs.append({
                "paragraph": result['paragraph'],
                "flagged": result['flagged'],
                "change_explanation": result['change_explanation'],
                "paragraph_diff": result['paragraph_diff'],
                "revised_paragraph": result['revised_paragraph'],
                "highlights": highlights
            })
        else:
            paragraphs.append({
                "paragraph": result['paragraph'],
                "flagged": result['flagged'],
                "highlights": highlights
            })

    logging.info("Returning verification results")
    return jsonify({
        "text": text,
        "paragraphs": paragraphs
    })


# Example Usage
if __name__ == "__main__":
    app.run(debug=True, port=9000)
