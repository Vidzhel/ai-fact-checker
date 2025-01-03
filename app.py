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
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
import asyncio

EMBEDDINGS_FOLDER = './embeddings'
VECTOR_STORE_PATH = os.path.join(EMBEDDINGS_FOLDER, 'vector_store.json')
ENABLE_QUERY_OPTIMIZATION = os.environ.get('ENABLE_QUERY_OPTIMIZATION') == 'true'
ENABLE_GOOGLE_SEARCH = os.environ.get('ENABLE_GOOGLE_SEARCH') == 'true'

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

client = openai.AsyncOpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)
model = "gpt-4o-mini"
advanced_model = "gpt-4o-mini"
system_prompt = (f"You are a highly objective scientist and researcher with expertise in evaluating claims "
                 f"against evidence. Your goal is to assess relationships between claims and provided evidence, "
                 f"relying only on factual information and verifiable sources"
                 f"Your analysis should avoid assumptions, focusing only on the evidence provided and its credibility. "
                 f"Respond concisely and clearly in the specified output format, ensuring objectivity and "
                 f"adherence to the criteria.")

vector_store: InMemoryVectorStore | None = None

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


class ClaimQuery(BaseModel):
    google: str
    wiki: str
    vector_store: str


class ClaimQueries(BaseModel):
    list: list[ClaimQuery]


class ClassifiedClaim(BaseModel):
    relevant: bool
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


async def extract_claims(paragraph):
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
        f"6. Analyse every sentence separately and paragraph as a whole"
        f"7. Ensure claims are extracted accurately, even if they are embedded in longer sentences.\n\n'''${paragraph}'''")
    response = await client.beta.chat.completions.parse(
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


async def retrieve_claim_queries(claims: list[Claim]):
    default_queries = ClaimQueries(
        list=[ClaimQuery(google=claim.claim_text, vector_store=claim.claim_text, wiki=claim.claim_text)
              for claim in claims])
    if not ENABLE_QUERY_OPTIMIZATION:
        return default_queries

    stringified_claims = "\n".join([f"{claim.claim_text}" for claim in claims])
    prompt = (
        f"Your task is to generate three distinct search queries to search for evidence for EACH of the given claims. "
        f"These queries should be optimized for:"
        f"1. Wikipedia (using a Python package like `wikipedia-api` or similar)."
        f"2. Google Search."
        f"3. A vector store for semantic search.\n\n"
        f"Guidelines:"
        f"1. **Wikipedia Query**: Create a concise query using the most relevant keywords from the claim, structured to match typical Wikipedia titles or content."
        f"2. **Google Query**: Formulate a detailed and natural-language query designed to retrieve the most relevant search results from a general search engine."
        f"3. **Vector Store Query**: Generate a semantic representation of the claim, preserving its full meaning, for use in vector-based retrieval systems.\n\nCLAIMS: {stringified_claims}")
    response = await client.beta.chat.completions.parse(
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
        response_format=ClaimQueries
    )
    queries = response.choices[0].message.parsed
    if queries:
        return queries

    return default_queries


async def retrieve_evidence(claimQuery: ClaimQuery):
    """Retrieve evidence for a claim from multiple sources."""
    results = []
    logging.info(f"Retrieving evidence for claim: {claimQuery}")

    try:
        docs = await vector_store.asimilarity_search(claimQuery.vector_store, k=3)
        results.extend(
            [{'snippet': result.page_content, 'short_snippet': result.page_content,
              'credibility_justification': 'User provided document', 'credibility': 1,
              'source': f"{result.metadata['source']} - page: {result.metadata['page']}"} for result
             in
             docs])
    except Exception as e:
        logging.error(f"Error retrieving vector_store results: {e}")

    try:
        wiki_results = await WikipediaLoader(query=claimQuery.wiki, load_max_docs=3).aload()
        results.extend([{'snippet': result.page_content, 'short_snippet': result.metadata['summary'],
                         'source': result.metadata['source']} for result in wiki_results])
    except Exception as e:
        logging.error(f"Error retrieving wiki results: {e}")

    if ENABLE_GOOGLE_SEARCH:
        try:
            google_results = GoogleSearchAPIWrapper(k=3).results(query=claimQuery.google, num_results=5)
            logging.info(f"Retrieved Google results: {google_results}")
            results.extend(
                [{'snippet': result["snippet"], 'short_snippet': result["snippet"], 'source': result['link']}
                 for result in google_results])
        except Exception as e:
            logging.error(f"Error retrieving google results: {e}")

    return results


async def classify_evidence(claim, evidence):
    """Classify evidence snippets as Supporting, Contradicting, or Neutral."""
    logging.info(f"Classifying evidence for claim: {claim}")
    prompt = f"Claim: '''{claim}'''"
    for item in evidence:
        prompt += f"\n\nEvidence: '''{item['snippet']}'''\nSource: '''{item['source']}'''"

    response = await client.beta.chat.completions.parse(
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
                            f"- Relevance: Assess whether the evidence is contextually relevant to the claim before"
                            f"determining the relationship. "
                            f"- If evidence is irrelevant, set 'relevant' field to false and the decision should "
                            f"default to '{DecisionClaimEnum.neutral}'.\n\n"
                            f"CLAIMS:\n{prompt}")
            }
        ],
        model=advanced_model,
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
        "credibility": claim_evidence.get('credibility', claim_classification.source_credibility),
        "evidence": claim_classification.key_evidence,
        "credibility_justification": claim_evidence.get('credibility_justification',
                                                        claim_classification.credibility_justification),
    } for (claim_evidence, claim_classification) in zip(evidence, claims.list)
        if claim_classification.source_credibility > 0.5 and claim_classification.relevant],
        key=lambda x: x["credibility"],
        reverse=True)


def aggregate_results(classifications):
    """Aggregate classifications to determine chunk-level validity."""
    logging.info("Aggregating classification results")
    supporting = sum(1 for c in classifications if c["classification"] == DecisionClaimEnum.supporting)
    contradicting = sum(1 for c in classifications if c["classification"] == DecisionClaimEnum.contradicting)
    neutral = sum(1 for c in classifications if c["classification"] == DecisionClaimEnum.neutral)

    supporting_credibility = sum(
        c['credibility'] for c in classifications if c["classification"] == DecisionClaimEnum.supporting)
    contradicting_credibility = sum(
        c['credibility'] for c in classifications if c["classification"] == DecisionClaimEnum.contradicting)
    flagged = False
    # Certain correct - we have only supporting evidence. No changes done. If the certainty is low, we do not have
    # contradicting evidence to update
    certainty = 0
    if contradicting_credibility == 0 and supporting_credibility != 0:
        certainty = supporting_credibility / supporting
    # Calculate how certain it's incorrect. We have both supporting and contradicting evidence,
    # so in case contradicting certainty is high, we flag the text for update
    if supporting_credibility != 0 and contradicting_credibility != 0:
        incorrect_certainty = contradicting_credibility / (supporting_credibility + contradicting_credibility)
        correct_certainty = 1 - incorrect_certainty
        flagged = incorrect_certainty >= 0.6
        certainty = incorrect_certainty if flagged else correct_certainty

    # Certain incorrect - we have only contradicting evidence, so we do change if we are confident enough
    # Maybe it's more logical
    if contradicting_credibility != 0 and supporting_credibility == 0:
        certainty = contradicting_credibility / contradicting
        flagged = certainty >= 0.6

    aggregate = {
        "supporting": supporting,
        "contradicting": contradicting,
        "neutral": neutral,
        "flagged": flagged,
        "certainty": round(certainty, 2)
    }
    logging.info(f"Aggregated results: {aggregate}")
    return aggregate


async def generate_revision(paragraph, claims_results):
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
    response = await client.beta.chat.completions.parse(
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


async def verify_text(input_text):
    """Main workflow to verify and revise input text."""
    logging.info("Starting text verification")
    paragraphs = chunk_text(input_text)
    results = await asyncio.gather(
        *[process_paragraph(paragraph) for paragraph in paragraphs]
    )

    logging.info("Completed text verification")
    return results


async def process_paragraph(paragraph):
    claims = await extract_claims(paragraph)
    claim_queries = await retrieve_claim_queries(claims)
    claims_results = await asyncio.gather(
        *[process_claim(claim_obj, claim_query) for claim_obj, claim_query in zip(claims, claim_queries.list)]
    )

    flagged = any(c["aggregate"]["flagged"] for c in claims_results)
    revised_paragraph, paragraph_diff = generate_revision(paragraph, claims_results) if flagged else (None, None)
    if (revised_paragraph is None) or (paragraph_diff is None):
        return {
            "paragraph": paragraph,
            "flagged": flagged,
            "claims_results": claims_results,
        }
    else:
        return {
            "paragraph": paragraph,
            "flagged": flagged,
            "claims_results": claims_results,
            "revised_paragraph": revised_paragraph.revised_paragraph,
            "change_explanation": revised_paragraph.explanation,
            "paragraph_diff": paragraph_diff
        }


async def process_claim(claim_obj, claim_query):
    claim = claim_obj.claim_text
    original_text = claim_obj.original_text
    evidence = await retrieve_evidence(claim_query)
    classified_evidence = await classify_evidence(claim, evidence)
    claim_aggregate = aggregate_results(classified_evidence)
    return {
        "claim": claim,
        "original_text": original_text,
        "classifications": classified_evidence,
        "aggregate": claim_aggregate
    }


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/verify', methods=['POST'])
async def verify():
    input_text = request.json.get('text')
    if not input_text:
        logging.error("No input text provided")
        return jsonify({"error": "No input text provided."}), 400

    logging.info("Received text for verification")
    verification_results = await verify_text(input_text)

    text = input_text
    paragraphs = []

    for result in verification_results:
        highlights = []
        for claim_result in result['claims_results']:
            highlights.append({
                "claim": claim_result['claim'],
                "original_text": claim_result['original_text'],
                "classifications": claim_result['classifications'],
                "aggregate": claim_result['aggregate']
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


def save_vector_store(vector_store, path=VECTOR_STORE_PATH):
    """Save the vector store to a JSON file."""
    vector_store.dump(path)


def load_vector_store(path=VECTOR_STORE_PATH):
    """Load the vector store from a JSON file."""
    if os.path.exists(path):
        return InMemoryVectorStore.load(path, embedding=OpenAIEmbeddings(model="text-embedding-3-small"))
    return InMemoryVectorStore(embedding=OpenAIEmbeddings(model="text-embedding-3-small"))


def embed_pdfs_from_folder(folder_path=EMBEDDINGS_FOLDER):
    """Embed PDFs from a folder and update the vector store."""
    pdf_files = {f for f in os.listdir(folder_path) if f.endswith('.pdf')}
    existing_files = set(vector_store.store.keys())

    # Add new files
    new_files = pdf_files - existing_files
    for pdf_file in new_files:
        file_path = os.path.join(folder_path, pdf_file)
        loader = PyPDFLoader(file_path)
        documents = loader.load_and_split()
        for document in documents:
            vector_store.add_documents(documents=[document], ids=[pdf_file])

    # Remove deleted files
    removed_files = existing_files - pdf_files
    vector_store.delete(ids=list(removed_files))

    # Save the updated vector store
    save_vector_store(vector_store)


vector_store = load_vector_store()
embed_pdfs_from_folder()
