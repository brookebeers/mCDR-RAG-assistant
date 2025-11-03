import os
import time
import streamlit as st
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from openai import OpenAI
from openai import RateLimitError


# Load API keys from Streamlit secrets
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
PINECONE_INDEX_NAME = st.secrets.get("PINECONE_INDEX_NAME", "your-index-name")


# Validate keys 
if not OPENAI_API_KEY or not PINECONE_API_KEY:
    st.error("Missing API keys. Please set them in Streamlit's Secrets Manager.")



# Initialize clients
model = SentenceTransformer('all-MiniLM-L6-v2')
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(name=PINECONE_INDEX_NAME)
client = OpenAI(api_key=OPENAI_API_KEY)

# Core Functions 
def retrieve_relevant_chunks(query, top_k=30):
    query_embedding = model.encode(query).tolist()
    results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    return results["matches"]

def format_citation(meta):
    authors = meta.get("author", "Unknown")
    if isinstance(authors, str):
        authors = [a.strip() for a in authors.split(";")]

    author_str = ", ".join(authors)
    year = meta.get("publication_year", "n.d.")
    title = meta.get("title", "Untitled")
    source = meta.get("source", "")  
    doi = meta.get("doi", "")        

    citation = f"{author_str} ({year}). {title}."
    if source:
        citation += f" {source}."
    if doi and doi.lower() != "na":
        citation += f" https://doi.org/{doi}"

    return citation



def truncate(text, max_chars=3000):
    return text[:max_chars]

def summarize_batch(batch, max_retries=3):
    context = "\n\n".join(truncate(m["metadata"]["text"]) for m in batch if "text" in m["metadata"])
    prompt = f"""You are a scientific assistant. Summarize key research gaps in mCDR based on the following context. Cite specific papers and authors where possible.

Context:
{context}

Answer:"""

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
        except RateLimitError:
            wait_time = 2 ** attempt
            st.warning(f"Rate limit hit. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)

    st.error("Rate limit exceeded. Please try again later.")
    st.stop()
    
def build_final_prompt(query, summaries):
    return f"""You are a scientific assistant tasked with synthesizing research on marine carbon dioxide removal (mCDR). Your goal is to produce a clear, well-structured answer that integrates insights from the provided summaries.

Guidelines:
- Use inline citations in the format (LastName, Year). If multiple authors are listed, use the first author's last name followed by 'et al.'.
- Group insights into thematic sections (e.g., Technical Challenges, Ecological Impacts, Social Considerations, Monitoring & Validation).
- Reference specific papers and authors wherever possible. Avoid vague phrases like "some studies" or "research shows".
- If the summaries mention tools, frameworks, or strategies (e.g., Digital Twins of the Ocean, MRV systems), explain their role and limitations.
- If there are more papers than could be processed, end with: "More papers available."

Question:
{query}

Summaries:
{'\n\n'.join(summaries)}

Answer:"""

def generate_response_with_citations(query, matches):
    batch_size = 5
    summaries = []
    citations = []

    for i in range(0, len(matches), batch_size):
        batch = matches[i:i+batch_size]
        summaries.append(summarize_batch(batch)) 
        for match in batch:
            citations.append(format_citation(match["metadata"]))

    if not any(summaries):
        st.error("Summarization failed. Please try again or check your API quota.")
        st.stop()

    final_prompt = build_final_prompt(query, summaries)

    final_response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": final_prompt}]
    )

    unique_citations = sorted(set(citations))
    bibliography = "\n".join(f"{i+1}. {cite}" for i, cite in enumerate(unique_citations))

    return final_response.choices[0].message.content + "\n\nReferences:\n" + bibliography




def check_for_hallucinations(query, response, retrieved_chunks, max_context_chars=6000):
    context = ""
    for chunk in retrieved_chunks:
        if "text" in chunk["metadata"]:
            next_chunk = chunk["metadata"]["text"]
            if len(context) + len(next_chunk) > max_context_chars:
                break
            context += "\n\n" + next_chunk

    prompt = f"""You are a scientific reviewer. Evaluate whether the following answer is fully supported by the context.

Query: {query}

Answer:
{response}

Context:
{context}

Does the answer contain any unsupported claims, hallucinations, or fabricated citations? If so, list them. Otherwise, say 'Fully grounded.'"""
    
    review = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return review.choices[0].message.content

# Streamlit 
st.set_page_config(page_title="mCDR Research Assistant", layout="wide")
st.title("🔬 mCDR Research Assistant")
st.markdown("Ask a research question and get synthesized insights from your literature database.")

query = st.text_input("Enter your research question:")
run_review = st.checkbox(" Run hallucination review")

if query:
    with st.spinner("Retrieving and synthesizing..."):
        docs = retrieve_relevant_chunks(query, top_k=30)
        if not docs:
            st.warning("No relevant documents found.")
        else:
            answer = generate_response_with_citations(query, docs)
            st.subheader("Assistant's results")
            st.markdown(answer)

            if run_review:
                with st.spinner("Checking for hallucinations..."):
                    review = check_for_hallucinations(query, answer, docs)
                    st.subheader(" Hallucination Review")
                    st.markdown(review)

st.markdown("---")
st.caption("Built by Brooke Beers using GPT-4, Pinecone, and SentenceTransformers. Data sourced from ICES mCDR database.")
