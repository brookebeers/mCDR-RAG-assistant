# mCDR Research Assistant

This repository contains a retrieval-augmented generation (RAG) assistant for synthesizing marine carbon dioxide removal (mCDR) literature. The assistant uses semantic search to find relevant text chunks from a curated database and GPT‑4 to generate answers grounded in those sources. The goal is to reduce hallucinations and provide citation‑rich responses for scientific research.

## Setup

Clone the repository:

```bash
git clone https://github.com/brookebeers/mcdr-assistant.git
cd mcdr-assistant
```
Install dependencies:

```bash
pip install -r requirements.txt
```
Create a secrets file for API keys:

```bash
mkdir -p .streamlit
touch .streamlit/secrets.toml
```
Open the secrets file in a text editor:

```bash
nano .streamlit/secrets.toml
```
Paste the following, replacing the placeholder values with your actual keys:

```bash
toml
openai_api_key = "..."
pinecone_api_key = "..."
pinecone_index_name = "ices-database-assistant"
```
Building the index
Before running the app, you need to populate Pinecone with your documents.

Place your cleaned .txt papers in the formatted_docs/ folder.

Ensure you have a metadata CSV (lit-tag-database.csv) with fields like key, title, author, publication_year, doi, and any relevant URLs.

Run the index builder script:

```bash
python build_index.py
```
This script will:

Chunk each document into ~300‑word segments

Embed chunks with SentenceTransformers

Upsert them into your Pinecone index with metadata

You only need to run this step when adding new papers or rebuilding the index.

Running the app
Once the index is built, launch the Streamlit interface:

```bash
streamlit run app.py
```
The app allows you to:

Enter research questions

Get synthesized answers with inline citations

Review answers for hallucinations

See a bibliography of sources used

Notes
The index building script (build_index.py) is separate from the app and should be run manually when updating the database.

The Streamlit app (app.py) handles querying, summarization, synthesis, and hallucination review.

Token limits may restrict very large queries; adjust top_k or chunk size if needed.

## References

- Martineau, K. (2023). *Artificial intelligence and the risk of hallucination in large language models.*  
- Mike HPG. (2023). [Creating a Custom GPT with RAG](https://medium.com/@mikehpg/creating-a-custom-gpt-with-rag-2441fcabe40f). Medium.  
- Pradeep Varanasi. (2023). [rags_customGPT](https://github.com/pradeepvaranasi/rags_customGPT). GitHub repository.  
- Microsoft Learn. (2023). [Azure Search + OpenAI demo](https://learn.microsoft.com/en-us/samples/azure-samples/azure-search-openai-demo/azure-search-openai-demo/).  
- Baeldung. (2023). [How to Train ChatGPT on Custom Data: A RAG Application](https://www.baeldung.com/cs/chatgpt-rag).  
- Firecrawl. (2023). [15 Best Open-Source RAG Frameworks](https://www.firecrawl.dev/blog/best-open-source-rag-frameworks).  

