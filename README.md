# mCDR research assistant setup

# clone the repository
git clone https://github.com/brookebeers/mcdr-assistant.git
cd mcdr-assistant

# install dependencies
pip install -r requirements.txt

# create secrets file for api keys
mkdir -p .streamlit
touch .streamlit/secrets.toml

# open the secrets file and add your keys
## (you can use nano, vim, or any text editor)
nano .streamlit/secrets.toml

# inside .streamlit/secrets.toml, paste the following:
# replace the placeholder values with your actual keys

openai_api_key = "..."
pinecone_api_key = "..."
pinecone_index_name = "..."

# run the app locally
streamlit run app.py
