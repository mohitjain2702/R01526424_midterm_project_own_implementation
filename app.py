import os
from flask import Flask, request, render_template
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
import openai
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# Set up OpenAI API
openai.api_type = "azure"
openai.api_key = os.getenv('OPENAI_API_KEY')
openai.api_base = os.getenv('OPENAI_API_BASE')
openai.api_version = os.getenv('OPENAI_API_VERSION')

# Set up OpenAI API
openai.api_type = "azure"
openai.api_key = os.getenv('OPENAI_API_KEY')
openai.api_base = os.getenv('OPENAI_API_BASE')
openai.api_version = os.getenv('OPENAI_API_VERSION')

# Set up Azure Search client
search_api_key = os.getenv('AZURE_SEARCH_API_KEY')
search_endpoint = os.getenv('AZURE_SEARCH_ENDPOINT')
index_name = os.getenv('AZURE_SEARCH_INDEX_NAME')
credential = AzureKeyCredential(search_api_key)
search_client = SearchClient(endpoint=search_endpoint, index_name=index_name, credential=credential)

chat_history = []  # List to store the chat history

def query_embedding(query):
    response = openai.Embedding.create(
        input=query,
        engine=os.getenv('OPENAI_DEPLOYMENT_NAME')
    )
    embedding = response['data'][0]['embedding']
    return embedding

def query_embedding(query):
    response = openai.Embedding.create(
        input=query,
        engine=os.getenv('OPENAI_DEPLOYMENT_NAME')
    )
    embedding = response['data'][0]['embedding']
    return embedding

def process_with_openai(query, results):
    context = "\n".join([result['paragraph'] for result in results])

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": f"""Answer the following question: '{query}'.

Use the provided context if it is relevant. The context is:
{context}

If the context is not relevant, please ignore it and answer the question as you normally would."""
        }
    ]

    response = openai.ChatCompletion.create(
        engine=os.getenv('OPENAI_GPT_DEPLOYMENT_NAME'),
        messages=messages,
        max_tokens=500
    )

    answer = response['choices'][0]['message']['content'].strip()
    return answer

def process_user_query(query):
    # Get the query embedding
    embedding = query_embedding(query)

    # Search the Azure Cognitive Search index
    results = search_client.search(
        search_text=None,
        vector_queries=[
            VectorizedQuery(
                vector=embedding,
                k_nearest_neighbors=5,
                fields="embedding"
            )
        ]
    )

    # Collect results into a list (if necessary)
    results_list = list(results)

    # Process results with OpenAI
    answer = process_with_openai(query, results_list)
    return answer

@app.route('/', methods=['GET', 'POST'])
def chat():
    global chat_history
    if request.method == 'POST':
        user_query = request.form['query']
        response = process_user_query(user_query)
        chat_history.append({'query': user_query, 'response': response})
    return render_template('chat.html', chat_history=chat_history)

if __name__ == '__main__':
    app.run(debug=True)
