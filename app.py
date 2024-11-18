import os
from flask import Flask, request, render_template, session, redirect, url_for
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
import openai
from dotenv import load_dotenv

import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY')  # Set a secret key for session management

# # Set up OpenAI API
# openai.api_type = "azure"
# openai.api_key = os.getenv('OPENAI_API_KEY')
# openai.api_base = os.getenv('OPENAI_API_BASE')
# openai.api_version = os.getenv('OPENAI_API_VERSION')

# # Set up Azure Search client
# search_api_key = os.getenv('AZURE_SEARCH_API_KEY')
# search_endpoint = os.getenv('AZURE_SEARCH_ENDPOINT')
# index_name = os.getenv('AZURE_SEARCH_INDEX_NAME')
# credential = AzureKeyCredential(search_api_key)
# search_client = SearchClient(endpoint=search_endpoint, index_name=index_name, credential=credential)

########
llm = AzureChatOpenAI(
    azure_deployment="gpt35",  # or your deployment
    api_version="2024-06-01",  # or your api version
    temperature=0.5,
    max_tokens= 500
)

chat_history_ai = []

embeddings: AzureOpenAIEmbeddings = AzureOpenAIEmbeddings(
    azure_deployment="text-embedding-ada-002" ,
    openai_api_version="2024-06-01",
    azure_endpoint="https://genai-llm-model.openai.azure.com/",
    api_key="10oPy0AW8FMu5aF9zWEMQKdQnA7bZN6SchzxpwTUV0CqgdmjSbGKJQQJ99AKACYeBjFXJ3w3AAABACOGbWyQ",
)

vector_store: AzureSearch = AzureSearch(
    azure_search_endpoint="https://demo-bro.search.windows.net",
    azure_search_key="s8nUN48JXVT0rCUEPoFD5wKZ4qWF4Y4zyb1lBgz4WlAzSeBm4nPJ",
    index_name="demo",
    embedding_function=embeddings.embed_query,
    
)
retriever = vector_store.as_retriever()

contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

'''
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
'''
@app.route('/', methods=['GET', 'POST'])
def chat():
    if 'chat_history' not in session:
        session['chat_history'] = [] 
  # Initialize session-specific chat history

    if request.method == 'POST':
        user_query = request.form['query']
        
        # response = process_user_query(user_query)
        # session['chat_history'].append({'query': user_query, 'response': response})

        chain_input = {
            "input": user_query,
            "chat_history": chat_history_ai
        }

        response = rag_chain.invoke(chain_input)

        print(response['answer'])

        chat_history_ai.extend([HumanMessage(content = request.form['query']), AIMessage(content= response['answer'])])

        session['chat_history'].append({'query': user_query, 'response': str(response['answer']) })

        session.modified = True
        # Redirect to clear the form submission state
        return redirect(url_for('chat'))

    return render_template('chat.html', chat_history=session['chat_history'])

if __name__ == '__main__':
    app.run(debug=True)
