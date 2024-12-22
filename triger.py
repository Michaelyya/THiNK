import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import ChatOpenAI 
from openai import OpenAI as GPTClient
import pinecone
import time
load_dotenv()
pinecone_api_key = os.environ.get("PINECONE_API_KEY")
embedder = OpenAIEmbeddings()
pc = Pinecone(api_key=pinecone_api_key)
pinecone_index_name = "bloom-testing"
index = pc.Index(pinecone_index_name)

def get_similar_docs(query, k, score=False):
    """Retrieve similar documents based on the input query."""
    query_vector = embedder.embed_query(query)
    results = index.query(vector=query_vector, top_k=3, include_metadata=True)
    similar_docs = results['matches']
    results = [doc['metadata'] for doc in similar_docs]
    return results

llm = ChatOpenAI(model_name="gpt-4", temperature=0)

client = GPTClient(api_key=os.environ.get("OPENAI_API_KEY"))

def trigger_problem(query):
    similar_docs = get_similar_docs(query, k=2, score=True)
    context = "Reference Questions:\n"
    for i, doc in enumerate(similar_docs, 1):
        # Get the text field from each document
        text = doc.get('text', '')
        print(text)
        context += f"Question Set {i}:\n{text}\n\n"
    
    prompt = f""" you are a question generation expert.
    """+context+"""

    Generate a new question that elevates cognitive engagement using Bloom's Taxonomy.

    Format your response as a JSON with the following structure:
    {
        "question": "The complete question text",
        "expected_answer": "The detailed solution approach",
        "hot_skills": ["Specific HOT skills targeted"],
        "blooms_levels": ["Specific Bloom's taxonomy levels"]
    }
    """

    messages = [
        {"role": "system", "content": "You are question generation agent."},
        {"role": "user", "content": prompt}
    ]
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Please follow the instructions provided and generate a response."},
            {"role": "user", "content": prompt}
        ],
        temperature=0,
        max_tokens=2000
    )
    
    return response.choices[0].message.content


if __name__ == "__main__":
    query = "Give me a implicit differentiation question?"
    results = get_similar_docs(query, k=2, score=True)
    print(results)
    new_question = trigger_problem(query)
    print(new_question)