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
        text = doc.get('text', '')
        print(text)
        context += f"Question Set {i}:\n{text}\n\n"
    
    prompt = f"""You are an expert educational question designer focusing on Higher Order Thinking (HOT) skills. 
    Based on the following reference questions, generate a new question that elevates cognitive engagement using Bloom's Taxonomy.

    """+context+"""

    Follow these specific guidelines for question generation:

    1. ANALYSIS Level (Breaking down information):
    - Incorporate elements that require students to:
    * Compare and contrast key concepts
    * Analyze relationships between components
    * Identify patterns or underlying principles
    * Deconstruct complex problems into manageable parts

    2. EVALUATION Level (Making judgments):
    - Include aspects that ask students to:
    * Assess the validity of arguments or solutions
    * Make informed decisions based on criteria
    * Justify their reasoning
    * Evaluate the effectiveness of different approaches

    3. SYNTHESIS/CREATION Level (Creating new understanding):
    - Design the question to encourage students to:
    * Combine concepts from different areas
    * Develop new solutions or approaches
    * Create original explanations or models
    * Apply knowledge in novel contexts

    4. Question Structure Requirements:
    * Maintain clear and precise language
    * Provide necessary context without giving away the solution
    * Ensure the question is challenging but achievable
    * Include any relevant constraints or parameters

    5. Learning Outcomes:
    * Challenge students' conceptual understanding
    * Require multi-step thinking processes
    * Encourage creative problem-solving
    * Demand justification of answers

    Generate a new question that builds upon the cognitive level of the reference questions while incorporating these HOT principles.

    Generated question should include:
    1. The main question
    2. Expected answer/solution approach
    3. Specific HOT skills targeted
    4. Bloom's taxonomy level(s) addressed

    Format your response as a JSON with the following structure:
    {
        "question": "The complete question text",
        "expected_answer": "The detailed solution approach",
        "hot_skills": ["Specific HOT skills targeted"],
        "blooms_levels": ["Specific Bloom's taxonomy levels"]
    }
    """

    messages = [
        {"role": "system", "content": "You are an expert educational question designer specializing in Higher Order Thinking skills."},
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