# from typing import Dict, List, Tuple, Any
# from langchain_openai import ChatOpenAI  # Updated import
# from langgraph.graph import StateGraph, END
# from langchain.schema import HumanMessage, SystemMessage
# from pydantic import BaseModel  # Updated import
# import json
# import operator
# from typing import TypedDict
# import os
# from dotenv import load_dotenv


# class QuestionInput(TypedDict):
#     question: str
#     expected_answer: str
#     hot_skills: List[str]
#     blooms_levels: List[str]
#     cognitive_triggers: List[str]

# class QuestionEvalState(TypedDict):
#     input_data: QuestionInput
#     hot_skills_eval: Dict[str, Any]
#     language_eval: Dict[str, Any]
#     final_decision: bool
#     improvement_feedback: List[str]

# def create_hot_skills_agent():
#     hot_skills_prompt = """You are an expert evaluator of Higher Order Thinking (HOT) skills in educational questions.
#     Analyze the given question and its components based on the following criteria:

#     Question Components to Evaluate:
#     1. Main Question: {0}
#     2. Expected Answer proccess: {1}
#     3. Intended HOT Skills: {2}
#     4. Intended Bloom's Levels: {3}
#     5. Intended Cognitive Triggers: {4}

#     Evaluation Criteria:
#     1. Analysis Skills:
#     - Does the question effectively require the listed analytical skills?
#     - Are the cognitive triggers appropriately integrated?
#     - Do the expected answers align with the intended analysis level?

#     2. Creation/Synthesis Skills:
#     - Does the question genuinely require the creation of new approaches?
#     - Are the synthesis elements clearly demonstrated?
#     - Does the expected answer guide toward proper synthesis?

#     3. Evaluation Skills:
#     - Are students required to make justified evaluations?
#     - Do the cognitive triggers promote evaluative thinking?
#     - Does the expected answer demonstrate proper evaluation criteria?

#     4. Alignment:
#     - Do the stated HOT skills match the question's requirements?
#     - Are the Bloom's levels accurately represented?
#     - Do the cognitive triggers effectively support the intended outcomes?

#     Provide your evaluation in JSON format with these exact keys:
#     {{
#         "meets_criteria": true or false,
#         "component_analysis": {{
#             "question_analysis": "detailed analysis here",
#             "answer_alignment": "alignment analysis here",
#             "hot_skills_alignment": "skills analysis here",
#             "blooms_alignment": "blooms analysis here",
#             "triggers_effectiveness": "triggers analysis here"
#         }},
#         "improvement_suggestions": ["suggestion1", "suggestion2"]
#     }}
#     """
    
#     llm = ChatOpenAI(
#         model="gpt-4o-mini",
#         temperature=0,
#         api_key=os.environ.get("OPENAI_API_KEY")
#     )
    
#     def evaluate_hot_skills(state: QuestionEvalState) -> QuestionEvalState:
#         input_data = state['input_data']
#         formatted_prompt = hot_skills_prompt.format(
#             input_data['question'],
#             input_data['expected_answer'],
#             ", ".join(input_data['hot_skills']),
#             ", ".join(input_data['blooms_levels']),
#             ", ".join(input_data['cognitive_triggers'])
#         )
        
#         messages = [
#             SystemMessage(content=formatted_prompt),
#             HumanMessage(content="Please evaluate the question components based on the given criteria.")
#         ]
        
#         response = llm.invoke(messages)
#         evaluation = json.loads(response.content)
#         state['hot_skills_eval'] = evaluation
#         if not evaluation['meets_criteria']:
#             state['improvement_feedback'].extend(evaluation.get('improvement_suggestions', []))
#         return state
    
#     return evaluate_hot_skills

# def create_language_agent():
#     language_prompt = """You are an expert evaluator of question quality and language.
#     Analyze the given question components:

#     Question: {0}
#     Expected Answer: {1}

#     Evaluate based on these criteria:

#     1. Fluency & Coherence:
#     - Is the question grammatically correct and well-structured?
#     - Does it flow logically and maintain coherence?
#     - Is the expected answer clear and well-articulated?

#     2. Clarity & Precision:
#     - Are the requirements clearly stated?
#     - Are mathematical concepts precisely defined?
#     - Is the expected answer sufficiently detailed?

#     3. Technical Language:
#     - Is mathematical terminology used correctly?
#     - Are symbolic notations clear and consistent?
#     - Does the answer use appropriate technical language?

#     4. Answerability:
#     - Are the expectations clear to students?
#     - Does the expected answer provide adequate guidance?
#     - Can the question be answered based on the given information?

#     Provide your evaluation in JSON format with these exact keys:
#     {{
#         "meets_criteria": true or false,
#         "language_analysis": {{
#             "fluency_score": "score description here",
#             "clarity_score": "score description here",
#             "technical_accuracy": "accuracy analysis here",
#             "answerability": "answerability analysis here"
#         }},
#         "improvement_suggestions": ["suggestion1", "suggestion2"]
#     }}
#     """
    
#     llm = ChatOpenAI(
#         model="gpt-4o-mini",
#         temperature=0,
#         api_key=os.environ.get("OPENAI_API_KEY")
#     )
    
#     def evaluate_language(state: QuestionEvalState) -> QuestionEvalState:
#         input_data = state['input_data']
#         formatted_prompt = language_prompt.format(
#             input_data['question'],
#             input_data['expected_answer']
#         )
        
#         messages = [
#             SystemMessage(content=formatted_prompt),
#             HumanMessage(content="Please evaluate the question's language and technical components.")
#         ]
        
#         response = llm.invoke(messages)
#         evaluation = json.loads(response.content)
#         state['language_eval'] = evaluation
#         if not evaluation['meets_criteria']:
#             state['improvement_feedback'].extend(evaluation.get('improvement_suggestions', []))
#         return state
#     return evaluate_language

# def make_decision(state: QuestionEvalState) -> QuestionEvalState:
#     hot_skills_passed = state['hot_skills_eval'].get('meets_criteria', False)
#     language_passed = state['language_eval'].get('meets_criteria', False)
    
#     state['final_decision'] = hot_skills_passed and language_passed
    
#     return state

# def create_evaluation_pipeline():
#     workflow = StateGraph(QuestionEvalState)
#     workflow.add_node("hot_skills", create_hot_skills_agent())
#     workflow.add_node("language", create_language_agent())
#     workflow.add_node("decide", make_decision)

#     workflow.add_edge("hot_skills", "language")
#     workflow.add_edge("language", "decide")
    
#     workflow.add_conditional_edges(
#         "decide",
#         lambda x: "end" if x["final_decision"] else "end",
#         {
#             "end": END
#         }
#     )

#     workflow.set_entry_point("hot_skills")
    
#     return workflow.compile()

# def evaluate_question_components(question_input: str) -> Dict[str, Any]:
#     try:
#         input_data = json.loads(question_input)
    
#         pipeline = create_evaluation_pipeline()
        
#         initial_state = {
#             "input_data": input_data,
#             "hot_skills_eval": {},
#             "language_eval": {},
#             "final_decision": False,
#             "improvement_feedback": []
#         }
#         result = pipeline.invoke(initial_state)
        
#         return {
#             "passed_evaluation": result["final_decision"],
#             "hot_skills_evaluation": result["hot_skills_eval"],
#             "language_evaluation": result["language_eval"],
#             "improvement_suggestions": result["improvement_feedback"]
#         }
#     except json.JSONDecodeError as e:
#         print(f"Error parsing input JSON: {str(e)}")
#         return {
#             "error": "Invalid JSON input format",
#             "passed_evaluation": False,
#             "hot_skills_evaluation": {},
#             "language_evaluation": {},
#             "improvement_suggestions": ["Please check the input JSON format"]
#         }
#     except Exception as e:
#         print(f"Error in evaluation pipeline: {str(e)}")
#         return {
#             "error": str(e),
#             "passed_evaluation": False,
#             "hot_skills_evaluation": {},
#             "language_evaluation": {},
#             "improvement_suggestions": ["Error in evaluation process"]
#         }

# if  __name__ == "__main__":
#     def test_bad_question():
#         bad_question = {
#         "question": "What is implicit differentiation and how do you do it? Also do some examples.",
#         "expected_answer": "Implicit differentiation is when you differentiate both sides. You do it by using the chain rule and stuff. Like for example you can do it with circles and other equations.",
#         "hot_skills": ["Understanding concepts", "Applying formulas", "Doing calculations"],
#         "blooms_levels": ["Knowledge", "Application"],
#         "cognitive_triggers": ["Define", "Calculate", "Apply"]
#     }
#         question_input = json.dumps(bad_question)
        
#         evaluation_result = evaluate_question_components(question_input)
        
#         print("=== Question Evaluation Results ===")
#         print("\nInput Question:")
#         print("-" * 50)
#         print(json.dumps(bad_question, indent=2))
#         print("\nEvaluation Results:")
#         print("-" * 50)
#         print(json.dumps(evaluation_result, indent=2))

#     test_bad_question()