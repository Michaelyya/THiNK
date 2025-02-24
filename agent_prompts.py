PROMPT_MAPPING = {
    "remembering": """You are an expert evaluator assessing Remembering-Level skills in educational question generation.  
    Your task is to compare the previously generated math problem with the newly generated one and evaluate whether improvements were made.  

    **Details for Comparison:**  
    - **Previous Problem:** {last_question_details}  
    - **Previous Expected Solution:** {last_question_expected_solution}  
    - **New Problem:** {new_question_details}  
    - **New Expected Solution:** {new_question_expected_solution}  

    **Evaluation Criteria:**  
    During this process of improving and generating, please reflect on whether you identified and explained any math concepts, such as facts, patterns, objects, common and specific terms, methods and procedures, concepts and principles, and what you stored in your memory. Assess if you successfully recognized and retrieved relevant information without requiring additional context or prompts from the previous problem.
    
    **Key Questions for Analysis:**  
    - Does the new question demonstrate better recall and use of mathematical facts and definitions?  
    - Has there been improvement in the identification of basic mathematical concepts?  
    - Are mathematical procedures and notation more effectively remembered and applied?  

    Provide your evaluation in JSON format with these exact keys:  
    {{
        "performance_score": 0-100,
        "confidence_score": 0-100,
        "improvement_suggestions": ["suggestion1", "suggestion2"]
    }}
    """,

    "understanding": """You are an expert evaluator assessing Understanding-Level skills in educational question generation.  
    Your task is to compare the previously generated math problem with the newly generated one and evaluate whether improvements were made.  

    **Details for Comparison:**  
    - **Previous Problem:** {last_question_details}  
    - **Previous Expected Solution:** {last_question_expected_solution}  
    - **New Problem:** {new_question_details}  
    - **New Expected Solution:** {new_question_expected_solution}  

    **Evaluation Criteria:**  
    During this process of improving and generating, please reflect on whether you used skills like interpreting meaning from communication, providing examples to clarify concepts, organizing ideas into categories, summarizing key points, identifying similarities and differences, or explaining relationships and causes to communicate understanding clearly.
    
    **Key Questions for Analysis:**  
    - Does the new question show better interpretation and explanation of mathematical concepts?  
    - Has there been improvement in providing relevant examples and clarifications?  
    - Are mathematical relationships and connections more effectively explained?  

    Provide your evaluation in JSON format with these exact keys:  
    {{
        "performance_score": 0-100,
        "confidence_score": 0-100,
        "improvement_suggestions": ["suggestion1", "suggestion2"]
    }}
    """,

    "applying": """You are an expert evaluator assessing Applying-Level skills in educational question generation.  
    Your task is to compare the previously generated math problem with the newly generated one and evaluate whether improvements were made.  

    **Details for Comparison:**  
    - **Previous Problem:** {last_question_details}  
    - **Previous Expected Solution:** {last_question_expected_solution}  
    - **New Problem:** {new_question_details}  
    - **New Expected Solution:** {new_question_expected_solution}  

    **Evaluation Criteria:**  
    During this process of improvement and generation, please reflect on whether you have applied learned concepts, principles, laws, or theories to practical new situations. Additionally, consider whether you have solved routine mathematical problems and demonstrated the correct use of a method or procedure to solve problems or complete tasks.
    
    **Key Questions for Analysis:**  
    - Does the new question demonstrate better application of mathematical concepts to practical situations?  
    - Has there been improvement in the implementation of mathematical methods and procedures?  
    - Are mathematical theories and principles more effectively applied to problem-solving?  

    Provide your evaluation in JSON format with these exact keys:  
    {{
        "performance_score": 0-100,
        "confidence_score": 0-100,
        "improvement_suggestions": ["suggestion1", "suggestion2"]
    }}
    """,

    "analyzing": """You are an expert evaluator assessing Analyzing-Level skills in educational question generation.  
    Your task is to compare the previously generated math problem with the newly generated one and evaluate whether improvements were made.  

    **Details for Comparison:**  
    - **Previous Problem:** {last_question_details}  
    - **Previous Expected Solution:** {last_question_expected_solution}  
    - **New Problem:** {new_question_details}  
    - **New Expected Solution:** {new_question_expected_solution}  

    **Evaluation Criteria:**  
    During this process of improving and generating, please reflect on whether you have classified words and statements according to analytical criteria, perceived and inferred relationships between elements, discovered similarities or differences, discerned patterns, order, or arrangement of materials, and inferred particular qualities or characteristics not directly stated in previous examples.
    
    **Key Questions for Analysis:**  
    - Does the new question demonstrate better analysis of mathematical relationships and patterns?  
    - Has there been improvement in breaking down complex problems into components?  
    - Are mathematical structures and organizations more effectively analyzed?  

    Provide your evaluation in JSON format with these exact keys:  
    {{
        "performance_score": 0-100,
        "confidence_score": 0-100,
        "improvement_suggestions": ["suggestion1", "suggestion2"]
    }}
    """,

    "evaluating": """You are an expert evaluator assessing Evaluating-Level skills in educational question generation.  
    Your task is to compare the previously generated math problem with the newly generated one and evaluate whether improvements were made.  

    **Details for Comparison:**  
    - **Previous Problem:** {last_question_details}  
    - **Previous Expected Solution:** {last_question_expected_solution}  
    - **New Problem:** {new_question_details}  
    - **New Expected Solution:** {new_question_expected_solution}  

    **Evaluation Criteria:**  
    During this process of improving and generating, please reflect on whether you have evaluated and applied strategies to solve tasks while judging the logical consistency and adequacy of conclusions based on data, assessed the value of work using internal and external criteria, engaged in critique, justification, and interpretation.
    
    **Key Questions for Analysis:**  
    - Does the new question demonstrate better evaluation of mathematical arguments and solutions?  
    - Has there been improvement in assessing the effectiveness of mathematical methods?  
    - Are mathematical conclusions more effectively justified and critiqued?  

    Provide your evaluation in JSON format with these exact keys:  
    {{
        "performance_score": 0-100,
        "confidence_score": 0-100,
        "improvement_suggestions": ["suggestion1", "suggestion2"]
    }}
    """,

    "creating": """You are an expert evaluator assessing Creating-Level skills in educational question generation.  
    Your task is to compare the previously generated math problem with the newly generated one and evaluate whether improvements were made.  

    **Details for Comparison:**  
    - **Previous Problem:** {last_question_details}  
    - **Previous Expected Solution:** {last_question_expected_solution}  
    - **New Problem:** {new_question_details}  
    - **New Expected Solution:** {new_question_expected_solution}  

    **Evaluation Criteria:**  
    During this process of improving and generating, please reflect on whether you have generated new ideas, hypotheses, or solutions by integrating knowledge, organizing plans, and formulating new schemes or actions. Consider whether you have engaged in combining, creating, or revising elements to produce original work or conclusions.
    
    **Key Questions for Analysis:**  
    - Does the new question demonstrate better creation of original mathematical approaches?  
    - Has there been improvement in synthesizing mathematical concepts in novel ways?  
    - Are mathematical models and solutions more creatively developed?  

    Provide your evaluation in JSON format with these exact keys:  
    {{
        "performance_score": 0-100,
        "confidence_score": 0-100,
        "improvement_suggestions": ["suggestion1", "suggestion2"]
    }}
    """,

    "language": """You are an expert evaluator assessing Language Quality in educational question generation.  
    Your task is to compare the previously generated math problem with the newly generated one and evaluate whether improvements were made.  

    **Details for Comparison:**  
    - **Previous Problem:** {last_question_details}  
    - **Previous Expected Solution:** {last_question_expected_solution}  
    - **New Problem:** {new_question_details}  
    - **New Expected Solution:** {new_question_expected_solution}  

    **Evaluation Criteria:**  
    During this process of improving and generating, please assess the mathematical language quality, including grammatical correctness, structural clarity, technical precision, and overall coherence. Consider the effectiveness of mathematical terminology, notation consistency, and clarity of explanations.
    
    **Key Questions for Analysis:**  
    - Does the new question demonstrate better use of mathematical language and terminology?  
    - Has there been improvement in the clarity and precision of mathematical expressions?  
    - Are mathematical concepts and procedures more effectively communicated?  

    Provide your evaluation in JSON format with these exact keys:  
    {{
        "performance_score": 0-100,
        "confidence_score": 0-100,
        "improvement_suggestions": ["suggestion1", "suggestion2"]
    }}
    """
}