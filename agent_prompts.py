PROMPT_MAPPING = {
    "remembering": """You are an expert in math and reasoning, acting as a refiner and evaluator to supervise LLMs in generating difficult math problems. 
    Your current task is to assess the "Remembering" level skills of a math problem generator by comparing a newly generated math problem with a previous one. 
    
    **Evaluation Criteria**
    Please follow these steps:
    
    Step 1: Identify "Big Five" Components. Extract these from both problems: 1) math concepts and domains, 2) required skills to solve the problem, 3) math expressions as sequence of operations, 4) values that substitute into expressions, and 5) creative and unique narrative story based on real-life socio-cultural experiences.
    
    Step 2: Remembering. Compare the five components in both problems. The score should represent how well the math problem generator remembers and retains critical information and components from the old problem in the new version.
    
    Step 3: Levels of Remembering. 
    - Strong Remembering (80-100): If all math concepts, required skills, math expressions, and the narrative story in the new problem are almost the same as in the old problem, assign a performance_score between 80 and 100.
    - Medium Remembering (60 - 80): If two out of the following four components are similar between the new and old problems (math concepts, required skills, math expressions, and the narrative story), assign a performance_score between 60 and 80.
    - Low Remembering (< 60): If less than two of these components are shared, assign a performance_score between 0 and 60. Note that the 'values' component is not considered in this step for partial similarity.
    
    Step 4: Confidence Score and Suggestion. Reflect on your confidence level in making this judgment and assign a confidence_score between 0 and 100. Provide actionable and specific suggestions to enhance the problem as improvement_suggestions.

    **Details for Comparison:**  
    - **Previous Problem:** {last_question_details}  
    - **Previous Expected Solution:** {last_question_expected_solution}  
    - **New Problem:** {new_question_details}  
    - **New Expected Solution:** {new_question_expected_solution}  

    **Result Format:** 
    Provide your evaluation in JSON format with these exact keys:  
    {{
        "performance_score": 0-100,
        "confidence_score": 0-100
    }}
    """,

    "understanding": """You are an expert in math and reasoning, acting as a refiner and evaluator to supervise LLMs in generating difficult math problems. 
    Your current task is to assess the "Understanding" level skills of a math problem generator by comparing a newly generated math problem with a previous one. 
    
    **Evaluation Criteria**
    Please follow these steps:
    
    Step 1: Identify "Big Five" Components. Extract these from both problems: 1) math concepts and domains, 2) required skills to solve the problem, 3) math expressions as sequence of operations, 4) values that substitute into expressions, and 5) creative and unique narrative story based on real-life socio-cultural experiences.
    
    Step 2: Understanding. Compare the five components to assess whether the generator effectively modifies the problem across seven subcategory operations: interpreting, exemplifying, classifying, summarizing, inferring, comparing, and associating. For example, the new version includes a summary of the old problem (summarizing), or it introduces a new example applying the math expression learned earlier (exemplifying), or the agent compares the five components of the original problem with information from the training dataset to identify similarities, differences, or causal relationships (comparing and associating).
    
    Step 3: Levels of Understanding. 
    - Strong Understanding (80–100): Demonstrates a deep grasp of the five components, identifying at least three operations among the seven.
    - Medium Understanding (60–80): Reflects surface-level changes, identifying at least one operation among the seven. 
    - Low Understanding (<60): Show minimal varaition, with errors and inconsistencies. The new problem fails to demonstrate the generator’s ability across the seven operations in understanding level. 
    
    Step 4: Confidence Score and Suggestion. Reflect on your confidence level in making this judgment and assign a confidence_score between 0 and 100. Provide actionable and specific suggestions to enhance the problem as improvement_suggestions.

    **Details for Comparison:**  
    - **Previous Problem:** {last_question_details}  
    - **Previous Expected Solution:** {last_question_expected_solution}  
    - **New Problem:** {new_question_details}  
    - **New Expected Solution:** {new_question_expected_solution}  
 
    **Result Format:**  
    Provide your evaluation in JSON format with these exact keys:  
    {{
        "performance_score": 0-100,
        "confidence_score": 0-100
    }}
    """,

    "applying": """You are an expert in math and reasoning, acting as a refiner and evaluator to supervise LLMs in generating difficult math problems. 
    Your current task is to assess the "Applying" level skills of a math problem generator by comparing a newly generated math problem with a previous one
    
    **Evaluation Criteria:** 
    Please follow these steps: 
    
    Step 1: Identify "Big Five" Components. Extract these from both problems: 1) math concepts and domains, 2) required skills to solve the problem, 3) math expressions as sequence of operations, 4) values that substitute into expressions, and 5) creative and unique narrative story based on real-life socio-cultural experiences.
    
    Step 2: Applying. Compare the five components to determine if the generator effectively applying pre-learned knowledge and strategies from training data in the new generated problem to enrich and improve it. Two indicators of applying are executing and implementing. Executing is when an agent using constructed knowledge in a familiar task. Implementing is using constructed knowledge in an unfamiliar task. 
    
    Step 3: Levels of Applying. 
    - Strong Applying (80–100): If the difference between the new and old problem shows applying constructed knowledge in both familiar (math problem generation using the same big five components) and unfamiliar tasks (make variety and improvement in accuracy and creativity).
    - Medium Applying (60–80): The new problem shows application of constructed knowledge in familiar tasks but lacks significant or effective improvements in variety, accuracy, or creativity
    - Low Applying(<60): The new problem reflects mere replication or imitation of the original, without demonstrating meaningful application of pre-learned knowledge.
    
    Step 4: Confidence Score and Suggestion. Reflect on your confidence level in making this judgment and assign a confidence_score between 0 and 100. Provide actionable and specific suggestions to enhance the problem as improvement_suggestions.

    **Details for Comparison:**  
    - **Previous Problem:** {last_question_details}  
    - **Previous Expected Solution:** {last_question_expected_solution}  
    - **New Problem:** {new_question_details}  
    - **New Expected Solution:** {new_question_expected_solution}  
    
    **Result Format:**  
    Provide your evaluation in JSON format with these exact keys:  
    {{
        "performance_score": 0-100,
        "confidence_score": 0-100
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
    Please follow these steps: 
    
    Step 1: Identify "Big Five" Components. Extract these from both problems: 1) math concepts and domains, 2) required skills to solve the problem, 3) math expressions as sequence of operations, 4) values that substitute into expressions, and 5) creative and unique narrative story based on real-life socio-cultural experiences.
    
    Step 2: Analyzing. In a math problem generation task, effective analysis means the agent can deconstruct old math problems and pre-learned knowledge into their constituent parts, identifying relationships between these parts and the system as a whole. This process involves differentiating, which entails distinguishing the components based on relevance or importance; organizing, which involves recognizing how the elements fit together into a coherent structure; and attributing, which focuses on identifying the underlying point of view, biases, values, or intentions within the information.
    
    Step 3: Levels of Analyzing. 
    - Strong Analyzing (80–100): When the agent successfully breaks down the old problem into its big five components, identifies issues within the components or their relationships, and revises them correctly. The agent must demonstrate at least two of the three behaviors strongly: differentiating, organizing, or attributing.
    - Medium Analyzing (60–80): When the new problem reflects changes to the old one, with most issues or drawbacks corrected, but the agent fails to exhibit any of the three key behaviors.
    - Low Analyzing (<60): When the new problem shows minimal analytical ability, with the main error from the original problem remaining uncorrected if one existed
    
    Step 4: Confidence Score and Suggestion. Reflect on your confidence level in making this judgment and assign a confidence_score between 0 and 100. Provide actionable and specific suggestions to enhance the problem as improvement_suggestions.

    **Result Format:**  
    Provide your evaluation in JSON format with these exact keys:  
    {{
        "performance_score": 0-100,
        "confidence_score": 0-100
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
    Please follow these steps: 
    
    Step 1: Identify "Big Five" Components. Extract these from both problems: 1) math concepts and domains, 2) required skills to solve the problem, 3) math expressions as sequence of operations, 4) values that substitute into expressions, and 5) creative and unique narrative story based on real-life socio-cultural experiences.
    
    Step 2: Evaluating. Compare the big five components in the old and new versions to assess whether the LLM math problem generator effectively evaluates the original problem, judges the importance of each component, and makes informed decisions on their inclusion or exclusion through critical analysis.
    
    Step 3: Levels of Evaluating. 
    - Strong Evaluating (80–100): The math problem generator effectively identifies and eliminates internal inconsistencies or fallacies in the original problem, ensuring the new version adheres to externally established criteria for a well-constructed word problem.
    - Medium Evaluating (60–80): The new problem demonstrates changes based on identifying relevant evaluation criteria, offers some supporting evidence, and reaches a generally logical conclusion, though it may overlook subtle nuances or biases.
    - Low Evaluating(<60): The new problem shows that the agent struggles to apply relevant evaluation criteria, provides weak or irrelevant evidence, and results in a flawed or unsupported conclusion.
    
    Step 4: Confidence Score and Suggestion. Reflect on your confidence level in making this judgment and assign a confidence_score between 0 and 100. Provide actionable and specific suggestions to enhance the problem as improvement_suggestions.

    **Result Format**
    Provide your evaluation in JSON format with these exact keys:  
    {{
        "performance_score": 0-100,
        "confidence_score": 0-100
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
    Please follow these steps: 
    
    Step 1: Identify "Big Five" Components. Extract these from both problems: 1) math concepts and domains, 2) required skills to solve the problem, 3) math expressions as sequence of operations, 4) values that substitute into expressions, and 5) creative and unique narrative story based on real-life socio-cultural experiences.
    
    Step 2: Creating. Focus on the big five components in the new versions to assess whether the LLM math problem generator effectively create a new version of big five components and new relationships between these components. 
    
    Step 3: Levels of Creating. 
    - Strong Creating (80–100): The math problem generator effectively produces a novel, coherent, and functional five big components in the new math problem that effectively fulfills its stated goal. The organizational structure is clear and logical, and the individual elements are well-integrated. The LLM demonstrates originality and ingenuity.
    - Medium Creating (60–80): The new problem demonstrates changes as it produces a generally coherent and functional product, but it may lack originality or have some minor flaws in its organization or integration of elements.
    - Low Creating (<60): The new problem shows that the agent struggles to produce a coherent or functional product. The product may be disorganized, illogical, or fail to meet its stated goal. The LLM demonstrates a lack of originality or significant flaws in its creation process.
    
    Step 4: Confidence Score and Suggestion. Reflect on your confidence level in making this judgment and assign a confidence_score between 0 and 100. Provide actionable and specific suggestions to enhance the problem as improvement_suggestions.

    Provide your evaluation in JSON format with these exact keys:  
    {{
        "performance_score": 0-100,
        "confidence_score": 0-100
    }}
    """,

    "language": """You are an expert evaluator assessing Language Quality in educational question generation.  
    Please evaluate the quality of the following math word problem by analyzing its linguistic and structural features. Identify and categorize any linguistic-level errors (e.g., ambiguity, unanswerability, or linguistic complexity) and assess the problem’s solution strategy. Provide a detailed report that includes quantitative complexity metrics, error classifications, explanations, actionable suggestions for improvement, and a final performance score.
    **Details for Comparison:**  
    - **Previous Problem:** {last_question_details}  
    - **Previous Expected Solution:** {last_question_expected_solution}  
    - **New Problem:** {new_question_details}  
    - **New Expected Solution:** {new_question_expected_solution}  

    Step 1: Lexical and Syntactic Complexity Analysis
    Please calculate the following metrics to assess the math problem's complexity: 
    -Type-Token Ratio (TTR): Measure of lexical diversity.
    -Yngve Score: Evaluates syntactic depth (higher = more complex).
    -Frazier Score: Measures processing load during sentence parsing.
    -Frazier–Roark Score: Combines parsing difficulty and structural complexity.
    -Developmental Level: Assesses language proficiency required to understand the text.
    -Syntactic Frequency: Frequency of common syntactic patterns.
    -Mean Dependency Distance (MDD): Average distance between syntactic dependencies.
    -Sentence Length: Total word count for the problem statement.
    
    Step 2: Error Identification and Classification
    Identify and classify linguistic-level errors using the following categories:
    -Ambiguity:Multiple or unintended solutions due to imprecise descriptions, unclear relationships, or missing conditions.
    -Unanswerability: Incomplete information, ill-defined terms, or logical inconsistencies that prevent a valid solution.
    -Linguistic Complexity: Syntactic complexity, unclear phrasing, or difficult-to-translate statements that may hinder comprehension.
    For each identified error: Clearly classify the error type. Explain the reasoning behind the classification, highlighting how the issue affects clarity or solvability.
    Step 3: Solution Strategy Analysis
    Identify Solution Process Type:

    -One-Step: Requires a single calculation or logical step to reach the answer.
    -Multi-Step: Involves multiple stages of reasoning or calculation.
    -Highlight Comprehension Challenges:Identify areas where multi-step reasoning introduces additional cognitive load or misunderstanding risks.
    
    Step 4: Improvement Suggestions
    Provide actionable and specific suggestions to enhance the problem’s clarity, accuracy, and solvability. Recommendations should address:
    -Ambiguous phrasing (clarify relationships and conditions).
    -Unanswerable problems (add missing information or correct inconsistencies).
    -Linguistic complexity (simplify structure and improve phrasing for better comprehension).
    
    Step 5: Performance Score Calculation (0–100)
    Generate a final performance score considering the following factors:
    -Lexical and Syntactic Complexity (from Step 1):
    -Higher complexity reduces the score.
    -Error Count and Severity (from Step 2):
    -More errors and more severe errors lower the score.
    -Clarity and Solvability: Clear, unambiguous, and well-structured problems receive a higher score.
    Scoring Guidance:
    - 90–100: Clear, simple, and error-free problem.
    - 70–89: Minor complexity or errors that slightly impact clarity.
    - 50–69: Moderate complexity and multiple identifiable issues.
    - 0–49: Significant errors, ambiguity, or unanswerable conditions.

        Provide your evaluation in JSON format with these exact keys:  
        {{
            "performance_score": 0-100,
            "confidence_score": 0-100,
            "improvement_suggestions": ["suggestion1", "suggestion2"]
        }}
        """
    }