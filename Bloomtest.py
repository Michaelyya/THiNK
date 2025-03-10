evaluation_criteria = {
    "remembering": """You are an expert evaluator assessing Remembering-Level skills in educational math problem generation. Your goal is to provide highly discriminative scoring that reflects real differences between problems.

Previous Problem:
{last_question_details}

Previous Solution:
{last_question_expected_solution}

New Problem:
{new_question_details}

New Solution:
{new_question_expected_solution}

Evaluation Process:
1. First identify the primary mathematical domains involved:
   - Arithmetic
   - Algebra
   - Geometry
   - Probability
   - Arithmetic and Algebra
   - Algebra and Geometry
   - Algebra and Probability
   - Geometry and Probability
   - Arithmetic, Algebra, and Geometry

2. For each category below, use the FULL range of scores (0-25) following this distribution:
   - 21-25: EXCEPTIONAL (only top 10% of improvements, must provide specific justification)
   - 16-20: STRONG (about 20% of improvements)
   - 11-15: MODERATE (about 40% of improvements)
   - 6-10: MINOR (about 20% of improvements)
   - 0-5: MINIMAL (about 10% of improvements)

Scoring Categories:

1. Fact Recall Improvement (0-25 points):
   - Enhancement in mathematical facts and definitions recall
   - Improvement in formula and theorem memorization
   - Increase in basic concept recognition
   [Must justify scores above 20]

2. Term Recognition (0-25 points):
   - Improvement in mathematical vocabulary usage
   - Enhancement in symbol and notation recognition
   - Progress in basic terminology recall
   [Must justify scores above 20]

3. Procedure Memory (0-25 points):
   - Improvement in remembering mathematical procedures
   - Enhancement in recalling solution steps
   - Progress in method memorization
   [Must justify scores above 20]

4. Basic Concept Retention (0-25 points):
   - Enhancement in fundamental concept recall
   - Improvement in basic principle recognition
   - Progress in mathematical rule remembering
   [Must justify scores above 20]

Confidence Score Categories:
1. Clarity (0-25):
   - How clear are the improvements?
   - How well can you measure the changes?

2. Consistency (0-25):
   - How consistent is the difficulty increase?
   - How well do different parts align?

3. Measurability (0-25):
   - How objectively can you measure the improvements?
   - How clear are the assessment criteria?

4. Alignment (0-25):
   - How well do improvements align with learning objectives?
   - How well do changes match intended skills?

Before providing scores, explicitly state:
1. Specific improvements made in each category
2. Direct comparison between original and new versions
3. Justification for each score, especially those above 20

Provide your evaluation in this exact JSON format:
{
    "mathematical_domains": ["domain1", "domain2"],
    "performance_score": {
        "complexity_increase": <0-25>,
        "conceptual_depth": <0-25>,
        "solution_approach_diversity": <0-25>,
        "real_world_application": <0-25>,
        "total": <sum of above>
    },
    "confidence_score": {
        "clarity": <0-25>,
        "consistency": <0-25>,
        "measurability": <0-25>,
        "alignment": <0-25>,
        "total": <sum of above>
    },
    "score_justification": {
        "complexity_increase": "detailed explanation",
        "conceptual_depth": "detailed explanation",
        "solution_approach_diversity": "detailed explanation",
        "real_world_application": "detailed explanation"
    },
    "improvement_suggestions": [
        "specific suggestion focusing on the lowest scoring category",
        "specific suggestion for the second lowest scoring category"
    ]
}
""",

    "understanding": """You are an expert evaluator assessing Understanding-Level skills in educational math problem generation. Your goal is to provide highly discriminative scoring that reflects real differences between problems.

Previous Problem:
{last_question_details}

Previous Solution:
{last_question_expected_solution}

New Problem:
{new_question_details}

New Solution:
{new_question_expected_solution}

Evaluation Process:
1. First identify the primary mathematical domains involved:
- Arithmetic
   - Algebra
   - Geometry
   - Probability
   - Arithmetic and Algebra
   - Algebra and Geometry
   - Algebra and Probability
   - Geometry and Probability
   - Arithmetic, Algebra, and Geometry

2. For each category below, use the FULL range of scores (0-25) following this distribution:
   - 21-25: EXCEPTIONAL (only top 10% of improvements, must provide specific justification)
   - 16-20: STRONG (about 20% of improvements)
   - 11-15: MODERATE (about 40% of improvements)
   - 6-10: MINOR (about 20% of improvements)
   - 0-5: MINIMAL (about 10% of improvements)

Scoring Categories:

1. Concept Comprehension (0-25 points):
   - Depth of mathematical concept understanding required
   - Quality of explanation requirements
   - Level of interpretation needed
   [Must justify scores above 20]

2. Relationship Understanding (0-25 points):
   - Connections between mathematical ideas
   - Understanding of mathematical relationships
   - Interpretation of mathematical structures
   [Must justify scores above 20]

3. Translation Ability (0-25 points):
   - Converting between mathematical representations
   - Explaining concepts in different ways
   - Interpreting mathematical language
   [Must justify scores above 20]

4. Example Application (0-25 points):
   - Using examples to demonstrate understanding
   - Applying concepts to simple cases
   - Illustrating mathematical ideas
   [Must justify scores above 20]

Confidence Score Categories:
1. Clarity (0-25):
   - How clear are the improvements?
   - How well can you measure the changes?

2. Consistency (0-25):
   - How consistent is the difficulty increase?
   - How well do different parts align?

3. Measurability (0-25):
   - How objectively can you measure the improvements?
   - How clear are the assessment criteria?

4. Alignment (0-25):
   - How well do improvements align with learning objectives?
   - How well do changes match intended skills?

Before providing scores, explicitly state:
1. Specific improvements made in each category
2. Direct comparison between original and new versions
3. Justification for each score, especially those above 20

Provide your evaluation in this exact JSON format:
{
    "mathematical_domains": ["domain1", "domain2"],
    "performance_score": {
        "complexity_increase": <0-25>,
        "conceptual_depth": <0-25>,
        "solution_approach_diversity": <0-25>,
        "real_world_application": <0-25>,
        "total": <sum of above>
    },
    "confidence_score": {
        "clarity": <0-25>,
        "consistency": <0-25>,
        "measurability": <0-25>,
        "alignment": <0-25>,
        "total": <sum of above>
    },
    "score_justification": {
        "complexity_increase": "detailed explanation",
        "conceptual_depth": "detailed explanation",
        "solution_approach_diversity": "detailed explanation",
        "real_world_application": "detailed explanation"
    },
    "improvement_suggestions": [
        "specific suggestion focusing on the lowest scoring category",
        "specific suggestion for the second lowest scoring category"
    ]
}
""",

    "applying": """You are an expert evaluator assessing Applying-Level skills in educational math problem generation. Your goal is to provide highly discriminative scoring that reflects real differences between problems.

Previous Problem:
{last_question_details}

Previous Solution:
{last_question_expected_solution}

New Problem:
{new_question_details}

New Solution:
{new_question_expected_solution}

Evaluation Process:
1. First identify the primary mathematical domains involved:
   - Arithmetic
   - Algebra
   - Geometry
   - Probability
   - Arithmetic and Algebra
   - Algebra and Geometry
   - Algebra and Probability
   - Geometry and Probability
   - Arithmetic, Algebra, and Geometry

2. For each category below, use the FULL range of scores (0-25) following this distribution:
   - 21-25: EXCEPTIONAL (only top 10% of improvements, must provide specific justification)
   - 16-20: STRONG (about 20% of improvements)
   - 11-15: MODERATE (about 40% of improvements)
   - 6-10: MINOR (about 20% of improvements)
   - 0-5: MINIMAL (about 10% of improvements)

Scoring Categories:

1. Application Complexity (0-25 points):
   - Complexity of application scenarios
   - Variety of contexts where concepts are applied
   - Depth of practical implementation
   [Must justify scores above 20]

2. Method Application (0-25 points):
   - Sophistication of mathematical methods applied
   - Integration of multiple techniques
   - Appropriateness of method selection
   [Must justify scores above 20]

3. Problem-Solving Strategy (0-25 points):
   - Effectiveness of problem-solving approaches
   - Strategic thinking requirements
   - Solution planning complexity
   [Must justify scores above 20]

4. Real-World Connection (0-25 points):
   - Relevance to practical situations
   - Authenticity of application context
   - Real-world constraint consideration
   [Must justify scores above 20]

Confidence Score Categories:
1. Clarity (0-25):
   - How clear are the improvements?
   - How well can you measure the changes?

2. Consistency (0-25):
   - How consistent is the difficulty increase?
   - How well do different parts align?

3. Measurability (0-25):
   - How objectively can you measure the improvements?
   - How clear are the assessment criteria?

4. Alignment (0-25):
   - How well do improvements align with learning objectives?
   - How well do changes match intended skills?

Before providing scores, explicitly state:
1. Specific improvements made in each category
2. Direct comparison between original and new versions
3. Justification for each score, especially those above 20

Provide your evaluation in this exact JSON format:
{
    "mathematical_domains": ["domain1", "domain2"],
    "performance_score": {
        "complexity_increase": <0-25>,
        "conceptual_depth": <0-25>,
        "solution_approach_diversity": <0-25>,
        "real_world_application": <0-25>,
        "total": <sum of above>
    },
    "confidence_score": {
        "clarity": <0-25>,
        "consistency": <0-25>,
        "measurability": <0-25>,
        "alignment": <0-25>,
        "total": <sum of above>
    },
    "score_justification": {
        "complexity_increase": "detailed explanation",
        "conceptual_depth": "detailed explanation",
        "solution_approach_diversity": "detailed explanation",
        "real_world_application": "detailed explanation"
    },
    "improvement_suggestions": [
        "specific suggestion focusing on the lowest scoring category",
        "specific suggestion for the second lowest scoring category"
    ]
}
""",

    "analyzing": """You are an expert evaluator assessing Analyzing-Level skills in educational math problem generation. Your goal is to provide highly discriminative scoring that reflects real differences between problems.

Previous Problem:
{last_question_details}

Previous Solution:
{last_question_expected_solution}

New Problem:
{new_question_details}

New Solution:
{new_question_expected_solution}

Evaluation Process:
1. First identify the primary mathematical domains involved:
   - Arithmetic
   - Algebra
   - Geometry
   - Probability
   - Arithmetic and Algebra
   - Algebra and Geometry
   - Algebra and Probability
   - Geometry and Probability
   - Arithmetic, Algebra, and Geometry

2. For each category below, use the FULL range of scores (0-25) following this distribution:
   - 21-25: EXCEPTIONAL (only top 10% of improvements, must provide specific justification)
   - 16-20: STRONG (about 20% of improvements)
   - 11-15: MODERATE (about 40% of improvements)
   - 6-10: MINOR (about 20% of improvements)
   - 0-5: MINIMAL (about 10% of improvements)

Scoring Categories:

1. Complexity Increase (0-25 points):
   - Number of analytical steps added
   - Complexity of relationships between variables
   - Integration of additional mathematical concepts
   [Must justify scores above 20]

2. Conceptual Depth (0-25 points):
   - Depth of mathematical understanding required
   - Connections between different mathematical ideas
   - Level of abstract thinking needed
   [Must justify scores above 20]

3. Solution Approach Diversity (0-25 points):
   - Number of valid solution methods
   - Variety of mathematical tools needed
   - Creativity in problem-solving approaches
   [Must justify scores above 20]

4. Real-world Application (0-25 points):
   - Authenticity of context
   - Relevance to practical situations
   - Integration of real-world constraints
   [Must justify scores above 20]

Confidence Score Categories:
1. Clarity (0-25):
   - How clear are the improvements?
   - How well can you measure the changes?

2. Consistency (0-25):
   - How consistent is the difficulty increase?
   - How well do different parts align?

3. Measurability (0-25):
   - How objectively can you measure the improvements?
   - How clear are the assessment criteria?

4. Alignment (0-25):
   - How well do improvements align with learning objectives?
   - How well do changes match intended skills?

Before providing scores, explicitly state:
1. Specific improvements made in each category
2. Direct comparison between original and new versions
3. Justification for each score, especially those above 20

Provide your evaluation in this exact JSON format:
{
    "mathematical_domains": ["domain1", "domain2"],
    "performance_score": {
        "complexity_increase": <0-25>,
        "conceptual_depth": <0-25>,
        "solution_approach_diversity": <0-25>,
        "real_world_application": <0-25>,
        "total": <sum of above>
    },
    "confidence_score": {
        "clarity": <0-25>,
        "consistency": <0-25>,
        "measurability": <0-25>,
        "alignment": <0-25>,
        "total": <sum of above>
    },
    "score_justification": {
        "complexity_increase": "detailed explanation",
        "conceptual_depth": "detailed explanation",
        "solution_approach_diversity": "detailed explanation",
        "real_world_application": "detailed explanation"
    },
    "improvement_suggestions": [
        "specific suggestion focusing on the lowest scoring category",
        "specific suggestion for the second lowest scoring category"
    ]
}
""",

    "evaluating": """You are an expert evaluator assessing Evaluating-Level skills in educational math problem generation. Your goal is to provide highly discriminative scoring that reflects real differences between problems.

Previous Problem:
{last_question_details}

Previous Solution:
{last_question_expected_solution}

New Problem:
{new_question_details}

New Solution:
{new_question_expected_solution}

Evaluation Process:
1. First identify the primary mathematical domains involved:
   - Arithmetic
   - Algebra
   - Geometry
   - Probability
   - Arithmetic and Algebra
   - Algebra and Geometry
   - Algebra and Probability
   - Geometry and Probability
   - Arithmetic, Algebra, and Geometry

2. For each category below, use the FULL range of scores (0-25) following this distribution:
   - 21-25: EXCEPTIONAL (only top 10% of improvements, must provide specific justification)
   - 16-20: STRONG (about 20% of improvements)
   - 11-15: MODERATE (about 40% of improvements)
   - 6-10: MINOR (about 20% of improvements)
   - 0-5: MINIMAL (about 10% of improvements)

Scoring Categories:

1. Evaluation Depth (0-25 points):
   - Depth of critical analysis required
   - Complexity of evaluation criteria
   - Sophistication of judgment needed
   [Must justify scores above 20]

2. Reasoning Quality (0-25 points):
   - Quality of mathematical reasoning required
   - Depth of logical analysis needed
   - Sophistication of argument construction
   [Must justify scores above 20]

3. Criteria Application (0-25 points):
   - Application of evaluation standards
   - Use of mathematical criteria
   - Assessment framework complexity
   [Must justify scores above 20]

4. Judgment Formation (0-25 points):
   - Development of mathematical judgments
   - Decision-making complexity
   - Conclusion justification requirements
   [Must justify scores above 20]

Confidence Score Categories:
1. Clarity (0-25):
   - How clear are the improvements?
   - How well can you measure the changes?

2. Consistency (0-25):
   - How consistent is the difficulty increase?
   - How well do different parts align?

3. Measurability (0-25):
   - How objectively can you measure the improvements?
   - How clear are the assessment criteria?

4. Alignment (0-25):
   - How well do improvements align with learning objectives?
   - How well do changes match intended skills?

Before providing scores, explicitly state:
1. Specific improvements made in each category
2. Direct comparison between original and new versions
3. Justification for each score, especially those above 20

Provide your evaluation in this exact JSON format:
{
    "mathematical_domains": ["domain1", "domain2"],
    "performance_score": {
        "complexity_increase": <0-25>,
        "conceptual_depth": <0-25>,
        "solution_approach_diversity": <0-25>,
        "real_world_application": <0-25>,
        "total": <sum of above>
    },
    "confidence_score": {
        "clarity": <0-25>,
        "consistency": <0-25>,
        "measurability": <0-25>,
        "alignment": <0-25>,
        "total": <sum of above>
    },
    "score_justification": {
        "complexity_increase": "detailed explanation",
        "conceptual_depth": "detailed explanation",
        "solution_approach_diversity": "detailed explanation",
        "real_world_application": "detailed explanation"
    },
    "improvement_suggestions": [
        "specific suggestion focusing on the lowest scoring category",
        "specific suggestion for the second lowest scoring category"
    ]
}
""",

    "creating": """You are an expert evaluator assessing Creating-Level skills in educational math problem generation. Your goal is to provide highly discriminative scoring that reflects real differences between problems.

Previous Problem:
{last_question_details}

Previous Solution:
{last_question_expected_solution}

New Problem:
{new_question_details}

New Solution:
{new_question_expected_solution}

Evaluation Process:
1. First identify the primary mathematical domains involved:
   - Arithmetic
   - Algebra
   - Geometry
   - Probability
   - Arithmetic and Algebra
   - Algebra and Geometry
   - Algebra and Probability
   - Geometry and Probability
   - Arithmetic, Algebra, and Geometry

2. For each category below, use the FULL range of scores (0-25) following this distribution:
   - 21-25: EXCEPTIONAL (only top 10% of improvements, must provide specific justification)
   - 16-20: STRONG (about 20% of improvements)
   - 11-15: MODERATE (about 40% of improvements)
   - 6-10: MINOR (about 20% of improvements)
   - 0-5: MINIMAL (about 10% of improvements)

Scoring Categories:

1. Innovation Level (0-25 points):
   - Originality of mathematical thinking required
   - Novelty of problem-solving approach
   - Creativity in solution development
   [Must justify scores above 20]

2. Synthesis Complexity (0-25 points):
   - Integration of mathematical concepts
   - Combination of different methods
   - Complexity of concept synthesis
   [Must justify scores above 20]

3. Design Sophistication (0-25 points):
   - Sophistication of solution design
   - Complexity of mathematical construction
   - Elegance of mathematical creation
   [Must justify scores above 20]

4. Original Production (0-25 points):
   - Generation of new mathematical ideas
   - Development of original solutions
   - Creation of mathematical structures
   [Must justify scores above 20]

Confidence Score Categories:
1. Clarity (0-25):
   - How clear are the improvements?
   - How well can you measure the changes?

2. Consistency (0-25):
   - How consistent is the difficulty increase?
   - How well do different parts align?

3. Measurability (0-25):
   - How objectively can you measure the improvements?
   - How clear are the assessment criteria?

4. Alignment (0-25):
   - How well do improvements align with learning objectives?
   - How well do changes match intended skills?

Before providing scores, explicitly state:
1. Specific improvements made in each category
2. Direct comparison between original and new versions
3. Justification for each score, especially those above 20

Provide your evaluation in this exact JSON format:
{
    "mathematical_domains": ["domain1", "domain2"],
    "performance_score": {
        "complexity_increase": <0-25>,
        "conceptual_depth": <0-25>,
        "solution_approach_diversity": <0-25>,
        "real_world_application": <0-25>,
        "total": <sum of above>
    },
    "confidence_score": {
        "clarity": <0-25>,
        "consistency": <0-25>,
        "measurability": <0-25>,
        "alignment": <0-25>,
        "total": <sum of above>
    },
    "score_justification": {
        "complexity_increase": "detailed explanation",
        "conceptual_depth": "detailed explanation",
        "solution_approach_diversity": "detailed explanation",
        "real_world_application": "detailed explanation"
    },
    "improvement_suggestions": [
        "specific suggestion focusing on the lowest scoring category",
        "specific suggestion for the second lowest scoring category"
    ]
}
""",

    "complexity": """You are an expert evaluator assessing the Complexity Level in educational math problem generation. Your goal is to provide highly discriminative scoring that reflects real differences between problems.

Previous Problem:
{last_question_details}

Previous Solution:
{last_question_expected_solution}

New Problem:
{new_question_details}

New Solution:
{new_question_expected_solution}

Evaluation Process:
1. First identify the primary mathematical domains involved:
   - Arithmetic
   - Algebra
   - Geometry
   - Probability
   - Arithmetic and Algebra
   - Algebra and Geometry
   - Algebra and Probability
   - Geometry and Probability
   - Arithmetic, Algebra, and Geometry

2. For each category below, use the FULL range of scores (0-25) following this distribution:
   - 21-25: EXCEPTIONAL (only top 10% of improvements, must provide specific justification)
   - 16-20: STRONG (about 20% of improvements)
   - 11-15: MODERATE (about 40% of improvements)
   - 6-10: MINOR (about 20% of improvements)
   - 0-5: MINIMAL (about 10% of improvements)

Scoring Categories:

1. Computational Complexity (0-25 points):
   - Number of computational steps
   - Difficulty of calculations
   - Algorithmic complexity
   [Must justify scores above 20]

2. Conceptual Complexity (0-25 points):
   - Number of concepts involved
   - Depth of mathematical understanding required
   - Abstraction level
   [Must justify scores above 20]

3. Structural Complexity (0-25 points):
   - Complexity of mathematical structure
   - Interconnection of components
   - Organizational sophistication
   [Must justify scores above 20]

4. Solution Complexity (0-25 points):
   - Number of solution steps
   - Variety of methods required
   - Sophistication of solution process
   [Must justify scores above 20]

[Same Confidence Score Categories and format requirements as remembering]"""
}