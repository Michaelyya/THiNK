from pipeline_GPT import QuestionImprovementPipeline

def test_example():
    # Initialize pipeline
    pipeline = QuestionImprovementPipeline(max_iterations=3, quality_threshold=0.7)
    
    # Test with a specific query
    initial_query = "Generate a basic implicit differentiation question about a circle equation"
    
    print("Starting test with query:", initial_query)
    result = pipeline.run_pipeline(initial_query)
    
    # Print results in a structured way
    if "error" not in result:
        print("\n=== Pipeline Results ===")
        print(f"\nTotal Iterations: {result['iterations_required']}")
        
        print("\nQuestion Evolution:")
        for i, entry in enumerate(result['question_history']):
            print(f"\n--- Iteration {i + 1} ---")
            # print("\nQuestion:")
            # print(entry['question']['question'])
            # print("\nSolution:")
            # print(entry['question']['solution'])
            if entry['evaluation']:
                print(f"\nQuality Score: {entry['quality_score']}")
                print("\nAgent Evaluations:")
                for agent, eval_data in entry['evaluation']['evaluations'].items():
                    print(f"{agent.capitalize()}: Score={eval_data['performance_score']}, Confidence={eval_data['confidence_score']}")
                if entry['evaluation']['improvement_suggestions']:
                    print("\nImprovement Suggestions:")
                    for suggestion in entry['evaluation']['improvement_suggestions']:
                        print(f"- {suggestion}")
    else:
        print("\nError in pipeline:", result['error'])

if __name__ == "__main__":
    test_example()