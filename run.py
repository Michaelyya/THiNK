import argparse
import json
import os
import sys
from dotenv import load_dotenv

# Add scripts directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

from pipeline_GPT_bad import run_bad_questions_evaluation
from analyze import main as run_analysis

def load_config():
    with open('config.json', 'r') as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser(description='Run the question improvement pipeline')
    parser.add_argument('--model', type=str, required=True, choices=['gpt', 'open_source'],
                      help='Model to use (gpt or open_source)')
    parser.add_argument('--api_key', type=str, required=True,
                      help='API key for the model')
    parser.add_argument('--num_questions', type=int, default=120,
                      help='Number of questions to process (default: 120)')
    parser.add_argument('--max_iterations', type=int, default=3,
                      help='Maximum number of iterations per question (default: 3)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config()
    model_config = config['models'][args.model]
    
    # Set up environment variables
    os.environ['OPENAI_API_KEY'] = args.api_key
    os.environ['OPENAI_BASE_URL'] = model_config['base_url']
    
    print(f"\n{'='*50}")
    print(f"Running pipeline with {args.model} model")
    print(f"{'='*50}\n")
    
    # Run the pipeline
    run_bad_questions_evaluation(
        num_questions=args.num_questions,
        max_iterations=args.max_iterations
    )
    
    print(f"\n{'='*50}")
    print("Running analysis")
    print(f"{'='*50}\n")
    
    # Run analysis
    run_analysis()
    
    print(f"\n{'='*50}")
    print("Pipeline completed successfully!")
    print(f"{'='*50}\n")

if __name__ == "__main__":
    main() 