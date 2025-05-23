import pandas as pd
import os
import numpy as np
from typing import Dict, List, Tuple

def calculate_performance_metrics(df: pd.DataFrame, quality_threshold: float = 0.85) -> Dict[str, float]:
    """
    Calculate performance metrics for the model.
    
    Args:
        df (pd.DataFrame): DataFrame containing the results
        quality_threshold (float): Quality threshold for pass rate (default: 0.85)
        
    Returns:
        Dict[str, float]: Dictionary containing the performance metrics
    """
    metrics = {}
    
    # Group by question_id to analyze each problem
    grouped = df.groupby('question_id')
    N = len(grouped)  # Number of problems
    
    # Calculate RoundsToPass for each problem
    rounds_to_pass = []
    final_round_quality = []
    
    for question_id, group in grouped:
        # Sort by round to ensure correct order
        group = group.sort_values('round')
        
        # Find first round where quality exceeds threshold
        passing_rounds = group[group['quality_score'] > quality_threshold]['round']
        if not passing_rounds.empty:
            rounds_to_pass.append(passing_rounds.iloc[0])
        
        # Get quality score of final round
        final_round = group.iloc[-1]
        final_round_quality.append(final_round['quality_score'])
    
    # Calculate final metrics
    if rounds_to_pass:  # Only calculate if we have any passing rounds
        metrics['rounds_to_pass'] = np.mean(rounds_to_pass)
    else:
        metrics['rounds_to_pass'] = np.nan
    
    # Calculate average quality score from final rounds only (scale to percentage)
    metrics['avg_quality_score'] = np.mean(final_round_quality) * 100
    
    # Calculate pass rate based on final rounds only
    final_round_pass_rate = np.mean([q > quality_threshold for q in final_round_quality]) * 100
    metrics['avg_pass_rate'] = final_round_pass_rate
    
    return metrics

def analyze_results(input_file: str) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    """
    Analyze the results from a CSV file and calculate cognitive level scores, improvements, and performance metrics.
    
    Args:
        input_file (str): Path to the input CSV file
        
    Returns:
        Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]: A tuple containing:
            - Dictionary with cognitive level scores
            - Dictionary with cognitive level improvements
            - Dictionary with performance metrics
    """
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    # Define cognitive levels
    lower_order = ["remembering_score", "understanding_score", "applying_score"]
    higher_order = ["analyzing_score", "evaluating_score", "creating_score"]
    
    # Get the final round for each question_id
    final_rounds = df.groupby('question_id')['round'].max()
    
    # Initialize results dictionaries
    cognitive_scores = {
        "remembering": 0.0,
        "understanding": 0.0,
        "applying": 0.0,
        "analyzing": 0.0,
        "evaluating": 0.0,
        "creating": 0.0,
        "lower_order": 0.0,
        "higher_order": 0.0
    }
    
    cognitive_improvements = {
        "remembering": [],
        "understanding": [],
        "applying": [],
        "analyzing": [],
        "evaluating": [],
        "creating": [],
        "quality": []
    }
    
    # Calculate cognitive level scores and improvements
    for question_id in final_rounds.index:
        final_round = final_rounds[question_id]
        previous_round = final_round - 1
        
        if previous_round > 0:  # Only calculate if there's a previous round
            final_row = df[(df['question_id'] == question_id) & (df['round'] == final_round)].iloc[0]
            previous_row = df[(df['question_id'] == question_id) & (df['round'] == previous_round)].iloc[0]
            
            # Calculate improvements
            cognitive_improvements["remembering"].append(final_row["remembering_score"] - previous_row["remembering_score"])
            cognitive_improvements["understanding"].append(final_row["understanding_score"] - previous_row["understanding_score"])
            cognitive_improvements["applying"].append(final_row["applying_score"] - previous_row["applying_score"])
            cognitive_improvements["analyzing"].append(final_row["analyzing_score"] - previous_row["analyzing_score"])
            cognitive_improvements["evaluating"].append(final_row["evaluating_score"] - previous_row["evaluating_score"])
            cognitive_improvements["creating"].append(final_row["creating_score"] - previous_row["creating_score"])
            cognitive_improvements["quality"].append(final_row["quality_score"] - previous_row["quality_score"])
            
            # Calculate final scores
            cognitive_scores["remembering"] += final_row["remembering_score"]
            cognitive_scores["understanding"] += final_row["understanding_score"]
            cognitive_scores["applying"] += final_row["applying_score"]
            cognitive_scores["analyzing"] += final_row["analyzing_score"]
            cognitive_scores["evaluating"] += final_row["evaluating_score"]
            cognitive_scores["creating"] += final_row["creating_score"]
            
            # Calculate lower and higher order scores
            lower_scores = sum(final_row[score] for score in lower_order)
            cognitive_scores["lower_order"] += lower_scores
            
            higher_scores = sum(final_row[score] for score in higher_order)
            cognitive_scores["higher_order"] += higher_scores
    
    # Calculate averages for scores
    total_questions = len(final_rounds)
    for key in cognitive_scores:
        cognitive_scores[key] /= total_questions
    
    # Calculate averages for improvements
    avg_improvements = {}
    for key in cognitive_improvements:
        if cognitive_improvements[key]:  # Check if there are any improvements calculated
            avg_improvements[key] = sum(cognitive_improvements[key]) / len(cognitive_improvements[key])
        else:
            avg_improvements[key] = 0.0
    
    # Calculate performance metrics
    performance_metrics = calculate_performance_metrics(df)
    
    return cognitive_scores, avg_improvements, performance_metrics

def main():
    # Create results directory if it doesn't exist
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Get all CSV files in the results directory
    csv_files = [f for f in os.listdir(results_dir) if f.endswith('.csv')]
    
    # Create a DataFrame to store all results
    results_df = pd.DataFrame(columns=[
        'model',
        'remembering',
        'understanding',
        'applying',
        'analyzing',
        'evaluating',
        'creating',
        'lower_order_thinking',
        'higher_order_thinking',
        'remembering_improvement',
        'understanding_improvement',
        'applying_improvement',
        'analyzing_improvement',
        'evaluating_improvement',
        'creating_improvement',
        'quality_improvement',
        'rounds_to_pass',
        'avg_quality_score',
        'avg_pass_rate'
    ])
    
    # Analyze each file
    for file in csv_files:
        if file == 'analysis_results.csv':  # Skip the results file itself
            continue
            
        model_name = file.replace('.csv', '')
        input_file = os.path.join(results_dir, file)
        
        try:
            cognitive_scores, improvements, performance = analyze_results(input_file)
            
            # Add results to DataFrame
            results_df = pd.concat([results_df, pd.DataFrame([{
                'model': model_name,
                'remembering': cognitive_scores['remembering'],
                'understanding': cognitive_scores['understanding'],
                'applying': cognitive_scores['applying'],
                'analyzing': cognitive_scores['analyzing'],
                'evaluating': cognitive_scores['evaluating'],
                'creating': cognitive_scores['creating'],
                'lower_order_thinking': cognitive_scores['lower_order'],
                'higher_order_thinking': cognitive_scores['higher_order'],
                'remembering_improvement': improvements['remembering'],
                'understanding_improvement': improvements['understanding'],
                'applying_improvement': improvements['applying'],
                'analyzing_improvement': improvements['analyzing'],
                'evaluating_improvement': improvements['evaluating'],
                'creating_improvement': improvements['creating'],
                'quality_improvement': improvements['quality'],
                'rounds_to_pass': performance['rounds_to_pass'],
                'avg_quality_score': performance['avg_quality_score'],
                'avg_pass_rate': performance['avg_pass_rate']
            }])], ignore_index=True)
            
            print(f"\nResults for {model_name}:")
            print(f"Individual Cognitive Levels:")
            print(f"  Remembering: {cognitive_scores['remembering']:.2f} (Improvement: {improvements['remembering']:.2f})")
            print(f"  Understanding: {cognitive_scores['understanding']:.2f} (Improvement: {improvements['understanding']:.2f})")
            print(f"  Applying: {cognitive_scores['applying']:.2f} (Improvement: {improvements['applying']:.2f})")
            print(f"  Analyzing: {cognitive_scores['analyzing']:.2f} (Improvement: {improvements['analyzing']:.2f})")
            print(f"  Evaluating: {cognitive_scores['evaluating']:.2f} (Improvement: {improvements['evaluating']:.2f})")
            print(f"  Creating: {cognitive_scores['creating']:.2f} (Improvement: {improvements['creating']:.2f})")
            print(f"Lower Order Thinking Average: {cognitive_scores['lower_order']:.2f}")
            print(f"Higher Order Thinking Average: {cognitive_scores['higher_order']:.2f}")
            print(f"Average Quality Improvement: {improvements['quality']:.2f}")
            print(f"\nPerformance Metrics:")
            print(f"  Rounds to Pass: {performance['rounds_to_pass']:.2f}")
            print(f"  Average Quality Score: {performance['avg_quality_score']:.2f}%")
            print(f"  Average Pass Rate: {performance['avg_pass_rate']:.2f}%")
            
        except Exception as e:
            print(f"Error processing {file}: {str(e)}")
    
    # Save results to CSV
    output_file = os.path.join(results_dir, 'analysis_results.csv')
    results_df.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    main() 