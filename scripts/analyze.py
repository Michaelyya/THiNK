import pandas as pd
import os
import numpy as np
from typing import Dict, List, Tuple

def calculate_performance_metrics(df: pd.DataFrame, quality_threshold: float = 0.85) -> Dict[str, float]:
    metrics = {}
    grouped = df.groupby('question_id')
    N = len(grouped)
    
    rounds_to_pass = []
    final_round_quality = []
    
    for question_id, group in grouped:
        group = group.sort_values('round')
        passing_rounds = group[group['quality_score'] > quality_threshold]['round']
        if not passing_rounds.empty:
            rounds_to_pass.append(passing_rounds.iloc[0])
        
        final_round = group.iloc[-1]
        final_round_quality.append(final_round['quality_score'])
    
    if rounds_to_pass:
        metrics['rounds_to_pass'] = np.mean(rounds_to_pass)
    else:
        metrics['rounds_to_pass'] = np.nan
    
    metrics['avg_quality_score'] = np.mean(final_round_quality) * 100
    final_round_pass_rate = np.mean([q > quality_threshold for q in final_round_quality]) * 100
    metrics['avg_pass_rate'] = final_round_pass_rate
    
    return metrics

def analyze_results(input_file: str) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    df = pd.read_csv(input_file)
    
    lower_order = ["remembering_score", "understanding_score", "applying_score"]
    higher_order = ["analyzing_score", "evaluating_score", "creating_score"]
    
    final_rounds = df.groupby('question_id')['round'].max()
    
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
    
    for question_id in final_rounds.index:
        final_round = final_rounds[question_id]
        previous_round = final_round - 1
        
        if previous_round > 0:
            final_row = df[(df['question_id'] == question_id) & (df['round'] == final_round)].iloc[0]
            previous_row = df[(df['question_id'] == question_id) & (df['round'] == previous_round)].iloc[0]
            
            cognitive_improvements["remembering"].append(final_row["remembering_score"] - previous_row["remembering_score"])
            cognitive_improvements["understanding"].append(final_row["understanding_score"] - previous_row["understanding_score"])
            cognitive_improvements["applying"].append(final_row["applying_score"] - previous_row["applying_score"])
            cognitive_improvements["analyzing"].append(final_row["analyzing_score"] - previous_row["analyzing_score"])
            cognitive_improvements["evaluating"].append(final_row["evaluating_score"] - previous_row["evaluating_score"])
            cognitive_improvements["creating"].append(final_row["creating_score"] - previous_row["creating_score"])
            cognitive_improvements["quality"].append(final_row["quality_score"] - previous_row["quality_score"])
            
            cognitive_scores["remembering"] += final_row["remembering_score"]
            cognitive_scores["understanding"] += final_row["understanding_score"]
            cognitive_scores["applying"] += final_row["applying_score"]
            cognitive_scores["analyzing"] += final_row["analyzing_score"]
            cognitive_scores["evaluating"] += final_row["evaluating_score"]
            cognitive_scores["creating"] += final_row["creating_score"]
            
            lower_scores = sum(final_row[score] for score in lower_order)
            cognitive_scores["lower_order"] += lower_scores
            
            higher_scores = sum(final_row[score] for score in higher_order)
            cognitive_scores["higher_order"] += higher_scores
    
    total_questions = len(final_rounds)
    for key in cognitive_scores:
        cognitive_scores[key] /= total_questions
    
    avg_improvements = {}
    for key in cognitive_improvements:
        if cognitive_improvements[key]:
            avg_improvements[key] = sum(cognitive_improvements[key]) / len(cognitive_improvements[key])
        else:
            avg_improvements[key] = 0.0
    
    performance_metrics = calculate_performance_metrics(df)
    
    return cognitive_scores, avg_improvements, performance_metrics

def analyze_baseline_results(input_file: str) -> Tuple[Dict[str, float], Dict[str, float]]:
    df = pd.read_csv(input_file)
    
    cognitive_scores = {
        "remembering": 0.0,
        "understanding": 0.0,
        "applying": 0.0,
        "analyzing": 0.0,
        "evaluating": 0.0,
        "creating": 0.0
    }
    
    cognitive_improvements = {
        "remembering": [],
        "understanding": [],
        "applying": [],
        "analyzing": [],
        "evaluating": [],
        "creating": []
    }
    
    for _, row in df.iterrows():
        cognitive_scores["remembering"] += row["remembering_score"]
        cognitive_scores["understanding"] += row["understanding_score"]
        cognitive_scores["applying"] += row["applying_score"]
        cognitive_scores["analyzing"] += row["analyzing_score"]
        cognitive_scores["evaluating"] += row["evaluating_score"]
        cognitive_scores["creating"] += row["creating_score"]
        
        improvement = row["quality_score"]
        cognitive_improvements["remembering"].append(improvement)
        cognitive_improvements["understanding"].append(improvement)
        cognitive_improvements["applying"].append(improvement)
        cognitive_improvements["analyzing"].append(improvement)
        cognitive_improvements["evaluating"].append(improvement)
        cognitive_improvements["creating"].append(improvement)
    
    total_questions = len(df)
    for key in cognitive_scores:
        cognitive_scores[key] = (cognitive_scores[key] / total_questions)
    
    avg_improvements = {}
    for key in cognitive_improvements:
        if cognitive_improvements[key]:
            avg_improvements[key] = (sum(cognitive_improvements[key]) / len(cognitive_improvements[key])) * 100
        else:
            avg_improvements[key] = 0.0
    
    return cognitive_scores, avg_improvements

def generate_latex_table(results: Dict[str, Dict[str, float]]) -> str:
    latex = "\\begin{table*}[h!]\n\\centering\n\\resizebox{\\textwidth}{!}{%\n\\begin{tabular}{lccccccc}\n\\hline\n"
    latex += "\\textbf{Model} & \\textbf{Remembering} & \\textbf{Understanding} & \\textbf{Applying} & \\textbf{Analyzing} & \\textbf{Evaluating} & \\textbf{Creating} & \\textbf{Avg.} \\\\ \\hline\n"
    
    for model_name, model_results in results.items():
        scores = model_results['scores']
        improvements = model_results['improvements']
        avg_score = sum(scores.values()) / len(scores)
        
        row = f"{model_name} & "
        for level in ['remembering', 'understanding', 'applying', 'analyzing', 'evaluating', 'creating']:
            score = scores[level]
            imp = improvements[level]
            color = "green" if imp >= 0 else "red"
            arrow = "\\uparrow" if imp >= 0 else "\\downarrow"
            row += f"{score:.2f} \\textcolor{{{color}}}{{$\\{arrow}$ {abs(imp):.2f}}} & "
        row += f"{avg_score:.2f} \\\\ \\hline\n"
        latex += row
    
    latex += "\\end{tabular}\n}\n\\caption{Cognitive Level Performance Across Models}\n\\label{tab:cognitive_performance}\n\\end{table*}"
    return latex

def main():
    # Create results directory if it doesn't exist
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Get all baseline CSV files in the results directory
    csv_files = [f for f in os.listdir(results_dir) if f.startswith('baseline_') and f.endswith('.csv')]
    
    # Store results for each model
    results = {}
    
    # Analyze each file
    for file in csv_files:
        model_name = file.replace('baseline_', '').replace('.csv', '')
        input_file = os.path.join(results_dir, file)
        
        try:
            cognitive_scores, improvements = analyze_baseline_results(input_file)
            results[model_name] = {
                'scores': cognitive_scores,
                'improvements': improvements
            }
            
            print(f"\nResults for {model_name}:")
            print("Individual Cognitive Levels:")
            for level in cognitive_scores:
                print(f"  {level.capitalize()}: {cognitive_scores[level]:.2f} (Improvement: {improvements[level]:.2f})")
            
        except Exception as e:
            print(f"Error processing {file}: {str(e)}")
    
    # Generate and print LaTeX table
    latex_table = generate_latex_table(results)
    print("\nLaTeX Table:")
    print(latex_table)

if __name__ == "__main__":
    main() 