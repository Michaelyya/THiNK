from typing import Dict, Any, List
from sklearn.metrics import cohen_kappa_score
from statsmodels.stats.inter_rater import fleiss_kappa
import numpy as np
from itertools import combinations
from collections import Counter

# Cohen Kappa = (P0- Pe)/(1-Pe) 
# -> P0 is the proportion of observed agreement, or the number of times two raters agreed on a classification 
# -> Pe is the probability of random agreement, or the number of times two raters would agree by chance. 
# For multi-rater kappa, we calculate Feiss' Kappa or Krippendorff's Alpha， Feiss' Kappa is better for classified variable
# How many subjects to be labeled? How many categories of labeling? How many raters? 

def calculate_pass_rate(evaluations: Dict[str, Any]) -> float:
    passing_scores = [
        eval_data['performance_score'] >= 85
        for eval_data in evaluations.values()
    ]
    return sum(passing_scores) / len(passing_scores)

def calculate_agent_agreement(evaluations: Dict[str, Any]) -> float:
   # Convert scores to binary decisions (pass/fail)
    agent_decisions = {
        agent: 1 if eval_data['performance_score'] >= 85 else 0
        for agent, eval_data in evaluations.items()
    }
    

    n_categories = 2  # pass/fail
    n_raters = len(agent_decisions)
    ratings = list(agent_decisions.values())
    # Create a matrix for Fleiss' Kap fipa calculation
    # pi is the agreement rate for each item to be labeled 
    # po is overall agreement rate for all items 
    # pj is the chance agreement that computes the probability of each category occurring across the entire sample. 
    # Count how many times each category appears 
    category_counts = Counter(ratings)
    # Calculate p_i (observed agreement for the item)
    sum_k = 0 
    for count in category_counts.values(): 
        sum_k += count*(count - 1)
        
    p_i = sum_k/(n_raters*(n_raters - 1))
    # When we have more than one item being labeled, later we can calculate Fleiss' Kappa for the whole matrix. 
    return p_i

def calculate_average_confidence(evaluations: Dict[str, Any]) -> float:
    confidence_scores = [
        eval_data['confidence_score'] / 100 
        for eval_data in evaluations.values()
    ]
    return sum(confidence_scores) / len(confidence_scores)

def calculate_quality_score(evaluations: Dict[str, Any]) -> float:
    # Quality Score = (0.5 × Pass Rate) + (0.3 × Agent Agreement) + (0.2 × Average Confidence)
    pass_rate = calculate_pass_rate(evaluations)
    agent_agreement = calculate_agent_agreement(evaluations)
    avg_confidence = calculate_average_confidence(evaluations)
    quality_score = (0.5 * pass_rate) + (0.3 * agent_agreement) + (0.2 * avg_confidence)
    return round(quality_score, 3)

def get_detailed_metrics(evaluations: Dict[str, Any]) -> Dict[str, float]:
    pass_rate = calculate_pass_rate(evaluations)
    agent_agreement = calculate_agent_agreement(evaluations)
    avg_confidence = calculate_average_confidence(evaluations)
    quality_score = calculate_quality_score(evaluations)
    
    return {
        "pass_rate": pass_rate,
        "agent_agreement": agent_agreement,
        "average_confidence": avg_confidence,
        "quality_score": quality_score
    }

# Example usage
if __name__ == "__main__":
    sample_evaluations = {
        "remembering": {
            "performance_score": 10,
            "confidence_score": 50
        },
        "understanding": {
            "performance_score": 20,
            "confidence_score": 85
        },
        "applying": {
            "performance_score": 20,
            "confidence_score": 50
        }
        #... we will have 7 agents here!
    }
    
    metrics = get_detailed_metrics(sample_evaluations)
    print("\nDetailed Metrics:")
    print(f"Pass Rate: {metrics['pass_rate']:.3f}")
    print(f"Agent Agreement: {metrics['agent_agreement']:.3f}")
    print(f"Average Confidence: {metrics['average_confidence']:.3f}")
    print(f"Quality Score: {metrics['quality_score']:.3f}")