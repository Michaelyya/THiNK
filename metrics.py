from typing import Dict, Any, List
from sklearn.metrics import cohen_kappa_score
from statsmodels.stats.inter_rater import fleiss_kappa
import numpy as np
from itertools import combinations

def calculate_pass_rate(evaluations: Dict[str, Any]) -> float:
    passing_scores = [
        eval_data['performance_score'] >= 70
        for eval_data in evaluations.values()
    ]
    return sum(passing_scores) / len(passing_scores)

def calculate_agent_agreement(evaluations: Dict[str, Any]) -> float:
    """!!!!I DOUBT THIS IS INCORRECT WAY TO IMPLEMENT!!!!"""
   # Convert scores to binary decisions (pass/fail)
    agent_decisions = {
        agent: 1 if eval_data['performance_score'] >= 70 else 0
        for agent, eval_data in evaluations.items()
    }
    
    # Create a matrix for Fleiss' Kappa calculation
    n_categories = 2  # pass/fail
    n_raters = len(agent_decisions)
    ratings = list(agent_decisions.values())
    
    # Fleiss' Kappa requires a matrix where rows are items, columns are categories
    matrix = np.zeros((1, n_categories))
    matrix[0, 0] = ratings.count(0)  # count of fails
    matrix[0, 1] = ratings.count(1)  # count of passes
    
    # Compute Fleiss' Kappa
    fleiss_k = fleiss_kappa(matrix)
    
    # Compute Cohen’s Kappa for all pairs
    cohen_kappas = []
    agents = list(agent_decisions.keys())
    
    for agent1, agent2 in combinations(agents, 2):
        rater1 = [agent_decisions[agent1]] * 2  
        rater2 = [agent_decisions[agent2]] * 2 
        k = cohen_kappa_score(rater1, rater2)
        if not np.isnan(k):
            cohen_kappas.append(k)
    
    avg_cohen_kappa = np.mean(cohen_kappas) if cohen_kappas else 0.0
    
    # Take the average of Fleiss' and mean Cohen's Kappa
    final_kappa = np.mean([fleiss_k, avg_cohen_kappa])
    
    # Normalize to 0-1 range (kappa can be negative)
    return max(0, min(1, (final_kappa + 1) / 2))
        

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
    # Sample evaluation data
    sample_evaluations = {
        "remembering": {
            "performance_score": 80,
            "confidence_score": 50
        },
        "understanding": {
            "performance_score": 80,
            "confidence_score": 85
        },
        "applying": {
            "performance_score": 80,
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