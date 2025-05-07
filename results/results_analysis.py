import pandas as pd
import numpy as np
from scipy.stats import spearmanr

#df = pd.read_csv("/Users/mqwu/Desktop/Writing&Reading/MathPrbGen_MultiAgent/Results/meta-llama_Meta-Llama-3.1-8B-Instruct_metrics.csv")
#df = pd.read_csv("/Users/mqwu/Desktop/Writing&Reading/MathPrbGen_MultiAgent/Results/mistralai_Ministral-8B-Instruct-2410_metrics.csv")
df = pd.read_csv("/Users/mqwu/Desktop/Writing&Reading/MathPrbGen_MultiAgent/Results/gpt-3.5-turbo.csv")

metrics = [
    'average_confidence', 'pass_rate', 'agent_agreement', 'quality_score',
    'remembering_score', 'understanding_score', 'applying_score', 'analyzing_score',
    'evaluating_score', 'creating_score', 'language_score'
]

# Ensure metric columns are numeric
for col in metrics:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Function to compute summary for each group
def summarize_question(group):

    summary = {}

    # Sort by round index
    group_sorted = group.sort_values(by='round')

    for col in metrics:
        values = group_sorted[col].values

        # Mean and SD
        summary[f'{col}_mean'] = np.mean(values)
        summary[f'{col}_sd'] = np.std(values)


    # Score improvement (last - first) for remembering_score
    summary['remembering_score_improvement'] = group_sorted['remembering_score'].iloc[-1] - group_sorted['remembering_score'].iloc[0]
    summary['understanding_score_improvement'] = group_sorted['understanding_score'].iloc[-1] - group_sorted['understanding_score'].iloc[0]
    summary['applying_score_improvement'] = group_sorted['applying_score'].iloc[-1] - group_sorted['applying_score'].iloc[0]
    summary['analyzing_score_improvement'] = group_sorted['analyzing_score'].iloc[-1] - group_sorted['analyzing_score'].iloc[0]
    summary['evaluating_score_improvement'] = group_sorted['evaluating_score'].iloc[-1] - group_sorted['evaluating_score'].iloc[0]
    summary['creating_score_improvement'] = group_sorted['creating_score'].iloc[-1] - group_sorted['creating_score'].iloc[0]
    summary['language_score_improvement'] = group_sorted['language_score'].iloc[-1] - group_sorted['language_score'].iloc[0]


    return pd.Series(summary)


# Grouping with updated behavior
summary_df = (
    df.drop(columns=["question_id"])
      .groupby(df["question_id"])
      .apply(summarize_question)
)

# Merge back to original df
df_merged = df.merge(summary_df, on='question_id', how='right')
#df_merged.to_csv("/Users/mqwu/Desktop/Writing&Reading/MathPrbGen_MultiAgent/Results/merge_meta-llama_Meta-Llama-3.1-8B-Instruct_metrics.csv")
df_merged.to_csv("/Users/mqwu/Desktop/Writing&Reading/MathPrbGen_MultiAgent/Results/merge_gpt-3.5-turbo.csv")

# merge_meta-llama_Meta-Llama-3.1-8B-Instruct_metrics.csv -> error rate: 98/327 = 0.29969
#