# 🧠 THiNK - Can Large Language Models Think-Aloud?
We propose **THiNK** (**T**esting **Hi**gher-order **N**otion of **K**nowledge), a multi-agent, feedback-driven evaluation framework grounded in *Bloom’s Taxonomy*. **THiNK** frames reasoning assessment as an iterative task of problem generation, critique, and revision, encouraging LLMs to *think-aloud* through step-by-step reflection and refinement.


<div align="center">
    <img src="./pic/Think_pipeline" alt="Link to PDF" height="auto" style="width:95%;">
</div>

## 📋 Prerequisites
- OpenAI API key (for GPT model)
- HuggingFace API key (for Open-source model)
- Your own tested model 

## 🚀 Quick Start

1. **Clone the repository**:
   ```bash
   git clone [your-repo-url]
   cd [your-repo-name]
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure the model**:
   Edit `config.json` to set your preferred model parameters:
   ```json
   {
       "models": {
           "gpt": {
               "name": "o1-mini",
               "base_url": "your-api-base-url",
               "temperature": 0,
               "max_iterations": 3,
               "quality_threshold": 0.7
           }
       }
   }
   ```

4. **Run the pipeline**:
   ```bash
   python run.py --model gpt --api_key your_api_key --num_questions 120 --max_iterations 3
   ```

## 📊 Output Files

The pipeline generates several output files:
- `bad_questions_evaluation_results.json`: Detailed evaluation results
- `round_metrics.csv`: Metrics for each iteration
- `results/cognitive_performance_table.tex`: LaTeX table of cognitive performance

## 🛠️ Usage

### Basic Usage
```bash
python run.py --model [gpt|open_source] --api_key YOUR_API_KEY
```

### Advanced Options
```bash
python run.py --model gpt \
              --api_key YOUR_API_KEY \
              --num_questions 50 \
              --max_iterations 5
```

### Parameters
- `--model`: Choose between 'gpt' or 'open_source'
- `--api_key`: Your API key for the selected model
- `--num_questions`: Number of questions to process (default: 120)
- `--max_iterations`: Maximum iterations per question (default: 3)


## 📈 Analysis

The framework provides a comprehensive analysis of question quality:
- Cognitive level performance
- Quality score progression
- Agent agreement metrics
- Improvement suggestions

Results are available in both JSON and CSV formats, with LaTeX table generation for academic papers.

## 🤝 Reference
This is an under-review anonymous GitHub conference page
