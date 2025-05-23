# 🧠 Educational Question Improvement Framework

An intelligent framework for improving educational questions using Large Language Models (LLMs). This system evaluates and enhances mathematical questions based on cognitive levels and educational quality metrics.

## 🌟 Features

- **Multi-Model Support**: Works with both GPT and open-source LLM models
- **Cognitive Level Analysis**: Evaluates questions across 7 cognitive levels:
  - Remembering
  - Understanding
  - Applying
  - Analyzing
  - Evaluating
  - Creating
  - Language
- **Quality Metrics**: Comprehensive evaluation including:
  - Pass rate
  - Agent agreement
  - Average confidence
  - Quality score
- **Iterative Improvement**: Automatically improves questions based on evaluation feedback
- **Analysis Tools**: Generates detailed performance reports and LaTeX tables

## 📋 Prerequisites

- Python 3.8+
- OpenAI API key (for GPT model)
- Required Python packages:
  ```bash
  pip install -r requirements.txt
  ```

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

## 📁 Project Structure

```
.
├── config.json              # Configuration file
├── run.py                   # Main execution script
├── requirements.txt         # Python dependencies
└── scripts/
    ├── agents.py           # Agent definitions and evaluation logic
    ├── analyze.py          # Analysis and reporting tools
    ├── metrics.py          # Quality metrics calculations
    ├── pipeline_GPT_bad.py # GPT model pipeline
    └── agent_prompts.py    # Agent prompt templates
```

## 📈 Analysis

The framework provides comprehensive analysis of question quality:
- Cognitive level performance
- Quality score progression
- Agent agreement metrics
- Improvement suggestions

Results are available in both JSON and CSV formats, with LaTeX table generation for academic papers.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- OpenAI for providing the GPT API
- The educational research community for cognitive level frameworks
- Contributors and users of this project

## 📧 Contact

For questions and support, please open an issue in the repository.
