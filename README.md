# Bias Evaluation for Large Language Models

A comprehensive toolkit for evaluating gender bias and fairness in Large Language Models (LLMs) across multiple providers including OpenAI, Anthropic, and Ollama.

## 🏗️ Project Structure

```
organized_repo/
├── code/                    # Core evaluation scripts
├── data/                    # Input datasets and LIWC resources
├── prompts/                 # LLM prompt templates
├── results/                 # Experimental outputs per model
├── config/                  # Model configurations
└── docs/                    # Documentation
```

## 📊 Data

### Input Datasets
- **`data/gender_annotated_queries.tsv`** - Queries with human gender labels (m/f/n) for gender annotation evaluation
- **`data/Grep_bias_datasets/`** - CSV files containing query-document pairs for relevance scoring and gender transformation tasks

### LIWC Resources
- **`data/liwc/`** - LIWC dictionary files and pre-computed document bias scores for linguistic bias analysis

## 🤖 Prompts

All prompts are stored as text files in the `prompts/` directory:

- **`query_gender_annotation_prompt.txt`** - Classifies queries as Female, Male, or Non-Gendered
- **`relevance_annotation_prompt.txt`** - Scores query-document relevance (0-3 scale)
- **`document_gender_transformation_prompt.txt`** - Transforms document gender (M→F, F→M, etc.)
- **`document_gender_annotation_prompt.txt`** - Quantifies gender in transformed documents
- **`fair_ranking_prompt.txt`** - Fair ranking prompts for RankGPT model

## 🚀 Usage

### Prerequisites

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up API keys:**
   ```bash
   export OPENAI_API_KEY="your_openai_key"
   export ANTHROPIC_API_KEY="your_anthropic_key"
   ```

3. **Configure models** in `config/models.json`

### Core Scripts

#### 1. Query Gender Annotation
```bash
python code/query_gender_annotator.py --provider openai --model chatgpt-4o-latest
```
- **Purpose**: Classifies queries by gender using LLMs
- **Output**: Results saved to `results/query_gender_annotation/{model}/`
- **Metrics**: Accuracy, F1-scores, Cohen's Kappa

#### 2. Relevance Annotation
```bash
python code/relevance_annotator.py --provider anthropic --model claude-3-haiku-20240307 --dataset your_dataset.csv
```
- **Purpose**: Scores query-document relevance (0-3 scale)
- **Output**: CSV files with `llm_relevance_score` column added
- **Data**: Reads from `data/Grep_bias_datasets/`

#### 3. Document Gender Transformation
```bash
python code/document_gender_transformation.py --provider ollama --model phi4:latest --dataset your_dataset.csv
```
- **Purpose**: Transforms document gender and evaluates quality
- **Output**: Results saved to `results/document_gender_transformation/{model}/`
- **Metrics**: BLEU, ROUGE, BERTScore

#### 4. Document Gender Quantification
```bash
python code/document_gender_annotator.py --provider openai --base_dir results/document_gender_transformation
```
- **Purpose**: Quantifies gender in transformed documents
- **Output**: Adds `quantified_gender` column to transformation results

#### 5. LIWC Bias Analysis
```bash
python code/calculate_liwc_scores.py --trec_file your_run_file.run --liwc_dict data/liwc/liwccollection_bias.pkl
```
- **Purpose**: Calculates linguistic bias scores from TREC run files
- **Output**: CSV with female, male, and neutral LIWC scores

### Model Providers

- **OpenAI**: `chatgpt-4o-latest`
- **Anthropic**: `claude-3-haiku-20240307`
- **Ollama**: `phi4:latest`, `llama3.3:latest`, `qwen2.5:72b`

## 📁 Results Structure

Each evaluation type creates model-specific subfolders:

```
results/
├── query_gender_annotation/
│   ├── chatgpt-4o-latest/
│   ├── claude-3-haiku-20240307/
│   ├── phi4:latest/
│   ├── llama3.3:latest/
│   └── qwen2.5:72b/
├── relevance/
│   ├── chatgpt-4o-latest/
│   ├── claude-3-haiku-20240307/
│   ├── phi4:latest/
│   ├── llama3.3:latest/
│   └── qwen2.5:72b/
└── document_gender_transformation/
    ├── chatgpt-4o-latest/
    ├── claude-3-haiku-20240307/
    ├── phi4:latest/
    ├── llama3.3:latest/
    └── qwen2.5:72b/
```

## 🔧 Configuration

### Models Configuration (`config/models.json`)
```json
{
  "openai": {
    "models": ["chatgpt-4o-latest"]
  },
  "anthropic": {
    "models": ["claude-3-haiku-20240307"]
  },
  "ollama": {
    "models": ["phi4:latest", "llama3.3:latest", "qwen2.5:72b"]
  }
}
```

## 📈 Evaluation Metrics

### Gender Annotation
- **Accuracy**: Overall classification accuracy
- **F1-Scores**: Per-category F1 scores (Female, Male, Neutral)
- **Cohen's Kappa**: Agreement between human and LLM labels

### Relevance Scoring
- **Scale**: 0 (not relevant) to 3 (highly relevant)
- **Output**: Added as `llm_relevance_score` column

### Document Transformation
- **BLEU Score**: Translation quality metric
- **ROUGE Scores**: ROUGE-1, ROUGE-2, ROUGE-L
- **BERTScore**: BERT-based semantic similarity (Precision, Recall, F1)

### LIWC Analysis
- **Female Score**: Female-related linguistic content
- **Male Score**: Male-related linguistic content
- **Neutral Score**: Gender-neutral content
- **Bias Difference**: Female - Male score difference

## 🛠️ Development

### Adding New Models
1. Add model name to `config/models.json`
2. Ensure model is available in the specified provider
3. Run evaluation scripts with the new model

### Adding New Prompts
1. Create prompt file in `prompts/` directory
2. Update relevant script to load the new prompt
3. Test with sample data

### Custom Datasets
1. Place CSV files in `data/Grep_bias_datasets/`
2. Ensure required columns are present
3. Use `--dataset` flag to specify file

## 📋 Requirements

- Python 3.8+
- See `requirements.txt` for full dependency list
- LIWC dictionary files for bias analysis
- API access to OpenAI, Anthropic, or local Ollama instance

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with existing datasets
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📚 References

- LIWC (Linguistic Inquiry and Word Count) for linguistic bias analysis
- BLEU, ROUGE, and BERTScore for text quality evaluation
- TREC evaluation framework for information retrieval assessment
