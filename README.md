# Assessing Gender Bias in Information Retrieval Tasks Using Large Language Models

This repository provides the data, code, and resources for our research on automated gender bias assessment in Information Retrieval (IR) using Large Language Models (LLMs). Our framework evaluates LLMs across five key bias assessment tasks: query gender classification, relevance scoring with bias minimization, gender-specific document generation, fair document ranking, and retrieval performance disparity mitigation. All code for conducting these bias assessments and the accompanying evaluation scripts are included here.

## Data

The `data/` directory contains the evaluation datasets and resources needed for gender bias assessment:

```
data/
├── gender_annotated_queries.tsv    # Human-labeled queries with gender associations (m/f/n)
├── Grep_bias_datasets/             # Query-document pairs for relevance and transformation tasks
│   ├── queries-documents_appearance.csv
│   ├── queries-documents_career.csv
│   └── ...
├── 215_neutral_queries.tsv    # Neutral query set from MSMARCOFair, used to evaluate gender bias in the retrieval results for these queries

```

- **`gender_annotated_queries.tsv`**: Contains queries with human-annotated gender labels for evaluating LLM classification accuracy
- **`Grep_bias_datasets/`**: Contains CSV files with query-document pairs for relevance scoring and gender transformation experiments
- **`215_neutral_queries.tsv`**: Contains neutral query set from MSMARCOFair, used to evaluate gender bias in the retrieval results for these queries

## 🤖 Prompts

The `prompts/` directory contains templates for each bias assessment task and shared instruction sets:
```
prompts/
├── query_gender_annotation_prompt.txt           # Template for query gender classification
├── relevance_annotation_prompt.txt              # Template for relevance scoring (0-3 scale)
├── document_gender_transformation_prompt.txt    # Template for gender transformation (M→F, F→M, etc.)
├── document_gender_annotation_prompt.txt        # Template for gender quantification in documents
└── fair_ranking_prompt.txt                      # Template for fair document ranking using RankGPT
```

Each .txt file contains the prompt to perform a specific bias assessment task.

## 🚀 Usage

### Cloning and Setup

**Clone the repository**
```bash
git clone https://github.com/yourusername/gender-bias-ir-llms.git
cd gender-bias-ir-llms
```

**Create a Python environment**
```bash
conda create -n gender-bias-env python=3.9 -y
conda activate gender-bias-env
pip install -r requirements.txt
```

**Set up API keys**
```bash
export OPENAI_API_KEY="your_openai_key"
export ANTHROPIC_API_KEY="your_anthropic_key"
```

**Configure models** in `config/models.json`

### Core Scripts


### Query Gender Classification
```bash
# Example: Classify query gender using GPT-4o (OpenAI)
python code/query_gender_annotator.py \
  --provider openai \
  --model chatgpt-4o-latest \
  --data_path data/gender_annotated_queries.tsv \
  --output_dir results/query_gender_annotation

# Example: Classify query gender using Claude (Anthropic)
python code/query_gender_annotator.py \
  --provider anthropic \
  --model claude-3-haiku-20240307 \
  --data_path data/gender_annotated_queries.tsv \
  --output_dir results/query_gender_annotation
```

### Relevance Scoring with Bias Minimization
```bash
# Example: Score relevance using Claude with bias awareness
python code/relevance_annotator.py \
  --provider anthropic \
  --model claude-3-haiku-20240307 \
  --dataset your_dataset.csv \
  --save_interval 100

# Example: Score relevance using local Ollama model
python code/relevance_annotator.py \
  --provider ollama \
  --model phi4:latest \
  --dataset your_dataset.csv
```

### Document Gender Transformation
```bash
# Example: Generate gender variants using GPT-4o
python code/document_gender_transformation.py \
  --provider openai \
  --model chatgpt-4o-latest \
  --dataset your_dataset.csv

# Example: Generate gender variants using local Ollama model
python code/document_gender_transformation.py \
  --provider ollama \
  --model llama3.3:latest \
  --dataset your_dataset.csv
```

### Document Gender Quantification
```bash
# Example: Quantify gender in transformed documents using GPT-4o
python code/document_gender_annotator.py \
  --provider openai \
  --base_dir results/document_gender_transformation

# Example: Quantify gender using Claude
python code/document_gender_annotator.py \
  --provider anthropic \
  --base_dir results/document_gender_transformation
```

### LIWC Bias Analysis
```bash
# Example: Calculate LIWC bias scores from TREC run files
python code/calculate_liwc_scores.py \
  --trec_file your_run_file.run \
  --liwc_dict data/liwc/liwccollection_bias.pkl \
  --cutoff 10 \
  --output results/liwc_analysis.csv
```

### Evaluating Model Performance
```bash
# List available models for a provider
python code/query_gender_annotator.py --provider openai --list_models

# List available datasets
python code/relevance_annotator.py --list_datasets

# List available models for relevance annotation
python code/relevance_annotator.py --provider anthropic --list_models
```

## Results

The `results/` directory contains evaluation outputs organized by task and model:

```
results/
├── query_gender_annotation/            # Query gender classification results
│   ├── chatgpt-4o-latest/              # GPT-4o results with performance metrics
│   ├── claude-3-haiku-20240307/        # Claude results with performance metrics
│   ├── phi4:latest/                    # Phi-4 results with performance metrics
│   ├── llama3.3:latest/                # LLaMA results with performance metrics
│   └── qwen2.5:72b/                    # Qwen results with performance metrics
├── relevance/                          # Relevance scoring results
│   ├── chatgpt-4o-latest/              # Model-specific subfolders
│   ├── claude-3-haiku-20240307/        # Each containing CSV files with llm_relevance_score
│   ├── phi4:latest/                    # and progress tracking
│   ├── llama3.3:latest/
│   └── qwen2.5:72b/
├── document_gender_transformation/     # Gender transformation results
│   ├── chatgpt-4o-latest/              # Model-specific subfolders
│   ├── claude-3-haiku-20240307/        # Each containing transformation results
│   ├── phi4:latest/                    # with BLEU, ROUGE, and BERTScore metrics
│   ├── llama3.3:latest/
│   └── qwen2.5:72b/
└── reranked_runs/                      # Re-ranked list of documents with LLMs
    ├── 215_neutral_queries/            # Ranked list of documents for 215 neutral queries
    │   ├── chatgpt-4o-latest/          # GPT-4o ranked list of documents
    │   ├── claude-3-haiku-20240307/    # Claude ranked list of documents
    │   ├── phi4:latest/                # Phi-4 ranked list of documents
    │   ├── llama3.3:latest/            # LLaMA ranked list of documents
    │   └── qwen2.5:72b/                # Qwen ranked list of documents
    └── gendered_queries/               # Ranked list of documents for female and male queries using GPT-4o

```

## Evaluation Metrics

### Gender Classification
- **Accuracy**: Overall classification accuracy across all queries
- **F1-Scores**: Per-category F1 scores (Female, Male, Neutral)
- **Cohen's Kappa**: Agreement between human and LLM labels
- **Confusion Matrix**: Detailed classification performance visualization

### Relevance Scoring
- **Scale**: 0 (not relevant) to 3 (highly relevant)
- **Bias Awareness**: Relevance scores assigned while minimizing gender bias
- **Output**: Added as `llm_relevance_score` column to original datasets

### Document Transformation
- **BLEU Score**: Translation quality metric for gender variants
- **ROUGE Scores**: ROUGE-1, ROUGE-2, ROUGE-L for content similarity
- **BERTScore**: BERT-based semantic similarity (Precision, Recall, F1)
- **Gender Preservation**: Assessment of gender transformation accuracy

### LIWC Analysis
- **Female Score**: Female-related linguistic content percentage
- **Male Score**: Male-related linguistic content percentage
- **Neutral Score**: Gender-neutral content percentage
- **Bias Difference**: Female - Male score difference for bias quantification

## Model Support

Our framework supports multiple LLM providers:

- **OpenAI**
- **Anthropic**
- **Ollama**
