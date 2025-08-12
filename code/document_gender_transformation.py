import os
import pandas as pd
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
import time
import json
from typing import List, Dict, Any
from datetime import datetime
from llm_client import get_llm_client
from bert_score import score


class DocumentGenderTransformer:
    def __init__(self, provider: str = "ollama"):
        self.provider = provider
        self.client = get_llm_client(provider)
        self.prompt_template = self._load_prompt_template()
        self.models_config = self._load_models_config()
    
    def _load_prompt_template(self) -> str:
        prompt_path = "prompts/document_gender_transformation_prompt.txt"
        try:
            with open(prompt_path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    
    def _load_models_config(self) -> Dict:
        config_path = "config/models.json"
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
    
    def get_available_models(self, provider: str = None) -> List[str]:
        if provider is None:
            provider = self.provider
        
        if provider in self.models_config:
            return self.models_config[provider].get('models', [])
        return []
    
    def validate_model(self, model: str, provider: str = None) -> bool:
        if provider is None:
            provider = self.provider
        
        available_models = self.get_available_models(provider)
        return model in available_models
    
    def get_available_datasets(self) -> List[str]:
        data_dir = "data/Grep_bias_datasets"
        if not os.path.exists(data_dir):
            print(f"Data directory not found: {data_dir}")
            return []
        
        csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        return csv_files
    
    def create_prompt(self, input_text: str, target_gender: str) -> str:
        return self.prompt_template.format(input_text=input_text, target_gender=target_gender)
    
    def process_with_llm(self, input_text: str, target_gender: str, model: str) -> str:
        prompt = self.create_prompt(input_text, target_gender)
        
        try:
            response = self.client.generate(
                prompt=prompt,
                model=model
            )
            
            answer = response.strip()
            if "Modified Document:" in answer:
                answer = answer.split("Modified Document:")[1].strip()
            
            return answer
            
        except Exception as e:
            print(f"Error processing with LLM: {e}")
            return "ERROR"
    
    def compute_bertscore(self, original_doc: str, transformed_doc: str, rescale: bool = True, model_type: str = 'bert-base-uncased') -> Dict[str, float]:
        try:
            P, R, F1 = score(
                [transformed_doc], 
                [original_doc], 
                lang='en', 
                rescale_with_baseline=rescale, 
                model_type=model_type,
                device="cpu"
            )
            
            return {
                'Precision': round(P.item(), 4),
                'Recall': round(R.item(), 4),
                'F1_Score': round(F1.item(), 4)
            }
        except Exception as e:
            print(f"BERTScore error: {e}")
            return {'Precision': 0.0, 'Recall': 0.0, 'F1_Score': 0.0}
    
    def evaluate_text_similarity(self, original_text: str, transformed_text: str) -> Dict[str, float]:
        try:
            original_tokens = original_text.split()
            transformed_tokens = transformed_text.split()

            bleu_score = sentence_bleu([original_tokens], transformed_tokens)

            rouge = Rouge()
            rouge_scores = rouge.get_scores(transformed_text, original_text)[0]

            return {
                "BLEU_Score": bleu_score,
                "ROUGE-1": rouge_scores["rouge-1"]["f"],
                "ROUGE-2": rouge_scores["rouge-2"]["f"],
                "ROUGE-L": rouge_scores["rouge-l"]["f"]
            }
        except Exception as e:
            print(f"Text similarity evaluation error: {e}")
            return {"BLEU_Score": 0.0, "ROUGE-1": 0.0, "ROUGE-2": 0.0, "ROUGE-L": 0.0}
    
    def create_model_results_dir(self, model: str) -> str:
        results_dir = f"results/document_gender_transformation/{model}"
        os.makedirs(results_dir, exist_ok=True)
        return results_dir
    
    def process_dataset(self, filename: str, model: str) -> pd.DataFrame:
        data_dir = "data/Grep_bias_datasets"
        file_path = os.path.join(data_dir, filename)
        
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return pd.DataFrame()
        
        print(f"Processing dataset: {filename}")
        return self.process_gender_transformations(file_path, model)
    
    def process_gender_transformations(self, data_path: str, model: str) -> pd.DataFrame:
        if not self.validate_model(model):
            print(f"Warning: Model '{model}' not found in config for provider '{self.provider}'")
            print(f"Available models: {self.get_available_models()}")
        
        print(f"Loading data from {data_path}")
        
        df = pd.read_csv(data_path)
        
        if "query" not in df.columns or "document" not in df.columns or "content_gender" not in df.columns:
            print("Error: CSV must contain 'query', 'document', and 'content_gender' columns")
            print(f"Available columns: {list(df.columns)}")
            return pd.DataFrame()
        
        results = []
        
        print(f"Processing gender transformations with {model}")
        
        for query in tqdm(df["query"].unique(), desc="Processing queries"):
            subset = df[(df["query"] == query) & (df["relevant"] == 1)]
            
            male_doc = subset[subset["content_gender"] == "M"]["document"].values
            female_doc = subset[subset["content_gender"] == "F"]["document"].values
            neutral_doc = subset[subset["content_gender"] == "N"]["document"].values

            if len(male_doc) == 0 or len(female_doc) == 0 or len(neutral_doc) == 0:
                continue

            male_doc, female_doc, neutral_doc = male_doc[0], female_doc[0], neutral_doc[0]

            transformations = [
                ("Male", "Female", male_doc),
                ("Female", "Male", female_doc),
                ("Male", "Neutral", male_doc),
                ("Female", "Neutral", female_doc),
                ("Neutral", "Male", neutral_doc),
                ("Neutral", "Female", neutral_doc)
            ]

            for orig_gender, target_gender, original_doc in transformations:
                transformed_doc = self.process_with_llm(original_doc, target_gender, model)
                
                if transformed_doc != "ERROR":
                    bert_scores = self.compute_bertscore(original_doc, transformed_doc)
                    similarity_scores = self.evaluate_text_similarity(original_doc, transformed_doc)
                    
                    results.append([
                        query, 1, original_doc, f"{orig_gender} → {target_gender}", 
                        transformed_doc, orig_gender, target_gender,
                        *similarity_scores.values(), *bert_scores.values()
                    ])
                
                time.sleep(0.5)
        
        columns = [
            "Query", "relevance_label", "original_Doc", "transformation", "transformed_Doc",
            "original_Gender", "final_Gender", "BLEU_Score", "ROUGE-1", "ROUGE-2", "ROUGE-L",
            "BERTScore_Precision", "BERTScore_Recall", "BERTScore_F1_Score"
        ]
        
        results_df = pd.DataFrame(results, columns=columns)
        return results_df
    
    def save_results(self, results_df: pd.DataFrame, model: str, filename: str = None):
        if results_df.empty:
            print("No results to save")
            return
        
        model_results_dir = self.create_model_results_dir(model)
        
        if filename:
            output_file = os.path.join(model_results_dir, f"{filename}_document_gender_transformation_results.csv")
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(model_results_dir, f"document_gender_transformation_results_{timestamp}.csv")
        
        results_df.to_csv(output_file, index=False, sep="\t")
        print(f"Results saved to {output_file}")
        
        return output_file


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Document gender transformation using LLMs')
    parser.add_argument('--provider', choices=['openai', 'anthropic', 'ollama'], 
                       default='ollama', help='LLM provider')
    parser.add_argument('--model', required=True, help='Model name to use')
    parser.add_argument('--dataset', help='Specific dataset filename to process')
    parser.add_argument('--data_path', help='Custom path to input CSV file')
    parser.add_argument('--list_models', action='store_true', 
                       help='List available models for the provider')
    parser.add_argument('--list_datasets', action='store_true', 
                       help='List available datasets in Grep_bias_datasets folder')
    
    args = parser.parse_args()
    
    transformer = DocumentGenderTransformer(provider=args.provider)
    
    if args.list_models:
        models = transformer.get_available_models()
        print(f"Available models for {args.provider}:")
        for model in models:
            print(f"  - {model}")
        return
    
    if args.list_datasets:
        datasets = transformer.get_available_datasets()
        print("Available datasets in Grep_bias_datasets folder:")
        for dataset in datasets:
            print(f"  - {dataset}")
        return
    
    if args.dataset:
        results_df = transformer.process_dataset(args.dataset, args.model)
        if not results_df.empty:
            transformer.save_results(results_df, args.model, args.dataset)
    elif args.data_path:
        results_df = transformer.process_gender_transformations(args.data_path, args.model)
        if not results_df.empty:
            transformer.save_results(results_df, args.model)
    else:
        print("Error: Must specify either --dataset or --data_path")
        print("Use --list_datasets to see available datasets")
        return
    
    print(f"\nDocument gender transformation processing complete! Results saved in results/document_gender_transformation/{args.model}/")


if __name__ == "__main__":
    main()
