import pandas as pd
import os
import time
import json
from typing import List, Dict, Any
from datetime import datetime
from llm_client import get_llm_client


class RelevanceAnnotator:
    def __init__(self, provider: str = "openai"):
        self.provider = provider
        self.client = get_llm_client(provider)
        self.prompt_template = self._load_prompt_template()
        self.models_config = self._load_models_config()
    
    def _load_prompt_template(self) -> str:
        prompt_path = "prompts/relevance_annotation_prompt.txt"
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
    
    def create_prompt(self, query: str, document: str) -> str:
        return self.prompt_template.format(query=query, document=document)
    
    def get_available_datasets(self) -> List[str]:
        data_dir = "data/Grep_bias_datasets"
        if not os.path.exists(data_dir):
            print(f"Data directory not found: {data_dir}")
            return []
        
        csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        return csv_files
    
    def process_dataset(self, filename: str, model: str, save_interval: int = 100) -> pd.DataFrame:
        data_dir = "data/Grep_bias_datasets"
        file_path = os.path.join(data_dir, filename)
        
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return pd.DataFrame()
        
        print(f"Processing dataset: {filename}")
        return self.annotate_queries(file_path, model, data_dir, save_interval)
    
    def annotate_queries(self, data_path: str, model: str, output_dir: str = None,
                        save_interval: int = 100) -> pd.DataFrame:
        if not self.validate_model(model):
            print(f"Warning: Model '{model}' not found in config for provider '{self.provider}'")
            print(f"Available models: {self.get_available_models()}")
        
        print(f"Loading data from {data_path}")
        
        if data_path.endswith('.tsv'):
            df = pd.read_csv(data_path, sep='\t')
        else:
            df = pd.read_csv(data_path)
        
        if "query" not in df.columns or "document" not in df.columns:
            print("Error: CSV must contain 'query' and 'document' columns")
            print(f"Available columns: {list(df.columns)}")
            return df
        
        df['llm_relevance_score'] = None
        
        print(f"Processing {len(df)} query-document pairs with {model}")
        
        for index, row in df.iterrows():
            try:
                query = row['query']
                document = row['document']
                prompt = self.create_prompt(query, document)
                
                response = self.client.generate(
                    prompt=prompt,
                    model=model
                )
                
                label = self._parse_relevance_response(response)
                df.at[index, 'llm_relevance_score'] = label
                
                if index % save_interval == 0 and index != 0:
                    if output_dir:
                        self._save_progress(df, output_dir, model, index)
                    print(f"Progress: {index}/{len(df)} pairs processed")
                
                time.sleep(0.5)
                
            except Exception as e:
                print(f"Error processing pair {index}: {e}")
                df.at[index, 'llm_relevance_score'] = "ERROR"
                continue
        
        if output_dir:
            self._save_final_results(df, output_dir, model)
        
        return df
    
    def _parse_relevance_response(self, response: str) -> str:
        response_lower = response.strip().lower()
        
        if "0" in response_lower:
            return "0"
        elif "1" in response_lower:
            return "1"
        elif "2" in response_lower:
            return "2"
        elif "3" in response_lower:
            return "3"
        else:
            return f"Not Detected: {response[:50]}"
    
    def _save_progress(self, df: pd.DataFrame, output_dir: str, model: str, index: int):
        progress_file = os.path.join(output_dir, f"{model}_relevance_progress_{index}.csv")
        df.to_csv(progress_file, index=False)
        print(f"Progress saved to {progress_file}")
    
    def _save_final_results(self, df: pd.DataFrame, output_dir: str, model: str):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_file = os.path.join(output_dir, f"{model}_relevance_annotations_{timestamp}.csv")
        df.to_csv(final_file, index=False)
        print(f"Final results saved to {final_file}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Relevance annotation using LLMs')
    parser.add_argument('--provider', choices=['openai', 'anthropic', 'ollama'], 
                       default='anthropic', help='LLM provider')
    parser.add_argument('--model', required=True, help='Model name to use')
    parser.add_argument('--dataset', help='Specific dataset filename to process')
    parser.add_argument('--data_path', help='Custom path to input CSV file')
    parser.add_argument('--save_interval', type=int, default=100, 
                       help='Save progress every N pairs')
    parser.add_argument('--list_models', action='store_true', 
                       help='List available models for the provider')
    parser.add_argument('--list_datasets', action='store_true', 
                       help='List available datasets in Grep_bias_datasets folder')
    
    args = parser.parse_args()
    
    annotator = RelevanceAnnotator(provider=args.provider)
    
    if args.list_models:
        models = annotator.get_available_models()
        print(f"Available models for {args.provider}:")
        for model in models:
            print(f"  - {model}")
        return
    
    if args.list_datasets:
        datasets = annotator.get_available_datasets()
        print("Available datasets in Grep_bias_datasets folder:")
        for dataset in datasets:
            print(f"  - {dataset}")
        return
    
    if args.dataset:
        df = annotator.process_dataset(args.dataset, args.model, args.save_interval)
    elif args.data_path:
        df = annotator.annotate_queries(args.data_path, args.model, save_interval=args.save_interval)
    else:
        print("Error: Must specify either --dataset or --data_path")
        print("Use --list_datasets to see available datasets")
        return
    
    print(f"\nProcessing complete! Results saved with llm_relevance_score column added")


if __name__ == "__main__":
    main()
