import pandas as pd
import os
import argparse
from typing import Dict, List
from llm_client import get_llm_client


class DocumentGenderQuantifier:
    def __init__(self, provider: str = "openai"):
        self.llm_client = get_llm_client(provider)
        self.prompt_template = self._load_prompt_template()
    
    def _load_prompt_template(self) -> str:
        try:
            with open("prompts/document_gender_annotation_prompt.txt", "r") as f:
                return f.read().strip()
        except FileNotFoundError:
            return self._get_default_prompt()
    
    
    def create_prompt(self, document: str) -> str:
        return self.prompt_template.replace("[document]", document)
    
    def quantify_gender(self, document: str, model: str) -> str:
        prompt = self.create_prompt(document)
        try:
            response = self.llm_client.generate(prompt, model)
            return response.strip()
        except Exception as e:
            print(f"Error quantifying gender: {e}")
            return "Error"
    
    def process_model_folder(self, model_folder: str, output_dir: str = None):
        print(f"Processing model folder: {model_folder}")
        
        csv_files = [f for f in os.listdir(model_folder) if f.endswith('.csv')]
        if not csv_files:
            print(f"No CSV files found in {model_folder}")
            return
        
        for csv_file in csv_files:
            file_path = os.path.join(model_folder, csv_file)
            print(f"Processing file: {csv_file}")
            
            try:
                df = pd.read_csv(file_path)
                
                if 'transformed_Doc' not in df.columns:
                    print(f"Column 'transformed_Doc' not found in {csv_file}")
                    continue
                
                df['quantified_gender'] = None
                
                for index, row in df.iterrows():
                    transformed_doc = row['transformed_Doc']
                    if pd.isna(transformed_doc) or transformed_doc == "":
                        continue
                    
                    model_name = os.path.basename(model_folder)
                    quantified_gender = self.quantify_gender(transformed_doc, model_name)
                    df.at[index, 'quantified_gender'] = quantified_gender
                
                if output_dir:
                    output_path = os.path.join(output_dir, f"quantified_{csv_file}")
                    os.makedirs(output_dir, exist_ok=True)
                else:
                    output_path = os.path.join(model_folder, f"quantified_{csv_file}")
                
                df.to_csv(output_path, index=False)
                print(f"Saved quantified results to: {output_path}")
                
            except Exception as e:
                print(f"Error processing {csv_file}: {e}")
    
    def process_all_models(self, base_dir: str, output_dir: str = None):
        if not os.path.exists(base_dir):
            print(f"Base directory not found: {base_dir}")
            return
        
        model_folders = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        
        if not model_folders:
            print(f"No model subfolders found in {base_dir}")
            return
        
        print(f"Found {len(model_folders)} model folders: {model_folders}")
        
        for model_folder in model_folders:
            model_path = os.path.join(base_dir, model_folder)
            self.process_model_folder(model_path, output_dir)


def main():
    parser = argparse.ArgumentParser(description='Quantify gender of transformed documents using LLMs')
    parser.add_argument('--base_dir', default='results/document_gender_transformation', 
                       help='Base directory containing model subfolders')
    parser.add_argument('--output_dir', help='Output directory for quantified results (optional)')
    parser.add_argument('--provider', default='openai', choices=['openai', 'anthropic', 'ollama'],
                       help='LLM provider to use')
    parser.add_argument('--model_folder', help='Process specific model folder only (optional)')
    
    args = parser.parse_args()
    
    quantifier = DocumentGenderQuantifier(args.provider)
    
    if args.model_folder:
        if os.path.exists(args.model_folder):
            quantifier.process_model_folder(args.model_folder, args.output_dir)
        else:
            print(f"Model folder not found: {args.model_folder}")
    else:
        quantifier.process_all_models(args.base_dir, args.output_dir)
    
    print("Document gender quantification complete!")


if __name__ == "__main__":
    main()
