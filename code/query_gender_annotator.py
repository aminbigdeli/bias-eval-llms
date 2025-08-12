
import pandas as pd
import os
import time
import json
from typing import List, Dict, Any
from datetime import datetime
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score, cohen_kappa_score
import seaborn as sns
import matplotlib.pyplot as plt
from llm_client import get_llm_client


class GenderAnnotator:
    def __init__(self, provider: str = "openai"):
        self.provider = provider
        self.client = get_llm_client(provider)
        self.prompt_template = self._load_prompt_template()
        self.models_config = self._load_models_config()
    
    def _load_prompt_template(self) -> str:
        prompt_path = "prompts/query_gender_annotation_prompt.txt"
        try:
            with open(prompt_path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            return self._get_default_prompt()
    
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
    
    def _get_default_prompt(self) -> str:
        return """Given the query below, classify it as Female, Male, or Non-Gendered based on gender indicators.

Query: [query]
Answer:"""
    
    def create_prompt(self, query: str) -> str:
        return self.prompt_template.replace("[query]", query)
    
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
        
        df['llm-output'] = None
        
        print(f"Processing {len(df)} queries with {model}")
        
        for index, row in df.iterrows():
            try:
                query = row['query']
                prompt = self.create_prompt(query)
                
                response = self.client.generate(
                    prompt=prompt,
                    model=model
                )
                
                label = self._parse_gender_response(response)
                df.at[index, 'llm-output'] = label
                
                if index % save_interval == 0 and index != 0:
                    if output_dir:
                        self._save_progress(df, output_dir, model, index)
                    print(f"Progress: {index}/{len(df)} queries processed")
                
                time.sleep(0.5)
                
            except Exception as e:
                print(f"Error processing query {index}: {e}")
                df.at[index, 'llm-output'] = "ERROR"
                continue
        
        if output_dir:
            self._save_final_results(df, output_dir, model)
        
        return df
    
    def _parse_gender_response(self, response: str) -> str:
        response_lower = response.strip().lower()
        
        if "female" in response_lower:
            return "f"
        elif "male" in response_lower:
            return "m"
        elif "non-gendered" in response_lower or "non gendered" in response_lower:
            return "n"
        else:
            return f"Unclear: {response[:50]}"
    
    def _save_progress(self, df: pd.DataFrame, output_dir: str, model: str, index: int):
        os.makedirs(output_dir, exist_ok=True)
        progress_file = os.path.join(output_dir, f"{model}_progress_{index}.tsv")
        df.to_csv(progress_file, sep='\t', index=False)
        print(f"Progress saved to {progress_file}")
    
    def _save_final_results(self, df: pd.DataFrame, output_dir: str, model: str):
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_file = os.path.join(output_dir, f"{model}_gender_annotations_{timestamp}.tsv")
        df.to_csv(final_file, sep='\t', index=False)
        print(f"Final results saved to {final_file}")
    
    def evaluate_accuracy(self, df: pd.DataFrame, human_label_col: str = 'label', 
                         llm_label_col: str = 'llm-output') -> Dict[str, Any]:
        if human_label_col not in df.columns:
            print(f"Warning: Human label column '{human_label_col}' not found")
            return {}
        
        valid_mask = (
            df[human_label_col].notna() & 
            df[llm_label_col].notna() & 
            ~df[llm_label_col].str.startswith('Unclear') &
            ~df[llm_label_col].str.startswith('ERROR')
        )
        
        valid_df = df[valid_mask].copy()
        
        if len(valid_df) == 0:
            print("No valid predictions to evaluate")
            return {}
        
        accuracy = accuracy_score(valid_df[human_label_col], valid_df[llm_label_col])
        
        f1_scores = f1_score(valid_df[human_label_col], valid_df[llm_label_col], 
                            average=None, labels=['f', 'm', 'n'])
        
        cohen_kappa = cohen_kappa_score(valid_df[human_label_col], valid_df[llm_label_col])
        
        report = classification_report(
            valid_df[human_label_col], 
            valid_df[llm_label_col], 
            output_dict=True
        )
        
        cm = confusion_matrix(valid_df[human_label_col], valid_df[llm_label_col])
        
        results = {
            'total_queries': len(df),
            'valid_predictions': len(valid_df),
            'accuracy': accuracy,
            'f1_scores': {
                'Female': f1_scores[0] if len(f1_scores) > 0 else 0,
                'Male': f1_scores[1] if len(f1_scores) > 1 else 0,
                'Neutral': f1_scores[2] if len(f1_scores) > 2 else 0
            },
            'cohen_kappa': cohen_kappa,
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'human_label_distribution': valid_df[human_label_col].value_counts().to_dict(),
            'llm_label_distribution': valid_df[llm_label_col].value_counts().to_dict()
        }
        
        return results
    
    def save_performance_stats(self, results: Dict[str, Any], output_dir: str, model: str):
        if not results:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        stats_file = os.path.join(output_dir, f"{model}_performance_stats_{timestamp}.xlsx")
        
        with pd.ExcelWriter(stats_file, engine='openpyxl') as writer:
            summary_df = pd.DataFrame([{
                'Metric': 'Total Queries',
                'Value': results['total_queries']
            }, {
                'Metric': 'Valid Predictions',
                'Value': results['valid_predictions']
            }, {
                'Metric': 'Accuracy',
                'Value': f"{results['accuracy']:.3f}"
            }, {
                'Metric': 'Cohen Kappa',
                'Value': f"{results['cohen_kappa']:.3f}"
            }])
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            f1_df = pd.DataFrame([{
                'Category': 'Female',
                'F1-Score': f"{results['f1_scores']['Female']:.3f}"
            }, {
                'Category': 'Male',
                'F1-Score': f"{results['f1_scores']['Male']:.3f}"
            }, {
                'Category': 'Neutral',
                'F1-Score': f"{results['f1_scores']['Neutral']:.3f}"
            }])
            f1_df.to_excel(writer, sheet_name='F1_Scores', index=False)
            
            if 'classification_report' in results:
                report_df = pd.DataFrame(results['classification_report']).T
                report_df.to_excel(writer, sheet_name='Classification_Report')
            
            if 'confusion_matrix' in results:
                cm_df = pd.DataFrame(results['confusion_matrix'])
                cm_df.to_excel(writer, sheet_name='Confusion_Matrix', index=False, header=False)
        
        print(f"Performance stats saved to {stats_file}")
    
    def plot_confusion_matrix(self, confusion_matrix: List[List[int]], 
                            labels: List[str], output_path: str = None):
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=labels, yticklabels=labels)
        plt.title('Confusion Matrix: Human vs LLM Labels')
        plt.ylabel('Human Labels')
        plt.xlabel('LLM Predictions')
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {output_path}")
        else:
            plt.show()
        
        plt.close()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Gender annotation using LLMs')
    parser.add_argument('--provider', choices=['openai', 'anthropic', 'ollama'], 
                       default='openai', help='LLM provider')
    parser.add_argument('--model', required=True, help='Model name to use')
    parser.add_argument('--data_path', default='data/gender_annotated_queries.tsv', 
                       help='Path to input data file')
    parser.add_argument('--output_dir', default='results/query_gender_annotation', help='Output directory')
    parser.add_argument('--save_interval', type=int, default=100, 
                       help='Save progress every N queries')
    parser.add_argument('--list_models', action='store_true', 
                       help='List available models for the provider')
    
    args = parser.parse_args()
    
    annotator = GenderAnnotator(provider=args.provider)
    
    if args.list_models:
        models = annotator.get_available_models()
        print(f"Available models for {args.provider}:")
        for model in models:
            print(f"  - {model}")
        return
    
    df = annotator.annotate_queries(
        data_path=args.data_path,
        model=args.model,
        output_dir=args.output_dir,
        save_interval=args.save_interval
    )
    
    if 'label' in df.columns:
        results = annotator.evaluate_accuracy(df)
        
        if results:
            print("\n=== EVALUATION RESULTS ===")
            print(f"Total queries: {results['total_queries']}")
            print(f"Valid predictions: {results['valid_predictions']}")
            print(f"Accuracy: {results['accuracy']:.3f}")
            print(f"Cohen Kappa: {results['cohen_kappa']:.3f}")
            print(f"F1-Score Female: {results['f1_scores']['Female']:.3f}")
            print(f"F1-Score Male: {results['f1_scores']['Male']:.3f}")
            print(f"F1-Score Neutral: {results['f1_scores']['Neutral']:.3f}")
            
            annotator.save_performance_stats(results, args.output_dir, args.model)
            
            if 'confusion_matrix' in results:
                labels = list(results['human_label_distribution'].keys())
                cm_path = os.path.join(args.output_dir, f"{args.model}_confusion_matrix.png")
                annotator.plot_confusion_matrix(
                    results['confusion_matrix'], 
                    labels, 
                    cm_path
                )


if __name__ == "__main__":
    main()
