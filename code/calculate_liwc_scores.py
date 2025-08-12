import csv
import pickle
import liwc
import os
from typing import Dict, List, Tuple


class LIWCScoreCalculator:
    def __init__(self, liwc_dict_path: str, collection_path: str = None):
        self.liwc_dict_path = liwc_dict_path
        self.collection_path = collection_path
        self.documents_bias = {}
        self.parse = None
        
        try:
            self.parse, _ = liwc.load_token_parser('data/liwc/LIWC2015Dictionary.dic') # add your liwc dictaionary to data/liwc 
            print("LIWC parser loaded successfully")
        except Exception as e:
            print(f"Error loading LIWC parser: {e}")
            print("Please ensure LIWC dictionary path is correct")
        
        self._load_document_bias()
    
    def _load_document_bias(self):
        try:
            with open(self.liwc_dict_path, "rb") as file_to_read:
                self.documents_bias = pickle.load(file_to_read)
            print(f"Loaded LIWC bias scores for {len(self.documents_bias)} documents")
        except Exception as e:
            print(f"Error loading LIWC bias scores: {e}")
            print("Will calculate scores from scratch if collection path provided")
    
    def calculate_doc_score(self, text: str) -> List[float]:
        if not self.parse:
            return [0.0, 0.0, 0.0]
        
        try:
            tokens = text.lower().split()
            categories = self.parse(tokens)
            
            female_score = sum(1 for cat in categories if 'female' in cat.lower()) / len(tokens) if tokens else 0
            male_score = sum(1 for cat in categories if 'male' in cat.lower()) / len(tokens) if tokens else 0
            neutral_score = 1 - (female_score + male_score)
            
            return [female_score, male_score, neutral_score]
        except Exception as e:
            print(f"Error calculating LIWC score: {e}")
            return [0.0, 0.0, 0.0]
    
    def find_top_n_docs(self, run_file_path: str, cutoff: int) -> Dict[str, List[str]]:
        top_n_docs = {}
        current_query = None
        doc_ids = []
        
        with open(run_file_path, 'r') as run_file:
            for line in run_file:
                parts = line.strip().split()
                if len(parts) >= 4:
                    query_id = parts[0]
                    doc_id = parts[2]
                    rank = int(parts[3])
                    
                    if query_id != current_query:
                        if current_query and doc_ids:
                            top_n_docs[current_query] = doc_ids
                        current_query = query_id
                        doc_ids = []
                    
                    if rank <= cutoff:
                        doc_ids.append(doc_id)
        
        if current_query and doc_ids:
            top_n_docs[current_query] = doc_ids
        
        return top_n_docs
    
    def calculate_query_score_cutoff(self, doc_ids: List[str]) -> List[float]:
        query_score = [0.0, 0.0, 0.0]
        valid_docs = 0
        
        for doc_id in doc_ids:
            try:
                doc_idx = int(doc_id)
                if doc_idx in self.documents_bias:
                    doc_scores = self.documents_bias[doc_idx]
                    query_score = [query_score[i] + doc_scores[i] for i in range(len(query_score))]
                    valid_docs += 1
            except (ValueError, KeyError):
                continue
        
        if valid_docs > 0:
            query_score = [abs(score / valid_docs) * 100 for score in query_score]
        else:
            query_score = [0.0, 0.0, 0.0]
        
        return query_score
    
    def calculate_score_cutoff(self, topn_docs_dict: Dict[str, List[str]]) -> List[float]:
        total_score = [0.0, 0.0, 0.0]
        query_count = len(topn_docs_dict)
        
        if query_count == 0:
            return total_score
        
        for doc_ids in topn_docs_dict.values():
            query_score = self.calculate_query_score_cutoff(doc_ids)
            total_score = [total_score[i] + query_score[i] for i in range(len(total_score))]
        
        total_score = [score / query_count for score in total_score]
        return total_score
    
    def process_trec_file(self, trec_file_path: str, cutoff: int = 10, output_path: str = None) -> Dict:
        print(f"Processing TREC file: {trec_file_path}")
        print(f"Cutoff: top-{cutoff} documents")
        
        print("Creating top-N documents dictionary...")
        topn_docs_dict = self.find_top_n_docs(trec_file_path, cutoff)
        print(f"Found {len(topn_docs_dict)} queries")
        
        fm_score = self.calculate_score_cutoff(topn_docs_dict)
        
        results = {
            'cutoff': cutoff,
            'female_score': round(fm_score[0], 4),
            'male_score': round(fm_score[1], 4),
            'neutral_score': round(fm_score[2], 4),
            'female_male_diff': round(fm_score[0] - fm_score[1], 4),
            'query_count': len(topn_docs_dict)
        }
        
        print("Results:")
        print(f"  Female Score: {results['female_score']}")
        print(f"  Male Score: {results['male_score']}")
        print(f"  Neutral Score: {results['neutral_score']}")
        print(f"  Female - Male Difference: {results['female_male_diff']}")
        print(f"  Query Count: {results['query_count']}")
        
        if output_path:
            self._save_results(results, output_path)
        
        return results
    
    def _save_results(self, results: Dict, output_path: str):
        try:
            with open(output_path, 'w', newline='') as csvfile:
                fieldnames = ['cutoff', 'female_score', 'male_score', 'neutral_score', 'female_male_diff', 'query_count']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(results)
            
            print(f"Results saved to: {output_path}")
        except Exception as e:
            print(f"Error saving results: {e}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Calculate LIWC bias scores from TREC run file')
    parser.add_argument('--trec_file', required=True, help='Path to TREC run file')
    parser.add_argument('--liwc_dict', required=True, help='Path to LIWC bias dictionary pickle file')
    parser.add_argument('--cutoff', type=int, default=10, help='Top-N documents cutoff (default: 10)')
    parser.add_argument('--output', help='Output CSV file path (optional)')
    
    args = parser.parse_args()
    
    calculator = LIWCScoreCalculator(args.liwc_dict)
    
    results = calculator.process_trec_file(
        trec_file_path=args.trec_file,
        cutoff=args.cutoff,
        output_path=args.output
    )
    
    print("\nLIWC bias analysis complete!")


if __name__ == "__main__":
    main()
