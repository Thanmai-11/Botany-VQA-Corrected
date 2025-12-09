"""
VQA answer validator for quality control.
Validates generated QA pairs for consistency and accuracy.
"""

import pandas as pd
from typing import List, Dict, Tuple
import re


class VQAValidator:
    """Validate generated VQA pairs for quality and consistency."""
    
    def __init__(self, flower_categories: List[str]):
        """
        Initialize validator.
        
        Args:
            flower_categories: List of valid flower category names
        """
        self.flower_categories = [f.lower() for f in flower_categories]
    
    def validate_answer_consistency(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Check if answers are consistent across related questions for each image.
        
        Args:
            df: DataFrame with columns: image_path, question, answer, question_type
            
        Returns:
            Dictionary with validation results
        """
        inconsistencies = []
        
        for image_path in df['image_path'].unique():
            image_qa = df[df['image_path'] == image_path]
            
            # Get flower name from identification question
            id_rows = image_qa[image_qa['question_type'] == 'identification']
            if len(id_rows) == 0:
                inconsistencies.append({
                    'image': image_path,
                    'issue': 'Missing identification question'
                })
                continue
            
            flower_name = id_rows.iloc[0]['answer'].lower()
            
            # Check yes/no questions match
            yes_no_rows = image_qa[image_qa['question_type'] == 'yes_no']
            for _, row in yes_no_rows.iterrows():
                question = row['question'].lower()
                answer = row['answer'].lower()
                
                # Check if question asks about the identified flower
                if flower_name in question:
                    if answer not in ['yes', 'true', '1']:
                        inconsistencies.append({
                            'image': image_path,
                            'issue': f"Yes/No inconsistency: '{row['question']}' answered '{row['answer']}' but flower is '{flower_name}'"
                        })
                else:
                    # Question asks about a different flower
                    if answer not in ['no', 'false', '0']:
                        # Extract flower name from question
                        match = re.search(r'is this (?:a |an )?(\w+)', question)
                        if match:
                            asked_flower = match.group(1)
                            inconsistencies.append({
                                'image': image_path,
                                'issue': f"Yes/No inconsistency: Asked about '{asked_flower}' but answered '{row['answer']}' (flower is '{flower_name}')"
                            })
        
        return {
            'total_images_checked': len(df['image_path'].unique()),
            'inconsistencies_found': len(inconsistencies),
            'inconsistencies': inconsistencies[:10],  # Show first 10
            'pass': len(inconsistencies) == 0
        }
    
    def validate_ground_truth(self, df: pd.DataFrame, ground_truth_labels: Dict[str, str]) -> Dict[str, any]:
        """
        Validate that identification answers match ground truth labels.
        
        Args:
            df: DataFrame with VQA pairs
            ground_truth_labels: Dict mapping image_path to true flower name
            
        Returns:
            Dictionary with validation results
        """
        mismatches = []
        
        id_rows = df[df['question_type'] == 'identification']
        
        for _, row in id_rows.iterrows():
            image_path = row['image_path']
            predicted = row['answer'].lower()
            
            if image_path in ground_truth_labels:
                true_label = ground_truth_labels[image_path].lower()
                
                # Allow fuzzy matching (predicted can contain true label or vice versa)
                if true_label not in predicted and predicted not in true_label:
                    mismatches.append({
                        'image': image_path,
                        'predicted': row['answer'],
                        'ground_truth': ground_truth_labels[image_path]
                    })
        
        accuracy = 1.0 - (len(mismatches) / len(id_rows)) if len(id_rows) > 0 else 0.0
        
        return {
            'total_checked': len(id_rows),
            'mismatches': len(mismatches),
            'accuracy': accuracy,
            'sample_mismatches': mismatches[:10],
            'pass': accuracy >= 0.9  # 90% accuracy threshold
        }
    
    def validate_question_diversity(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Check if questions are diverse (not too repetitive).
        
        Args:
            df: DataFrame with VQA pairs
            
        Returns:
            Dictionary with diversity metrics
        """
        question_type_counts = df['question_type'].value_counts().to_dict()
        total_questions = len(df)
        
        # Calculate diversity score (entropy-based)
        diversity_score = len(df['question'].unique()) / total_questions
        
        # Check if any question type dominates (>50%)
        max_type_ratio = max(question_type_counts.values()) / total_questions
        
        return {
            'total_questions': total_questions,
            'unique_questions': len(df['question'].unique()),
            'diversity_score': diversity_score,
            'question_type_distribution': question_type_counts,
            'max_type_ratio': max_type_ratio,
            'pass': diversity_score >= 0.3 and max_type_ratio <= 0.5
        }
    
    def validate_answer_quality(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Check answer quality (not too short, not empty, etc.).
        
        Args:
            df: DataFrame with VQA pairs
            
        Returns:
            Dictionary with quality metrics
        """
        issues = []
        
        for idx, row in df.iterrows():
            answer = str(row['answer']).strip()
            
            # Check for empty answers
            if not answer or answer.lower() in ['none', 'n/a', 'unknown']:
                issues.append({
                    'index': idx,
                    'image': row['image_path'],
                    'question': row['question'],
                    'issue': 'Empty or invalid answer'
                })
            
            # Check for very short answers (except yes/no questions)
            if row['question_type'] != 'yes_no' and len(answer) < 2:
                issues.append({
                    'index': idx,
                    'image': row['image_path'],
                    'question': row['question'],
                    'issue': 'Answer too short'
                })
        
        return {
            'total_answers_checked': len(df),
            'issues_found': len(issues),
            'sample_issues': issues[:10],
            'pass': len(issues) < len(df) * 0.05  # Less than 5% issues
        }
    
    def run_all_validations(self, df: pd.DataFrame, ground_truth_labels: Dict[str, str] = None) -> Dict[str, any]:
        """
        Run all validation checks.
        
        Args:
            df: DataFrame with VQA pairs
            ground_truth_labels: Optional ground truth labels for validation
            
        Returns:
            Dictionary with all validation results
        """
        results = {
            'consistency': self.validate_answer_consistency(df),
            'diversity': self.validate_question_diversity(df),
            'quality': self.validate_answer_quality(df),
        }
        
        if ground_truth_labels:
            results['ground_truth'] = self.validate_ground_truth(df, ground_truth_labels)
        
        # Overall pass/fail
        all_pass = all(r.get('pass', True) for r in results.values())
        results['overall_pass'] = all_pass
        
        return results
    
    def generate_validation_report(self, validation_results: Dict[str, any]) -> str:
        """
        Generate a human-readable validation report.
        
        Args:
            validation_results: Results from run_all_validations
            
        Returns:
            Formatted report string
        """
        report = "=" * 60 + "\n"
        report += "VQA DATASET VALIDATION REPORT\n"
        report += "=" * 60 + "\n\n"
        
        # Consistency check
        cons = validation_results['consistency']
        report += f"1. ANSWER CONSISTENCY CHECK\n"
        report += f"   Images checked: {cons['total_images_checked']}\n"
        report += f"   Inconsistencies found: {cons['inconsistencies_found']}\n"
        report += f"   Status: {'✓ PASS' if cons['pass'] else '✗ FAIL'}\n\n"
        
        # Diversity check
        div = validation_results['diversity']
        report += f"2. QUESTION DIVERSITY CHECK\n"
        report += f"   Total questions: {div['total_questions']}\n"
        report += f"   Unique questions: {div['unique_questions']}\n"
        report += f"   Diversity score: {div['diversity_score']:.2f}\n"
        report += f"   Status: {'✓ PASS' if div['pass'] else '✗ FAIL'}\n\n"
        
        # Quality check
        qual = validation_results['quality']
        report += f"3. ANSWER QUALITY CHECK\n"
        report += f"   Answers checked: {qual['total_answers_checked']}\n"
        report += f"   Issues found: {qual['issues_found']}\n"
        report += f"   Status: {'✓ PASS' if qual['pass'] else '✗ FAIL'}\n\n"
        
        # Ground truth check (if available)
        if 'ground_truth' in validation_results:
            gt = validation_results['ground_truth']
            report += f"4. GROUND TRUTH VALIDATION\n"
            report += f"   Identifications checked: {gt['total_checked']}\n"
            report += f"   Accuracy: {gt['accuracy']:.2%}\n"
            report += f"   Status: {'✓ PASS' if gt['pass'] else '✗ FAIL'}\n\n"
        
        # Overall
        report += "=" * 60 + "\n"
        report += f"OVERALL: {'✓ ALL CHECKS PASSED' if validation_results['overall_pass'] else '✗ SOME CHECKS FAILED'}\n"
        report += "=" * 60 + "\n"
        
        return report
