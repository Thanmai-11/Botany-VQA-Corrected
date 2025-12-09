"""
Main dataset generator for image-grounded Botany-VQA.
Uses vision-language models to generate accurate QA pairs from actual images.
"""

import os
import json
import pandas as pd
from PIL import Image
from tqdm import tqdm
from typing import List, Dict, Tuple
import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from question_templates import QuestionGenerator
from visual_feature_extractor import VisualFeatureExtractor
from vqa_validator import VQAValidator


class BotanyVQAGenerator:
    """Generate image-grounded VQA dataset using vision-language models."""
    
    def __init__(self, model_name: str = "Salesforce/blip2-opt-2.7b", device: str = None, use_8bit: bool = True):
        """
        Initialize the dataset generator.
        
        Args:
            model_name: HuggingFace model name for VLM
            device: Device to run model on ('cuda', 'cpu', or None for auto)
            use_8bit: Use 8-bit quantization to reduce memory usage (recommended for Colab)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Clear GPU cache before loading
        if self.device == 'cuda':
            torch.cuda.empty_cache()
            print(f"GPU Memory before loading: {torch.cuda.memory_allocated()/1e9:.2f} GB")
        
        # Load VLM model with memory optimizations
        print(f"Loading model: {model_name}...")
        print(f"8-bit quantization: {use_8bit}")
        
        self.processor = Blip2Processor.from_pretrained(model_name)
        
        if use_8bit and self.device == 'cuda':
            # Load with 8-bit quantization to save memory
            self.model = Blip2ForConditionalGeneration.from_pretrained(
                model_name,
                load_in_8bit=True,
                device_map="auto",
                torch_dtype=torch.float16
            )
        else:
            # Standard loading
            self.model = Blip2ForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32
            )
            self.model.to(self.device)
        
        # Enable gradient checkpointing to save memory
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
        
        print("Model loaded successfully!")
        
        if self.device == 'cuda':
            print(f"GPU Memory after loading: {torch.cuda.memory_allocated()/1e9:.2f} GB")
            print(f"GPU Memory reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")
        
        # Initialize utilities
        self.feature_extractor = VisualFeatureExtractor()
        self.question_generator = None  # Will be initialized with flower categories
        self.validator = None
    
    def load_oxford_flowers_labels(self, labels_file: str) -> Dict[str, str]:
        """
        Load Oxford Flowers 102 category labels.
        
        Args:
            labels_file: Path to labels JSON file
            
        Returns:
            Dictionary mapping image paths to flower names
        """
        with open(labels_file, 'r') as f:
            labels = json.load(f)
        return labels
    
    def ask_question(self, image_path: str, question: str) -> str:
        """
        Ask a question about an image using the VLM.
        
        Args:
            image_path: Path to the image
            question: Question to ask
            
        Returns:
            Generated answer
        """
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            # Prepare inputs with question as prompt
            # For BLIP-2, we need to format the question properly
            prompt = f"Question: {question} Answer:"
            inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)
            
            # Generate answer
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs, 
                    max_new_tokens=50,
                    min_length=1,
                    num_beams=5,
                    temperature=1.0
                )
            
            # Decode answer
            answer = self.processor.decode(outputs[0], skip_special_tokens=True)
            
            # Remove the prompt from the answer if it's included
            if answer.startswith(prompt):
                answer = answer[len(prompt):].strip()
            
            # Clear GPU cache to prevent memory buildup
            if self.device == 'cuda':
                del inputs, outputs
                torch.cuda.empty_cache()
            
            return answer.strip()
        
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            # Clear cache on error too
            if self.device == 'cuda':
                torch.cuda.empty_cache()
            return "Error"
    
    def generate_qa_for_image(
        self, 
        image_path: str, 
        flower_name: str,
        num_questions: int = 10
    ) -> List[Dict[str, any]]:
        """
        Generate QA pairs for a single image.
        
        Args:
            image_path: Path to the image
            flower_name: Ground truth flower name
            num_questions: Number of questions to generate
            
        Returns:
            List of QA dictionaries
        """
        # Generate questions
        questions_meta = self.question_generator.generate_diverse_questions(
            flower_name, 
            num_questions
        )
        
        qa_pairs = []
        
        for q_meta in questions_meta:
            question = q_meta['question']
            
            # Special handling for yes/no questions with expected answers
            if q_meta['question_type'] == 'yes_no' and q_meta['expected_answer_contains']:
                # Use expected answer directly for yes/no questions
                answer = q_meta['expected_answer_contains'].capitalize()
            else:
                # Ask VLM for answer
                answer = self.ask_question(image_path, question)
            
            qa_pairs.append({
                'image_path': image_path,
                'question': question,
                'answer': answer,
                'question_type': q_meta['question_type'],
                'difficulty_level': q_meta['difficulty_level'],
                'flower_category': flower_name
            })
        
        return qa_pairs
    
    def generate_dataset(
        self,
        image_dir: str,
        labels_file: str,
        output_csv: str,
        num_images: int = None,
        qa_per_image: int = 10
    ) -> pd.DataFrame:
        """
        Generate the complete VQA dataset.
        
        Args:
            image_dir: Directory containing Oxford Flowers images
            labels_file: Path to labels JSON file
            output_csv: Path to save output CSV
            num_images: Number of images to process (None for all)
            qa_per_image: Number of QA pairs per image
            
        Returns:
            DataFrame with generated QA pairs
        """
        # Load labels
        print("Loading flower labels...")
        labels = self.load_oxford_flowers_labels(labels_file)
        
        # Get unique flower categories
        flower_categories = list(set(labels.values()))
        print(f"Found {len(flower_categories)} flower categories")
        
        # Initialize question generator and validator
        self.question_generator = QuestionGenerator(flower_categories)
        self.validator = VQAValidator(flower_categories)
        
        # Get image paths
        image_paths = sorted([
            os.path.join(image_dir, f) 
            for f in os.listdir(image_dir) 
            if f.endswith(('.jpg', '.jpeg', '.png'))
        ])
        
        if num_images:
            image_paths = image_paths[:num_images]
        
        print(f"Processing {len(image_paths)} images...")
        
        # Generate QA pairs
        all_qa_pairs = []
        
        for image_path in tqdm(image_paths, desc="Generating QA pairs"):
            # Get relative path for lookup
            image_filename = os.path.basename(image_path)
            
            if image_filename not in labels:
                print(f"Warning: No label found for {image_filename}, skipping...")
                continue
            
            flower_name = labels[image_filename]
            
            # Generate QA pairs for this image
            qa_pairs = self.generate_qa_for_image(
                image_path,
                flower_name,
                num_questions=qa_per_image
            )
            
            all_qa_pairs.extend(qa_pairs)
        
        # Create DataFrame
        df = pd.DataFrame(all_qa_pairs)
        
        # Save to CSV
        df.to_csv(output_csv, index=False)
        print(f"\nDataset saved to: {output_csv}")
        print(f"Total QA pairs: {len(df)}")
        
        return df
    
    def generate_statistics(self, df: pd.DataFrame, output_json: str):
        """
        Generate dataset statistics.
        
        Args:
            df: DataFrame with VQA pairs
            output_json: Path to save statistics JSON
        """
        stats = {
            'total_images': len(df['image_path'].unique()),
            'total_qa_pairs': len(df),
            'avg_qa_per_image': len(df) / len(df['image_path'].unique()),
            'question_types': df['question_type'].value_counts().to_dict(),
            'difficulty_levels': df['difficulty_level'].value_counts().to_dict(),
            'unique_questions': len(df['question'].unique()),
            'unique_answers': len(df['answer'].unique()),
            'flower_categories_covered': len(df['flower_category'].unique()),
            'avg_answer_length': df['answer'].str.len().mean(),
        }
        
        with open(output_json, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"\nStatistics saved to: {output_json}")
        print("\nDataset Statistics:")
        print(f"  Total images: {stats['total_images']}")
        print(f"  Total QA pairs: {stats['total_qa_pairs']}")
        print(f"  Avg QA per image: {stats['avg_qa_per_image']:.1f}")
        print(f"  Unique questions: {stats['unique_questions']}")
        print(f"  Flower categories: {stats['flower_categories_covered']}")


def main():
    """Main execution function."""
    # Configuration
    IMAGE_DIR = "oxford_flowers_102/jpg"  # Update this path
    LABELS_FILE = "oxford_flowers_102/labels.json"  # Update this path
    OUTPUT_CSV = "botany_vqa_grounded.csv"
    STATS_JSON = "dataset_statistics.json"
    
    # For testing, process only 100 images first
    NUM_IMAGES = 100  # Set to None to process all images
    QA_PER_IMAGE = 10
    
    # Initialize generator
    generator = BotanyVQAGenerator()
    
    # Generate dataset
    df = generator.generate_dataset(
        image_dir=IMAGE_DIR,
        labels_file=LABELS_FILE,
        output_csv=OUTPUT_CSV,
        num_images=NUM_IMAGES,
        qa_per_image=QA_PER_IMAGE
    )
    
    # Generate statistics
    generator.generate_statistics(df, STATS_JSON)
    
    # Run validation
    print("\nRunning validation checks...")
    validation_results = generator.validator.run_all_validations(df)
    report = generator.validator.generate_validation_report(validation_results)
    print("\n" + report)
    
    # Save validation report
    with open("validation_report.txt", "w") as f:
        f.write(report)
    print("Validation report saved to: validation_report.txt")


if __name__ == "__main__":
    main()
