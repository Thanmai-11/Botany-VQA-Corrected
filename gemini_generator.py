"""
Gemini API generator for image-grounded Botany-VQA.
Uses Google's Gemini models to generate highly accurate QA pairs from images.
No GPU required!
"""

import os
import time
import json
import logging
import pandas as pd
from PIL import Image
from tqdm import tqdm
from typing import List, Dict
import google.generativeai as genai
from google.api_core import exceptions

from question_templates import QuestionGenerator
from visual_feature_extractor import VisualFeatureExtractor
from vqa_validator import VQAValidator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GeminiVQAGenerator:
    """Generate image-grounded VQA dataset using Google Gemini models."""
    
    def __init__(self, api_key: str, model_name: str = "gemini-2.0-flash"):
        """
        Initialize the Gemini dataset generator.
        
        Args:
            api_key: Google AI Studio API Key
            model_name: Gemini model to use (default: 'gemini-2.0-flash')
        """
        if not api_key:
            raise ValueError("API Key is required. Get one at https://aistudio.google.com/")
            
        genai.configure(api_key=api_key)
        
        # Debug: List available models to help with 404 errors
        try:
            logger.info("Verifying available models...")
            # genai.list_models() returns an iterable
            available_models = [m.name.replace('models/', '') for m in genai.list_models()]
            logger.info(f"Available models: {available_models}")
            
            # auto-correct common aliases and upgrade to 2.0 if 1.5 not found
            if model_name not in available_models:
                # Prefer 2.0 Flash
                if "gemini-2.0-flash" in available_models:
                    model_name = "gemini-2.0-flash"
                    logger.info(f"-> Switching to newer model: {model_name}")
                elif "gemini-2.0-flash-exp" in available_models:
                     model_name = "gemini-2.0-flash-exp"
                     logger.info(f"-> Switching to newer model: {model_name}")
                elif "gemini-1.5-flash-001" in available_models:
                    model_name = "gemini-1.5-flash-001"
                    logger.info(f"-> Auto-corrected model name to: {model_name}")
                elif "gemini-1.5-flash-latest" in available_models:
                    model_name = "gemini-1.5-flash-latest"
                    logger.info(f"-> Auto-corrected model name to: {model_name}")

            # Check if requested model exists (handling 'models/' prefix)
            if model_name not in available_models:
                logger.warning(f"⚠️ Requested model '{model_name}' not found in available list!")
                logger.warning("Trying to find a valid vision model...")
                
                # Auto-fallback to any valid flush/pro model (preferring 2.0, then 1.5)
                fallback_found = False
                # Prioritize 2.0 models
                for m in available_models:
                    if 'gemini-2.0-flash' in m:
                        model_name = m
                        logger.info(f"-> Switching to valid model: {model_name}")
                        fallback_found = True
                        break
                
                if not fallback_found:
                    for m in available_models:
                        if 'gemini-1.5-flash' in m and 'vision' not in m: 
                            model_name = m
                            logger.info(f"-> Switching to valid model: {model_name}")
                            fallback_found = True
                            break
                            
                if not fallback_found:
                     # Try Pro as last resort
                    for m in available_models:
                        if 'pro' in m and 'vision' not in m:
                            model_name = m
                            logger.info(f"-> Switching to valid model: {model_name}")
                            fallback_found = True
                            break
                            
                # HARD FALLBACK: If we still haven't found a valid model in the list,
                # force a known-good alias instead of crashing with the original name.
                if not fallback_found:
                    # If user has 2.0 access but listing failed for some reason, guess 2.0
                    model_name = "gemini-2.0-flash"
                    logger.warning(f"-> Could not find model in list. Forcing hard fallback to: {model_name}")

        except Exception as e:
            logger.warning(f"Could not verify models (API key issue?): {e}")

        self.model = genai.GenerativeModel(model_name)
        self.model_name = model_name
        
        logger.info(f"Initialized Gemini Generator using model: {model_name}")
        
        # Initialize utilities
        self.feature_extractor = VisualFeatureExtractor()
        self.question_generator = None
        self.validator = None
        
        # Rate limiting (requests per minute)
        # Flash: 15 RPM (free), Pro: 2 RPM (free)
        self.rpm_limit = 14 if 'flash' in model_name else 2
        self.delay_per_request = 60.0 / self.rpm_limit
        self.last_request_time = 0

    def _wait_for_rate_limit(self):
        """Ensure we respect rate limits."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.delay_per_request:
            sleep_time = self.delay_per_request - time_since_last
            time.sleep(sleep_time)
        self.last_request_time = time.time()

    def ask_questions_batch(self, image_path: str, questions: List[str]) -> Dict[str, str]:
        """
        Ask multiple questions about an image in a single API call.
        Returns a dictionary mapping questions to answers.
        """
        try:
            self._wait_for_rate_limit()
            
            # Load image
            img = Image.open(image_path)
            
            # Construct batch prompt
            prompt = (
                "Analyze this flower image and answer the following questions.\n"
                "Provide the output STRICTLY as a raw JSON object where keys are the questions and values are the concise answers.\n"
                "Do not use Markdown formatting (no ```json blocks).\n\n"
                "Questions:\n"
            )
            for q in questions:
                prompt += f"- {q}\n"
            
            # Generate response
            response = self.model.generate_content([prompt, img])
            text_response = response.text.strip()
            
            # Clean up potential markdown formatting if the model disregards instructions
            if text_response.startswith("```"):
                text_response = text_response.replace("```json", "").replace("```", "").strip()
            
            try:
                answers_dict = json.loads(text_response)
                return answers_dict
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse JSON response for {image_path}: {text_response[:100]}...")
                # Fallback: empty dict, will result in individual retires or errors
                return {}
            
        except exceptions.ResourceExhausted:
            logger.warning("Rate limit hit in batch mode. Waiting 60 seconds...")
            time.sleep(60)
            return self.ask_questions_batch(image_path, questions)  # Retry
        except Exception as e:
            logger.error(f"Error processing batch for {image_path}: {e}")
            return {}

    def generate_qa_for_image(self, image_path: str, flower_name: str, num_questions: int = 10) -> List[Dict[str, any]]:
        """Generate QA pairs for a single image using batch processing."""
        questions_meta = self.question_generator.generate_diverse_questions(flower_name, num_questions)
        qa_pairs = []
        
        # separate questions that need API calls
        api_questions = []
        api_indices = []
        
        # Pre-process questions
        for idx, q_meta in enumerate(questions_meta):
            if q_meta['question_type'] == 'yes_no' and q_meta.get('expected_answer_contains'):
                # Already have answer
                continue
            api_questions.append(q_meta['question'])
            api_indices.append(idx)
        
        # Make batch API call
        if api_questions:
            answers_map = self.ask_questions_batch(image_path, api_questions)
        else:
            answers_map = {}
            
        for idx, q_meta in enumerate(questions_meta):
            question = q_meta['question']
            
            # Determine answer
            if q_meta['question_type'] == 'yes_no' and q_meta.get('expected_answer_contains'):
                answer = q_meta['expected_answer_contains'].capitalize()
            else:
                answer = answers_map.get(question, "Error generation failed")
                if answer == "Error generation failed":
                    # Optional: Retry individually or just log
                    pass
            
            qa_pairs.append({
                'image_path': image_path,
                'question': question,
                'answer': answer,
                'question_type': q_meta['question_type'],
                'difficulty_level': q_meta['difficulty_level'],
                'flower_category': flower_name,
                'model_used': self.model_name
            })
            
        return qa_pairs

    def generate_dataset(self, image_dir: str, labels_file: str, output_csv: str, num_images: int = None, qa_per_image: int = 10):
        """
        Generate the complete VQA dataset.
        Supports resuming from existing CSV to handle API limits.
        """
        logger.info("Loading flower labels...")
        labels = self.load_oxford_flowers_labels(labels_file)
        
        self.question_generator = QuestionGenerator(list(set(labels.values())))
        self.validator = VQAValidator(list(set(labels.values())))
        
        # Check for existing progress
        existing_images = set()
        all_qa_pairs = []
        
        if os.path.exists(output_csv):
            try:
                existing_df = pd.read_csv(output_csv)
                existing_images = set(existing_df['image_path'].unique())
                all_qa_pairs = existing_df.to_dict('records')
                logger.info(f"Found existing dataset with {len(existing_images)} images. Resuming...")
            except Exception as e:
                logger.warning(f"Could not load existing CSV: {e}. Starting fresh.")
        
        # Get all potential images
        image_paths = sorted([
            os.path.join(image_dir, f) for f in os.listdir(image_dir) 
            if f.endswith(('.jpg', '.jpeg', '.png'))
        ])
        
        # Filter out already processed images
        images_to_process = [img for img in image_paths if img not in existing_images]
        
        # Apply num_images limit (counting what's already done)
        if num_images:
            remaining_slots = start_limit = num_images - len(existing_images)
            if remaining_slots <= 0:
                logger.info(f"Target of {num_images} images already reached!")
                return pd.DataFrame(all_qa_pairs)
            images_to_process = images_to_process[:remaining_slots]
            
        logger.info(f"Processing {len(images_to_process)} new images using {self.model_name}...")
        
        new_qa_pairs = []
        for i, image_path in enumerate(tqdm(images_to_process, desc="Generating QA pairs")):
            image_filename = os.path.basename(image_path)
            if image_filename not in labels:
                continue
                
            flower_name = labels[image_filename]
            qa_pairs = self.generate_qa_for_image(image_path, flower_name, qa_per_image)
            
            new_qa_pairs.extend(qa_pairs)
            all_qa_pairs.extend(qa_pairs)
            
            # Periodic save (every 10 images)
            if (i + 1) % 10 == 0:
                pd.DataFrame(all_qa_pairs).to_csv(output_csv, index=False)
        
        # Final save
        df = pd.DataFrame(all_qa_pairs)
        df.to_csv(output_csv, index=False)
        logger.info(f"Dataset saved to: {output_csv} with {len(df)} QA pairs ({len(df['image_path'].unique())} images)")
        return df

    def generate_statistics(self, df: pd.DataFrame, output_json: str):
        stats = {
            'total_images': len(df['image_path'].unique()),
            'total_qa_pairs': len(df),
            'unique_questions': len(df['question'].unique()),
            'unique_answers': len(df['answer'].unique()),
            'model': self.model_name
        }
        with open(output_json, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"Statistics saved to {output_json}")

if __name__ == "__main__":
    # Example usage
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("Please set GOOGLE_API_KEY environment variable")
    else:
        generator = GeminiVQAGenerator(api_key=api_key)
        # Add call to generate_dataset here for testing
