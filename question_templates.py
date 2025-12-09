"""
Question templates and generation strategies for Botany-VQA dataset.
Defines hierarchical question types with varying difficulty levels.
"""

from typing import List, Dict, Tuple
import random

class QuestionTemplates:
    """Question generation templates organized by type and difficulty level."""
    
    # Level 1: Basic Identification
    IDENTIFICATION = [
        "What type of flower is this?",
        "What is the name of this flower?",
        "Can you identify this flower?",
        "What flower species is shown in the image?",
    ]
    
    # Level 1-2: Visual Attributes
    COLOR_QUESTIONS = [
        "What color are the petals?",
        "What is the primary color of this flower?",
        "Describe the color of the petals.",
        "What color is the center of the flower?",
    ]
    
    SHAPE_QUESTIONS = [
        "What is the shape of the petals?",
        "Describe the petal shape.",
        "Are the petals rounded or pointed?",
        "What is the overall shape of the flower?",
    ]
    
    TEXTURE_QUESTIONS = [
        "What is the texture of the petals?",
        "Do the petals appear smooth or textured?",
        "Describe the surface of the petals.",
    ]
    
    STRUCTURE_QUESTIONS = [
        "Does this flower have a visible center?",
        "Are the stamens visible?",
        "Does this flower have visible pistils?",
        "What structures are visible in the flower center?",
    ]
    
    # Level 2: Counting
    COUNTING_QUESTIONS = [
        "How many petals are visible?",
        "How many petals does this flower have?",
        "Count the number of visible petals.",
        "How many flowers are in the image?",
    ]
    
    # Level 2: Spatial/Compositional
    SPATIAL_QUESTIONS = [
        "Is the flower fully open or partially closed?",
        "What is the background of the image?",
        "Is this a close-up or distant view?",
        "What is the orientation of the flower?",
    ]
    
    # Level 3: Comparative (Yes/No)
    @staticmethod
    def generate_yes_no_questions(flower_name: str, other_flowers: List[str]) -> List[Tuple[str, str]]:
        """Generate yes/no questions with answers."""
        questions = []
        
        # Positive question
        questions.append((f"Is this a {flower_name}?", "Yes"))
        
        # Negative questions (random other flowers)
        for other in random.sample(other_flowers, min(2, len(other_flowers))):
            questions.append((f"Is this a {other}?", "No"))
        
        return questions
    
    # Level 3-4: Reasoning
    REASONING_QUESTIONS = [
        "Based on the petal arrangement, what type of symmetry does this flower have?",
        "What family does this flower belong to?",
        "What are the distinguishing features of this flower?",
        "How would you describe the bloom stage of this flower?",
    ]
    
    SEASONAL_QUESTIONS = [
        "What season would this flower typically bloom in?",
        "Is this a spring, summer, fall, or winter blooming flower?",
    ]
    
    @staticmethod
    def get_question_set(difficulty_level: int = None) -> List[str]:
        """Get questions based on difficulty level (1-4)."""
        all_questions = {
            1: QuestionTemplates.IDENTIFICATION + QuestionTemplates.COLOR_QUESTIONS,
            2: QuestionTemplates.SHAPE_QUESTIONS + QuestionTemplates.COUNTING_QUESTIONS + QuestionTemplates.SPATIAL_QUESTIONS,
            3: QuestionTemplates.STRUCTURE_QUESTIONS + QuestionTemplates.TEXTURE_QUESTIONS,
            4: QuestionTemplates.REASONING_QUESTIONS + QuestionTemplates.SEASONAL_QUESTIONS,
        }
        
        if difficulty_level:
            return all_questions.get(difficulty_level, [])
        
        # Return all questions
        return [q for qs in all_questions.values() for q in qs]
    
    @staticmethod
    def categorize_question(question: str) -> Tuple[str, int]:
        """Categorize a question by type and difficulty level."""
        question_lower = question.lower()
        
        if any(word in question_lower for word in ["what type", "what is the name", "identify", "species"]):
            return ("identification", 1)
        elif any(word in question_lower for word in ["color", "colour"]):
            return ("visual_attribute_color", 1)
        elif any(word in question_lower for word in ["shape"]):
            return ("visual_attribute_shape", 2)
        elif any(word in question_lower for word in ["texture", "surface"]):
            return ("visual_attribute_texture", 3)
        elif any(word in question_lower for word in ["how many", "count"]):
            return ("counting", 2)
        elif any(word in question_lower for word in ["center", "stamen", "pistil"]):
            return ("structure", 3)
        elif any(word in question_lower for word in ["is this a", "is this"]):
            return ("yes_no", 3)
        elif any(word in question_lower for word in ["background", "orientation", "open", "closed"]):
            return ("spatial", 2)
        elif any(word in question_lower for word in ["symmetry", "family", "distinguishing", "bloom stage"]):
            return ("reasoning", 4)
        elif any(word in question_lower for word in ["season", "spring", "summer", "fall", "winter"]):
            return ("seasonal", 4)
        else:
            return ("other", 2)


class QuestionGenerator:
    """Generate diverse question sets for each image."""
    
    def __init__(self, flower_categories: List[str]):
        """
        Initialize question generator.
        
        Args:
            flower_categories: List of all flower category names
        """
        self.flower_categories = flower_categories
        self.templates = QuestionTemplates()
    
    def generate_diverse_questions(self, flower_name: str, num_questions: int = 10) -> List[Dict[str, any]]:
        """
        Generate a diverse set of questions for an image.
        
        Args:
            flower_name: Ground truth flower name
            num_questions: Number of questions to generate
            
        Returns:
            List of question dictionaries with metadata
        """
        questions = []
        
        # Always include identification (Level 1)
        questions.append({
            "question": random.choice(self.templates.IDENTIFICATION),
            "question_type": "identification",
            "difficulty_level": 1,
            "expected_answer_contains": flower_name.lower()
        })
        
        # Add color question (Level 1)
        questions.append({
            "question": random.choice(self.templates.COLOR_QUESTIONS),
            "question_type": "visual_attribute_color",
            "difficulty_level": 1,
            "expected_answer_contains": None
        })
        
        # Add counting question (Level 2)
        questions.append({
            "question": random.choice(self.templates.COUNTING_QUESTIONS),
            "question_type": "counting",
            "difficulty_level": 2,
            "expected_answer_contains": None
        })
        
        # Add shape question (Level 2)
        questions.append({
            "question": random.choice(self.templates.SHAPE_QUESTIONS),
            "question_type": "visual_attribute_shape",
            "difficulty_level": 2,
            "expected_answer_contains": None
        })
        
        # Add structure question (Level 3)
        questions.append({
            "question": random.choice(self.templates.STRUCTURE_QUESTIONS),
            "question_type": "structure",
            "difficulty_level": 3,
            "expected_answer_contains": None
        })
        
        # Add yes/no questions (Level 3)
        other_flowers = [f for f in self.flower_categories if f.lower() != flower_name.lower()]
        yes_no_qs = self.templates.generate_yes_no_questions(flower_name, other_flowers)
        
        for q, a in yes_no_qs[:2]:  # Add 2 yes/no questions
            questions.append({
                "question": q,
                "question_type": "yes_no",
                "difficulty_level": 3,
                "expected_answer_contains": a.lower()
            })
        
        # Add spatial question (Level 2)
        questions.append({
            "question": random.choice(self.templates.SPATIAL_QUESTIONS),
            "question_type": "spatial",
            "difficulty_level": 2,
            "expected_answer_contains": None
        })
        
        # Add reasoning question if we need more (Level 4)
        if len(questions) < num_questions:
            questions.append({
                "question": random.choice(self.templates.REASONING_QUESTIONS),
                "question_type": "reasoning",
                "difficulty_level": 4,
                "expected_answer_contains": None
            })
        
        return questions[:num_questions]
