"""
Language Annotation Stage.
Generates text descriptions for segmented actions using LLMs.
"""

from typing import Dict, Any, List, Optional
import time
from .base import TimedStage
from ..config import AnnotationConfig


class LanguageAnnotationStage(TimedStage):
    """
    Stage 3: Language Annotation.
    
    Generates natural language descriptions for the segmented actions.
    In the full implementation, this calls an LLM (e.g., GPT-4) with
    the trajectory summary.
    
    For this implementation, we provide a structured interface that can
    call an external API or use a mock generator for testing.
    """
    
    def __init__(self, config: AnnotationConfig, logger=None):
        super().__init__(config, logger)
        self.llm_client = None

    def _do_initialize(self) -> None:
        """Initialize LLM client."""
        if self.config.llm_provider == "openai":
            # import openai
            # self.llm_client = openai.OpenAI(api_key=self.config.api_key)
            self.logger.info("Initialized OpenAI client (Mock)")
        elif self.config.llm_provider == "azure":
            self.logger.info("Initialized Azure OpenAI client (Mock)")
        else:
            self.logger.info("Using local/dummy annotation provider")

    def _do_process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Annotate episodes with text descriptions.
        """
        episodes = input_data.get("episodes", [])
        reconstruction = input_data.get("reconstruction", {})
        
        self.logger.info(f"Annotating {len(episodes)} episodes...")
        
        annotated_episodes = []
        
        for ep in episodes:
            # 1. Summarize motion for the prompt
            summary = self._summarize_motion(
                reconstruction, 
                ep["start_frame"], 
                ep["end_frame"], 
                ep["anno_type"]
            )
            
            # 2. Generate description
            description = self._generate_description(summary)
            
            # 3. Generate rephrasings (optional)
            rephrasings = self._generate_rephrasings(description)
            
            # 4. Attach to episode data structure matching the VITRA format
            # VITRA format: text: {'left': [(desc, (start, end))], 'right': [...]}
            
            # We create a new structure for the processed episode
            annotated_ep = ep.copy()
            annotated_ep["text"] = {
                "left": [],
                "right": []
            }
            annotated_ep["text_rephrase"] = {
                "left": [],
                "right": []
            }
            
            # Helper to create the tuple structure
            range_tuple = (ep["start_frame"], ep["end_frame"])
            
            hand = ep["anno_type"] # 'left' or 'right'
            annotated_ep["text"][hand].append((description, range_tuple))
            annotated_ep["text_rephrase"][hand].append((rephrasings, range_tuple))
            
            annotated_episodes.append(annotated_ep)
            
        return {
            **input_data,
            "episodes": annotated_episodes
        }

    def _summarize_motion(self, recon: Dict, start: int, end: int, hand: str) -> Dict[str, Any]:
        """Create a statistical summary of the motion for the prompt."""
        # Get trajectory slice
        # In a real implementation, utilize the actual recon data
        # Here we mock the summary extraction
        
        return {
            "hand_type": hand,
            "start_frame": start,
            "end_frame": end,
            "duration": end - start,
            # Placeholders
            "displacement": 0.5, 
            "direction": "forward",
            "finger_motion": "grasping"
        }

    def _generate_description(self, summary: Dict) -> str:
        """Call LLM to get main description."""
        prompt = self.config.action_prompt_template.format(
            hand_type=summary["hand_type"],
            start_pos="[0,0,0]", # Mock
            end_pos="[0.1, 0.2, 0.3]", # Mock
            displacement=summary["displacement"],
            duration=summary["duration"],
            direction=summary["direction"],
            finger_motion=summary["finger_motion"]
        )
        
        if self.config.llm_provider in ["openai", "azure"] and self.config.api_key:
            # Real API call would go here
            # response = self.llm_client.chat.completions.create(...)
            # return response.choices[0].message.content
             pass

        # Mock response based on hand
        return f"Move {summary['hand_type']} hand {summary['direction']}."

    def _generate_rephrasings(self, original: str) -> List[str]:
        """Call LLM to rephrase."""
        if self.config.num_rephrase <= 0:
            return []
            
        # Mock response
        return [f"{original} (Variant {i+1})" for i in range(self.config.num_rephrase)]
