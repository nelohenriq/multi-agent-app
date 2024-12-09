# agents/summarize_validator_agent.py

from .agent_base import AgentBase
import spacy
import os
from textblob import TextBlob
from typing import Dict, List, Tuple


class SummarizeValidatorAgent(AgentBase):
    def __init__(self, max_retries=2, verbose=True):
        super().__init__(
            name="SummarizeValidatorAgent", max_retries=max_retries, verbose=verbose
        )
        # Load spaCy model
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            self.logger.info("Downloading spaCy model...")
            os.system("python -m spacy download en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")

    def _validate_entities(self, original_doc, summary_entities: Dict) -> Dict:
        """Validate that important entities from original text are present in summary"""
        original_entities = {
            'conditions': [],
            'medications': [],
            'procedures': [],
            'measurements': [],
            'dates': [],
            'organizations': []
        }
        
        # Extract entities from original text
        for ent in original_doc.ents:
            if ent.label_ in ['CONDITION', 'DISEASE']:
                original_entities['conditions'].append(ent.text.lower())
            elif ent.label_ == 'CHEMICAL':
                original_entities['medications'].append(ent.text.lower())
            elif ent.label_ == 'PROCEDURE':
                original_entities['procedures'].append(ent.text.lower())
            elif ent.label_ in ['QUANTITY', 'PERCENT']:
                original_entities['measurements'].append(ent.text.lower())
            elif ent.label_ == 'DATE':
                original_entities['dates'].append(ent.text.lower())
            elif ent.label_ == 'ORG':
                original_entities['organizations'].append(ent.text.lower())

        # Compare entities
        entity_coverage = {}
        for category in original_entities:
            if not original_entities[category]:
                entity_coverage[category] = 1.0  # Perfect score if no entities of this type
                continue
                
            summary_ents = [e.lower() for e in summary_entities[category]]
            matched = sum(1 for e in original_entities[category] if any(s in e or e in s for s in summary_ents))
            entity_coverage[category] = matched / len(original_entities[category])

        return entity_coverage

    def _validate_sentiment_consistency(self, original_text: str, summary_sentiment: Dict) -> float:
        """Check if summary sentiment aligns with original text sentiment"""
        original_blob = TextBlob(original_text)
        original_sentiment = {
            'polarity': round(original_blob.sentiment.polarity, 2),
            'subjectivity': round(original_blob.sentiment.subjectivity, 2)
        }
        
        # Calculate sentiment similarity (1 = perfect match, 0 = complete opposite)
        polarity_diff = abs(original_sentiment['polarity'] - summary_sentiment['polarity'])
        subjectivity_diff = abs(original_sentiment['subjectivity'] - summary_sentiment['subjectivity'])
        
        # Weight polarity more heavily than subjectivity
        sentiment_score = 1 - ((polarity_diff * 0.7) + (subjectivity_diff * 0.3))
        return max(0, sentiment_score)

    def execute(self, original_text: str, summary: Dict) -> Dict:
        """
        Validate the structured summary against the original text
        Returns a detailed validation report
        """
        # Process original text with spaCy
        original_doc = self.nlp(original_text)
        
        # Validate entity coverage
        entity_coverage = self._validate_entities(original_doc, summary['entities'])
        
        # Validate sentiment consistency
        sentiment_score = self._validate_sentiment_consistency(original_text, summary['sentiment'])
        
        # Calculate content density score
        content_density = min(len(summary['key_points']) / (len(TextBlob(original_text).sentences) * 0.3), 1.0)
        
        # Calculate overall quality score (1-5 scale)
        entity_score = sum(entity_coverage.values()) / len(entity_coverage)
        quality_score = round(
            (entity_score * 0.4 +      # Entity coverage
             sentiment_score * 0.3 +    # Sentiment accuracy
             content_density * 0.3)     # Content density
            * 5
        )
        
        validation_report = {
            'quality_score': quality_score,  # 1-5 scale
            'entity_coverage': entity_coverage,  # Detailed entity coverage
            'sentiment_accuracy': round(sentiment_score, 2),
            'content_density': round(content_density, 2),
            'recommendations': []
        }
        
        # Add specific recommendations based on scores
        if entity_score < 0.8:
            validation_report['recommendations'].append(
                "Consider including more key medical entities from the original text"
            )
        if sentiment_score < 0.8:
            validation_report['recommendations'].append(
                "Review sentiment alignment with original text"
            )
        if content_density < 0.6:
            validation_report['recommendations'].append(
                "Summary might be too brief relative to original content"
            )
        
        return validation_report
