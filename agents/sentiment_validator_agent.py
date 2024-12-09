from .agent_base import AgentBase
import spacy
import os
from textblob import TextBlob
from typing import Dict, List, Union
import re


class SentimentValidatorAgent(AgentBase):
    def __init__(self, max_retries=2, verbose=True):
        super().__init__(
            name="SentimentValidatorAgent", max_retries=max_retries, verbose=verbose
        )
        # Load spaCy model
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            self.logger.info("Downloading spaCy model...")
            os.system("python -m spacy download en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")

    def _validate_price_terms(self, text: str, sentiment_score: float) -> Dict[str, Union[float, List[str]]]:
        """Validate sentiment against price-related terms"""
        price_up_terms = ['surge', 'jump', 'rise', 'gain', 'bull', 'breakout', 'rally']
        price_down_terms = ['drop', 'fall', 'decline', 'bear', 'crash', 'plunge', 'dip']
        
        # Convert text to lowercase for matching
        text_lower = text.lower()
        
        # Count occurrences
        up_mentions = sum(1 for term in price_up_terms if term in text_lower)
        down_mentions = sum(1 for term in price_down_terms if term in text_lower)
        
        # Calculate expected sentiment direction
        expected_direction = 1 if up_mentions > down_mentions else -1 if down_mentions > up_mentions else 0
        actual_direction = 1 if sentiment_score > 0.1 else -1 if sentiment_score < -0.1 else 0
        
        # Check consistency
        is_consistent = expected_direction == actual_direction
        
        return {
            'consistency_score': 1.0 if is_consistent else 0.0,
            'price_terms_found': {
                'bullish': [term for term in price_up_terms if term in text_lower],
                'bearish': [term for term in price_down_terms if term in text_lower]
            }
        }

    def _validate_entity_sentiment(self, text: str, entities: List[str], sentiment_score: float) -> Dict:
        """Validate sentiment for specific entities"""
        doc = self.nlp(text)
        entity_sentiments = {}
        
        for entity in entities:
            # Find sentences containing the entity
            entity_sentences = [sent.text for sent in doc.sents 
                              if entity.lower() in sent.text.lower()]
            
            if entity_sentences:
                # Calculate average sentiment for entity mentions
                entity_sentiment = sum(TextBlob(sent).sentiment.polarity 
                                    for sent in entity_sentences) / len(entity_sentences)
                
                entity_sentiments[entity] = {
                    'local_sentiment': round(entity_sentiment, 2),
                    'context': entity_sentences
                }
        
        return entity_sentiments

    def execute(self, text: str, sentiment_analysis: Dict) -> Dict:
        """
        Validate sentiment analysis results
        Returns a detailed validation report
        """
        # Get TextBlob sentiment for comparison
        blob_sentiment = TextBlob(text).sentiment.polarity
        
        # Validate price terms consistency
        price_validation = self._validate_price_terms(text, sentiment_analysis['score'])
        
        # Validate entity-specific sentiment
        entity_validation = self._validate_entity_sentiment(
            text, 
            sentiment_analysis.get('entities', []),
            sentiment_analysis['score']
        )
        
        # Calculate sentiment deviation score
        sentiment_deviation = abs(blob_sentiment - sentiment_analysis['score'])
        deviation_score = max(0, 1 - sentiment_deviation)
        
        # Calculate overall confidence score (1-5 scale)
        confidence_score = round(
            (deviation_score * 0.4 +                 # Agreement with TextBlob
             price_validation['consistency_score'] * 0.4 +  # Price term consistency
             (1 if entity_validation else 0) * 0.2   # Entity coverage
            ) * 5
        )
        
        validation_report = {
            'confidence_score': confidence_score,
            'textblob_comparison': {
                'textblob_sentiment': round(blob_sentiment, 2),
                'sentiment_deviation': round(sentiment_deviation, 2)
            },
            'price_terms_analysis': price_validation,
            'entity_sentiments': entity_validation,
            'recommendations': []
        }
        
        # Add recommendations based on validation results
        if sentiment_deviation > 0.5:
            validation_report['recommendations'].append(
                "Large deviation from baseline sentiment - review analysis"
            )
        if not price_validation['consistency_score']:
            validation_report['recommendations'].append(
                "Sentiment may not align with price movement terms"
            )
        if not entity_validation:
            validation_report['recommendations'].append(
                "Consider including entity-specific sentiment analysis"
            )
        
        return validation_report
