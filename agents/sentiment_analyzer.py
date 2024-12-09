from .agent_base import AgentBase
from typing import List, Dict
import os
import json
import re
import spacy
from textblob import TextBlob
from pathlib import Path

class SentimentAnalyzerTool(AgentBase):
    def __init__(self, max_retries=2, verbose=True):
        super().__init__(name="SentimentAnalyzerTool", max_retries=max_retries, verbose=verbose)
        # Load spaCy model for English
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            # If model not found, download it
            self.logger.info("Downloading spaCy model...")
            os.system("python -m spacy download en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")

    def _analyze_sentiment(self, text: str) -> Dict:
        """
        Analyze sentiment using spaCy and TextBlob
        Returns score between -1 and 1, and a brief explanation
        """
        # Process text with spaCy
        doc = self.nlp(text)
        
        # Extract relevant entities and their context
        entities = []
        prices = []
        percentages = []
        
        for ent in doc.ents:
            if ent.label_ in ['MONEY', 'PERCENT']:
                if ent.label_ == 'MONEY':
                    prices.append(ent.text)
                else:
                    percentages.append(ent.text)
            elif ent.label_ in ['ORG', 'PRODUCT']:
                entities.append(ent.text)
        
        # Use TextBlob for sentiment scoring
        blob = TextBlob(text)
        
        # Get polarity score (-1 to 1)
        score = blob.sentiment.polarity
        
        # Adjust score based on price movements and percentages
        price_terms = ['surge', 'jump', 'rise', 'gain', 'bull', 'up']
        negative_terms = ['drop', 'fall', 'decline', 'bear', 'down', 'loss']
        
        # Look for price movement terms
        text_lower = text.lower()
        price_movement = 0
        for term in price_terms:
            if term in text_lower:
                price_movement += 0.2
        for term in negative_terms:
            if term in text_lower:
                price_movement -= 0.2
        
        # Combine scores with more weight on actual price movements
        final_score = (score * 0.3) + (price_movement * 0.7)
        final_score = max(min(final_score, 1.0), -1.0)  # Clamp between -1 and 1
        
        # Generate explanation
        explanation_parts = []
        if prices:
            explanation_parts.append(f"Price mentioned: {prices[0]}")
        if percentages:
            explanation_parts.append(f"Movement: {percentages[0]}")
        if not explanation_parts:
            if final_score > 0:
                explanation_parts.append("Positive sentiment in market discussion")
            elif final_score < 0:
                explanation_parts.append("Negative market signals detected")
            else:
                explanation_parts.append("Neutral market discussion")
        
        return {
            "score": round(final_score, 2),
            "explanation": " | ".join(explanation_parts)
        }

    def execute(self, news_items: List[Dict]) -> List[Dict]:
        """Analyze sentiment of news articles using spaCy and TextBlob"""
        analyzed_items = []
        
        for item in news_items:
            try:
                # Combine title and description for analysis
                text = f"{item.get('title', '')} {item.get('description', '')}"
                sentiment_data = self._analyze_sentiment(text)
                item['sentiment_analysis'] = sentiment_data
                
            except Exception as e:
                self.logger.error(f"Error analyzing sentiment: {e}")
                item['sentiment_analysis'] = {
                    'score': 0,
                    'explanation': 'Error analyzing sentiment'
                }
            
            analyzed_items.append(item)
            
        return analyzed_items
