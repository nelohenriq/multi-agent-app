from .agent_base import AgentBase
import spacy
import os
from textblob import TextBlob
from typing import Dict, List, Tuple

class SummarizeTool(AgentBase):
    def __init__(self, max_retries=2, verbose=True):
        super().__init__(name="SummarizeTool", max_retries=max_retries, verbose=verbose)
        # Load spaCy model for English
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            # If model not found, download it
            self.logger.info("Downloading spaCy model...")
            os.system("python -m spacy download en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")

    def _extract_key_info(self, text: str) -> Dict:
        """Extract key information using spaCy"""
        doc = self.nlp(text)
        
        # Extract named entities
        entities = {
            'conditions': [],    # Medical conditions
            'medications': [],   # Medications
            'procedures': [],    # Medical procedures
            'measurements': [],  # Measurements and values
            'dates': [],        # Temporal information
            'organizations': [] # Healthcare organizations
        }
        
        for ent in doc.ents:
            if ent.label_ in ['CONDITION', 'DISEASE']:
                entities['conditions'].append(ent.text)
            elif ent.label_ == 'CHEMICAL':
                entities['medications'].append(ent.text)
            elif ent.label_ == 'PROCEDURE':
                entities['procedures'].append(ent.text)
            elif ent.label_ in ['QUANTITY', 'PERCENT']:
                entities['measurements'].append(ent.text)
            elif ent.label_ == 'DATE':
                entities['dates'].append(ent.text)
            elif ent.label_ == 'ORG':
                entities['organizations'].append(ent.text)
        
        return entities

    def _get_key_sentences(self, text: str, num_sentences: int = 3) -> List[str]:
        """Extract key sentences using TextBlob"""
        blob = TextBlob(text)
        
        # Score sentences by importance
        sentence_scores = []
        for sentence in blob.sentences:
            # Score based on length (not too short, not too long)
            length_score = min(len(sentence.words) / 20.0, 1.0)
            
            # Score based on presence of medical terms
            medical_terms = ['patient', 'treatment', 'diagnosis', 'symptoms', 'disease', 
                           'condition', 'medical', 'clinical', 'health', 'care']
            term_score = sum(1 for word in sentence.words if word.lower() in medical_terms) / len(sentence.words)
            
            # Score based on sentence position (earlier sentences often more important)
            position_score = 1.0 - (blob.sentences.index(sentence) / len(blob.sentences))
            
            # Combine scores
            total_score = (length_score + term_score + position_score) / 3
            sentence_scores.append((sentence, total_score))
        
        # Sort by score and get top sentences
        sentence_scores.sort(key=lambda x: x[1], reverse=True)
        return [str(sent[0]) for sent in sentence_scores[:num_sentences]]

    def execute(self, text: str) -> Dict:
        """
        Summarize medical text using NLP techniques
        Returns a dictionary with key information and summary
        """
        # Extract entities and key information
        entities = self._extract_key_info(text)
        
        # Get key sentences
        key_sentences = self._get_key_sentences(text)
        
        # Create structured summary
        summary = {
            'key_points': key_sentences,
            'entities': entities,
            'statistics': {
                'word_count': len(text.split()),
                'sentence_count': len(TextBlob(text).sentences),
                'medical_terms_found': len(entities['conditions']) + 
                                     len(entities['medications']) + 
                                     len(entities['procedures'])
            }
        }
        
        # Add sentiment analysis for context
        blob = TextBlob(text)
        summary['sentiment'] = {
            'polarity': round(blob.sentiment.polarity, 2),  # -1 to 1
            'subjectivity': round(blob.sentiment.subjectivity, 2)  # 0 to 1
        }
        
        return summary
