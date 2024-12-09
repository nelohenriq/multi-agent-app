from .agent_base import AgentBase
from typing import Dict, List, Union
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class MarketDataValidatorAgent(AgentBase):
    def __init__(self, max_retries=2, verbose=True):
        super().__init__(
            name="MarketDataValidatorAgent", max_retries=max_retries, verbose=verbose
        )

    def _validate_data_freshness(self, timestamps: List[str], max_age_minutes: int = 15) -> Dict:
        """Validate the freshness of market data"""
        if not timestamps:
            return {'is_fresh': False, 'age_minutes': None}
            
        # Convert timestamps to datetime objects
        dates = [pd.to_datetime(ts) for ts in timestamps]
        latest = max(dates)
        now = pd.Timestamp.now()
        
        age_minutes = (now - latest).total_seconds() / 60
        return {
            'is_fresh': age_minutes <= max_age_minutes,
            'age_minutes': round(age_minutes, 2)
        }

    def _validate_price_consistency(self, prices: List[float]) -> Dict:
        """Check for price consistency and anomalies"""
        if not prices or len(prices) < 2:
            return {'is_consistent': True, 'anomalies': []}
            
        prices = np.array(prices)
        
        # Calculate price changes
        price_changes = np.diff(prices) / prices[:-1]
        
        # Detect anomalies (sudden large changes)
        anomaly_threshold = 0.1  # 10% change
        anomalies = []
        for i, change in enumerate(price_changes):
            if abs(change) > anomaly_threshold:
                anomalies.append({
                    'index': i + 1,
                    'change_percent': round(change * 100, 2),
                    'price_before': prices[i],
                    'price_after': prices[i + 1]
                })
        
        return {
            'is_consistent': len(anomalies) == 0,
            'anomalies': anomalies
        }

    def _validate_volume_profile(self, volumes: List[float]) -> Dict:
        """Analyze trading volume patterns"""
        if not volumes:
            return {'volume_score': 0, 'issues': ['No volume data available']}
            
        volumes = np.array(volumes)
        
        # Calculate basic statistics
        avg_volume = np.mean(volumes)
        std_volume = np.std(volumes)
        zero_volumes = np.sum(volumes == 0)
        
        issues = []
        if zero_volumes > len(volumes) * 0.1:  # More than 10% zero volumes
            issues.append('High number of zero volume periods detected')
        
        # Check for unusual volume spikes
        volume_spikes = np.where(volumes > avg_volume + 2 * std_volume)[0]
        if len(volume_spikes) > 0:
            issues.append(f'Unusual volume spikes detected at indices: {volume_spikes.tolist()}')
        
        # Calculate volume consistency score
        volume_cv = std_volume / avg_volume if avg_volume > 0 else float('inf')
        volume_score = max(0, min(1, 1 - volume_cv))
        
        return {
            'volume_score': round(volume_score, 2),
            'issues': issues
        }

    def execute(self, market_data: Dict) -> Dict:
        """
        Validate market data quality and consistency
        Returns a detailed validation report
        """
        # Extract data points
        timestamps = market_data.get('timestamps', [])
        prices = market_data.get('prices', [])
        volumes = market_data.get('volumes', [])
        
        # Run validations
        freshness_check = self._validate_data_freshness(timestamps)
        price_check = self._validate_price_consistency(prices)
        volume_check = self._validate_volume_profile(volumes)
        
        # Calculate overall quality score (1-5 scale)
        quality_components = [
            freshness_check['is_fresh'],
            not price_check['anomalies'],
            volume_check['volume_score']
        ]
        quality_score = round(sum(float(c) for c in quality_components if c is not None) 
                            / len(quality_components) * 5)
        
        validation_report = {
            'quality_score': quality_score,
            'data_freshness': freshness_check,
            'price_analysis': price_check,
            'volume_analysis': volume_check,
            'recommendations': []
        }
        
        # Add recommendations based on validation results
        if not freshness_check['is_fresh']:
            validation_report['recommendations'].append(
                f"Data is {freshness_check['age_minutes']} minutes old - consider refreshing"
            )
        
        if price_check['anomalies']:
            validation_report['recommendations'].append(
                "Price anomalies detected - verify market conditions"
            )
            
        if volume_check['issues']:
            validation_report['recommendations'].extend(
                f"Volume issue: {issue}" for issue in volume_check['issues']
            )
        
        return validation_report
