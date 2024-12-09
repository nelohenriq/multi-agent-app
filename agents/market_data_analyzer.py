from .agent_base import AgentBase
import pandas as pd
import numpy as np
from typing import Dict, List, Union, Tuple
from scipy import stats
from datetime import datetime, timedelta


class MarketDataAnalyzer(AgentBase):
    def __init__(self, max_retries=2, verbose=True):
        super().__init__(
            name="MarketDataAnalyzer", max_retries=max_retries, verbose=verbose
        )
        self.trend_threshold = 0.05  # 5% change threshold for trend detection
        self.volatility_window = 14  # Days for volatility calculation
        self.ma_windows = [7, 14, 30]  # Moving average periods

    def _calculate_sma(self, data: np.array, period: int) -> float:
        """Calculate Simple Moving Average"""
        if len(data) < period:
            return None
        return np.mean(data[-period:])

    def _calculate_ema(self, data: np.array, period: int) -> float:
        """Calculate Exponential Moving Average"""
        if len(data) < period:
            return None
        weights = np.exp(np.linspace(-1., 0., period))
        weights /= weights.sum()
        return np.convolve(data, weights, mode='valid')[-1]

    def _calculate_rsi(self, prices: np.array, period: int = 14) -> float:
        """Calculate Relative Strength Index"""
        if len(prices) < period + 1:
            return None
            
        # Calculate price changes
        deltas = np.diff(prices)
        
        # Separate gains and losses
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        # Calculate average gains and losses
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return round(rsi, 2)

    def _calculate_macd(self, prices: np.array) -> Dict:
        """Calculate MACD (Moving Average Convergence Divergence)"""
        if len(prices) < 26:
            return None
            
        # Calculate EMAs
        ema12 = self._calculate_ema(prices, 12)
        ema26 = self._calculate_ema(prices, 26)
        
        if ema12 is None or ema26 is None:
            return None
            
        macd_line = ema12 - ema26
        signal_line = self._calculate_ema(np.array([macd_line]), 9)
        histogram = macd_line - signal_line if signal_line is not None else None
        
        return {
            'macd': round(macd_line, 4),
            'signal': round(signal_line, 4) if signal_line is not None else None,
            'histogram': round(histogram, 4) if histogram is not None else None
        }

    def _calculate_bollinger_bands(self, prices: np.array, period: int = 20) -> Dict:
        """Calculate Bollinger Bands"""
        if len(prices) < period:
            return None
            
        sma = self._calculate_sma(prices, period)
        std = np.std(prices[-period:])
        
        return {
            'upper': round(sma + (2 * std), 2),
            'middle': round(sma, 2),
            'lower': round(sma - (2 * std), 2)
        }

    def _calculate_technical_indicators(self, prices: np.array, volumes: np.array) -> Dict:
        """Calculate various technical indicators"""
        indicators = {}
        
        # Moving Averages
        for window in self.ma_windows:
            ma = self._calculate_sma(prices, window)
            indicators[f'MA_{window}'] = round(ma, 2) if ma is not None else None
            
        # RSI
        indicators['RSI'] = self._calculate_rsi(prices)
        
        # MACD
        indicators['MACD'] = self._calculate_macd(prices)
        
        # Bollinger Bands
        indicators['Bollinger'] = self._calculate_bollinger_bands(prices)
        
        # Volume indicators
        if volumes is not None and len(volumes) > 0:
            indicators['Volume_SMA'] = self._calculate_sma(volumes, 20)
        
        return indicators

    def _analyze_trend(self, prices: np.array, window: int = 30) -> Dict:
        """Analyze price trend and momentum"""
        if len(prices) < window:
            return {'trend': 'insufficient_data'}
            
        # Calculate returns
        returns = np.diff(prices) / prices[:-1]
        
        # Linear regression for trend
        x = np.arange(len(prices))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, prices)
        
        # Determine trend strength and direction
        trend_strength = abs(r_value)
        trend_direction = 'bullish' if slope > 0 else 'bearish'
        
        # Calculate momentum
        momentum = (prices[-1] / prices[-window] - 1) if len(prices) >= window else None
        
        return {
            'direction': trend_direction,
            'strength': round(trend_strength, 2),
            'slope': round(slope, 4),
            'r_squared': round(r_value ** 2, 2),
            'momentum': round(momentum, 4) if momentum is not None else None
        }

    def _analyze_volatility(self, prices: np.array) -> Dict:
        """Analyze price volatility"""
        if len(prices) < 2:
            return {'volatility': 'insufficient_data'}
            
        # Calculate returns
        returns = np.diff(prices) / prices[:-1]
        
        # Historical volatility
        hist_vol = np.std(returns) * np.sqrt(252)  # Annualized
        
        # Calculate volatility percentiles
        rolling_vol = pd.Series(returns).rolling(window=self.volatility_window).std()
        vol_percentile = stats.percentileofscore(
            rolling_vol.dropna(), 
            rolling_vol.iloc[-1]
        )
        
        return {
            'current_volatility': round(hist_vol, 4),
            'volatility_percentile': round(vol_percentile, 2),
            'is_high_volatility': vol_percentile > 75
        }

    def _analyze_support_resistance(self, prices: np.array, window: int = 20) -> Dict:
        """Identify potential support and resistance levels"""
        if len(prices) < window:
            return {'levels': 'insufficient_data'}
            
        # Find local maxima and minima
        peaks = []
        troughs = []
        for i in range(1, len(prices) - 1):
            if prices[i] > prices[i-1] and prices[i] > prices[i+1]:
                peaks.append(prices[i])
            if prices[i] < prices[i-1] and prices[i] < prices[i+1]:
                troughs.append(prices[i])
        
        # Calculate potential levels
        resistance = np.mean(peaks) if peaks else prices[-1]
        support = np.mean(troughs) if troughs else prices[-1]
        
        # Current price position
        current_price = prices[-1]
        price_position = (current_price - support) / (resistance - support) if resistance != support else 0.5
        
        return {
            'support': round(support, 2),
            'resistance': round(resistance, 2),
            'price_position': round(price_position, 2),
            'distance_to_support': round((current_price - support) / current_price * 100, 2),
            'distance_to_resistance': round((resistance - current_price) / current_price * 100, 2)
        }

    def _generate_insights(self, 
                          trend_analysis: Dict, 
                          volatility_analysis: Dict,
                          support_resistance: Dict,
                          indicators: Dict) -> List[str]:
        """Generate trading insights based on analysis"""
        insights = []
        
        # Trend-based insights
        if trend_analysis['direction'] == 'bullish' and trend_analysis['strength'] > 0.7:
            insights.append(f"Strong bullish trend detected (R² = {trend_analysis['r_squared']})")
        elif trend_analysis['direction'] == 'bearish' and trend_analysis['strength'] > 0.7:
            insights.append(f"Strong bearish trend detected (R² = {trend_analysis['r_squared']})")
        
        # Volatility insights
        if volatility_analysis['is_high_volatility']:
            insights.append(f"High volatility period (percentile: {volatility_analysis['volatility_percentile']}%)")
        
        # Support/Resistance insights
        if support_resistance['price_position'] > 0.8:
            insights.append("Price near resistance level - potential reversal zone")
        elif support_resistance['price_position'] < 0.2:
            insights.append("Price near support level - potential bounce zone")
        
        # Technical indicator insights
        if indicators['RSI'] and indicators['RSI'] > 70:
            insights.append("Overbought conditions (RSI)")
        elif indicators['RSI'] and indicators['RSI'] < 30:
            insights.append("Oversold conditions (RSI)")
        
        if indicators['MACD']:
            macd = indicators['MACD']
            if macd['histogram'] > 0 and macd['macd'] > macd['signal']:
                insights.append("Positive MACD crossover - bullish signal")
            elif macd['histogram'] < 0 and macd['macd'] < macd['signal']:
                insights.append("Negative MACD crossover - bearish signal")
        
        return insights

    def execute(self, market_data: Dict) -> Dict:
        """
        Perform comprehensive market data analysis
        Returns detailed analysis report
        """
        # Convert data to numpy arrays
        prices = np.array(market_data.get('prices', []))
        volumes = np.array(market_data.get('volumes', []))
        timestamps = market_data.get('timestamps', [])
        
        if len(prices) < 2:
            return {'error': 'Insufficient data for analysis'}
        
        # Perform various analyses
        technical_indicators = self._calculate_technical_indicators(prices, volumes)
        trend_analysis = self._analyze_trend(prices)
        volatility_analysis = self._analyze_volatility(prices)
        support_resistance = self._analyze_support_resistance(prices)
        
        # Generate insights
        insights = self._generate_insights(
            trend_analysis,
            volatility_analysis,
            support_resistance,
            technical_indicators
        )
        
        # Calculate risk metrics
        risk_metrics = {
            'sharpe_ratio': self._calculate_sharpe_ratio(prices),
            'max_drawdown': self._calculate_max_drawdown(prices),
            'risk_reward_ratio': self._calculate_risk_reward_ratio(
                prices, 
                support_resistance['support'], 
                support_resistance['resistance']
            )
        }
        
        analysis_report = {
            'timestamp': datetime.now().isoformat(),
            'technical_indicators': technical_indicators,
            'trend_analysis': trend_analysis,
            'volatility_analysis': volatility_analysis,
            'support_resistance': support_resistance,
            'risk_metrics': risk_metrics,
            'insights': insights,
            'market_context': self._get_market_context(prices, volumes)
        }
        
        return analysis_report

    def _calculate_sharpe_ratio(self, prices: np.array, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe Ratio"""
        returns = np.diff(prices) / prices[:-1]
        excess_returns = returns - risk_free_rate/252  # Daily risk-free rate
        if len(excess_returns) < 2:
            return 0
        return round(np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252), 2)

    def _calculate_max_drawdown(self, prices: np.array) -> float:
        """Calculate Maximum Drawdown"""
        peak = prices[0]
        max_dd = 0
        for price in prices[1:]:
            if price > peak:
                peak = price
            dd = (peak - price) / peak
            if dd > max_dd:
                max_dd = dd
        return round(max_dd * 100, 2)

    def _calculate_risk_reward_ratio(self, 
                                   prices: np.array, 
                                   support: float, 
                                   resistance: float) -> float:
        """Calculate Risk/Reward Ratio"""
        current_price = prices[-1]
        reward = abs(resistance - current_price)
        risk = abs(current_price - support)
        return round(reward / risk, 2) if risk != 0 else 0

    def _get_market_context(self, prices: np.array, volumes: np.array) -> Dict:
        """Provide broader market context"""
        return {
            'price_range': {
                'min': round(np.min(prices), 2),
                'max': round(np.max(prices), 2),
                'current': round(prices[-1], 2)
            },
            'volume_profile': {
                'average': round(np.mean(volumes), 2),
                'current': round(volumes[-1], 2),
                'trend': 'increasing' if volumes[-1] > np.mean(volumes) else 'decreasing'
            },
            'price_distribution': {
                'mean': round(np.mean(prices), 2),
                'std': round(np.std(prices), 2),
                'skew': round(stats.skew(prices), 2)
            }
        }
