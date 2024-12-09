from .agent_base import AgentBase
import yfinance as yf
from typing import List, Dict
from datetime import datetime, timedelta
from enum import Enum

class TimePeriod(Enum):
    ONE_DAY = "1d"
    FIVE_DAYS = "5d"
    ONE_MONTH = "1mo"
    THREE_MONTHS = "3mo"
    SIX_MONTHS = "6mo"
    ONE_YEAR = "1y"
    TWO_YEARS = "2y"
    FIVE_YEARS = "5y"
    TEN_YEARS = "10y"
    YTD = "ytd"
    MAX = "max"

class MarketDataTool(AgentBase):
    def __init__(self, max_retries=2, verbose=True):
        super().__init__(name="MarketDataTool", max_retries=max_retries, verbose=verbose)

    def execute(self, assets: List[str], period: str = "1y") -> Dict:
        """
        Fetch market data for specified digital assets
        
        Args:
            assets: List of asset symbols (e.g., ["BTC-USD", "ETH-USD"])
            period: Time period for historical data. Valid values:
                   - "1d", "5d" (n days)
                   - "1mo", "3mo", "6mo" (n months)
                   - "1y", "2y", "5y", "10y" (n years)
                   - "ytd" (year to date)
                   - "max" (maximum available data)
        """
        market_data = {}
        
        # Validate period
        try:
            period = TimePeriod(period).value
        except ValueError:
            self.logger.warning(f"Invalid period '{period}', defaulting to '1y'")
            period = "1y"
        
        for asset in assets:
            try:
                # Convert asset name to ticker (e.g., "bitcoin" -> "BTC-USD")
                ticker = f"{asset}-USD" if not asset.endswith("-USD") else asset
                data = yf.Ticker(ticker)
                
                # Get historical data
                hist = data.history(period=period)
                
                if hist.empty:
                    self.logger.error(f"No data available for {asset}")
                    market_data[asset] = {"error": "No data available"}
                    continue
                
                # Calculate basic metrics
                market_data[asset] = {
                    "current_price": hist['Close'].iloc[-1],
                    "price_change": ((hist['Close'].iloc[-1] - hist['Close'].iloc[0]) / hist['Close'].iloc[0]) * 100,
                    "volume_24h": hist['Volume'].iloc[-1],
                    "high": hist['High'].max(),
                    "low": hist['Low'].min(),
                    "period": period,
                    "start_date": hist.index[0].strftime('%Y-%m-%d'),
                    "end_date": hist.index[-1].strftime('%Y-%m-%d'),
                    "historical_data": hist.to_dict('records')
                }
            except Exception as e:
                self.logger.error(f"Error fetching data for {asset}: {e}")
                market_data[asset] = {"error": str(e)}
                
        return market_data
