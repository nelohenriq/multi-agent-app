from .agent_base import AgentBase
from typing import Dict, List

class ReportGeneratorTool(AgentBase):
    def __init__(self, max_retries=2, verbose=True):
        super().__init__(name="ReportGeneratorTool", max_retries=max_retries, verbose=verbose)

    def execute(self, market_data: Dict, analyzed_news: List[Dict]) -> str:
        """Generate a comprehensive financial report using Ollama"""
        
        # Prepare the context for the report
        context = self._prepare_context(market_data, analyzed_news)
        
        messages = [
            {
                "role": "system",
                "content": "You are a professional financial analyst. Generate a comprehensive report based on the provided market data and news analysis. Focus on key trends, sentiment analysis, and potential market implications.",
            },
            {
                "role": "user",
                "content": context,
            },
        ]
        
        report = self.call_ollama(messages, max_tokens=4000)
        return report

    def _prepare_context(self, market_data: Dict, analyzed_news: List[Dict]) -> str:
        """Prepare context for the report generation"""
        context = "Market Data Summary:\n"
        
        # Add market data
        for asset, data in market_data.items():
            if "error" not in data:
                context += f"\n{asset.upper()}:\n"
                context += f"Current Price: ${data['current_price']:.2f}\n"
                context += f"Price Change ({data['period']}): {data['price_change']:.2f}%\n"
                context += f"Period High: ${data['high']:.2f}\n"
                context += f"Period Low: ${data['low']:.2f}\n"
                context += f"Analysis Period: {data['start_date']} to {data['end_date']}\n"
                context += f"24h Volume: ${data['volume_24h']:,.2f}\n"
        
        # Add news analysis
        context += "\nNews Analysis:\n"
        for news in analyzed_news[:5]:  # Include top 5 most relevant news items
            context += f"\nHeadline: {news.get('title', '')}\n"
            context += f"Sentiment: {news.get('sentiment_analysis', '')}\n"
            if news.get('asset'):
                context += f"Related Asset: {news['asset']}\n"
        
        return context