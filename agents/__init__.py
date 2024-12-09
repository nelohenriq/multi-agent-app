from .agent_base import AgentBase
from .summarize_tool import SummarizeTool
from .write_article_tool import WriteArticleTool
from .validator_agent import ValidatorAgent
from .refiner_agent import RefinerAgent
from .sanitize_data_tool import SanitizeDataTool
from .sanitiza_data_validator_agent import SanitizeDataValidatorAgent
from .write_article_validator_agent import WriteArticleValidatorAgent
from .summarize_validator_agent import SummarizeValidatorAgent
from .news_fetcher import NewsFetcherTool
from .sentiment_analyzer import SentimentAnalyzerTool
from .market_data import MarketDataTool
from .report_generator import ReportGeneratorTool
from .market_data_analyzer import MarketDataAnalyzer


class AgentManager:
    def __init__(self, max_retries=2, verbose=True):
        self.agents = {
            "SummarizeTool": SummarizeTool(max_retries=max_retries, verbose=verbose),
            "WriteArticleTool": WriteArticleTool(
                max_retries=max_retries, verbose=verbose
            ),
            "ValidatorAgent": ValidatorAgent(max_retries=max_retries, verbose=verbose),
            "RefinerAgent": RefinerAgent(max_retries=max_retries, verbose=verbose),
            "SanitizeDataTool": SanitizeDataTool(
                max_retries=max_retries, verbose=verbose
            ),
            "SanitizeDataValidatorAgent": SanitizeDataValidatorAgent(
                max_retries=max_retries, verbose=verbose
            ),
            "WriteArticleValidatorAgent": WriteArticleValidatorAgent(
                max_retries=max_retries, verbose=verbose
            ),
            "SummarizeValidatorAgent": SummarizeValidatorAgent(
                max_retries=max_retries, verbose=verbose
            ),
            # Financial Analysis Agents
            "NewsFetcherTool": NewsFetcherTool(
                max_retries=max_retries, verbose=verbose
            ),
            "SentimentAnalyzerTool": SentimentAnalyzerTool(
                max_retries=max_retries, verbose=verbose
            ),
            "MarketDataTool": MarketDataTool(
                max_retries=max_retries, verbose=verbose
            ),
            "MarketDataAnalyzer": MarketDataAnalyzer(
                max_retries=max_retries, verbose=verbose
            ),
            "ReportGeneratorTool": ReportGeneratorTool(
                max_retries=max_retries, verbose=verbose
            ),
        }

    def get_agent(self, agent_name):
        agent = self.agents.get(agent_name)
        if not agent:
            raise ValueError(f"Agent {agent_name} not found.")
        return agent
