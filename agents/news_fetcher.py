from .agent_base import AgentBase
from datetime import datetime, timedelta
import os
import json
import feedparser
from typing import List, Dict
from pathlib import Path
import re

class NewsFetcherTool(AgentBase):
    def __init__(self, max_retries=2, verbose=True):
        super().__init__(name="NewsFetcherTool", max_retries=max_retries, verbose=verbose)
        # Load RSS feeds from json
        rss_file = Path(__file__).parent.parent / 'rss_feeds.json'
        with open(rss_file, 'r') as f:
            self.rss_feeds = json.load(f)['crypto']

    def execute(self, assets: List[str], period: str = "1y") -> List[Dict]:
        """
        Fetch news for specified digital assets from RSS feeds
        
        Args:
            assets: List of asset symbols (e.g., ["BTC-USD", "ETH-USD"])
            period: Time period (will be used to filter articles by date)
        """
        news_items = []
        
        # Convert period to timedelta for date filtering
        delta = self._period_to_timedelta(period)
        cutoff_date = datetime.now() - delta
        
        # Prepare search terms for each asset
        asset_terms = {}
        for asset in assets:
            base_symbol = asset.replace('-USD', '')
            asset_terms[asset] = {
                'symbol': base_symbol,
                'name': self._get_full_name(base_symbol).lower()
            }
        
        # Fetch and parse RSS feeds
        for feed_url in self.rss_feeds:
            try:
                self.logger.info(f"Fetching RSS feed: {feed_url}")
                feed = feedparser.parse(feed_url)
                
                for entry in feed.entries:
                    # Convert entry date to datetime
                    try:
                        pub_date = datetime(*entry.published_parsed[:6])
                    except (AttributeError, TypeError):
                        # If date parsing fails, skip date filtering
                        pub_date = datetime.now()
                    
                    # Skip if article is too old
                    if pub_date < cutoff_date:
                        continue
                    
                    # Check if article mentions any of our assets
                    title = entry.get('title', '').lower()
                    description = entry.get('description', '').lower()
                    content = title + ' ' + description
                    
                    for asset, terms in asset_terms.items():
                        # Use word boundaries to match whole words only
                        symbol_pattern = r'\b' + terms['symbol'].lower() + r'\b'
                        name_pattern = r'\b' + terms['name'].lower() + r'\b'
                        
                        if re.search(symbol_pattern, content) or re.search(name_pattern, content):
                            # Double check it's not a false positive (e.g., AAVE when looking for SOL)
                            # by checking if any other crypto name appears in the title
                            other_cryptos = {
                                'AAVE': 'Aave', 'LINK': 'Chainlink', 'UNI': 'Uniswap',
                                'MATIC': 'Polygon', 'AVAX': 'Avalanche', 'ATOM': 'Cosmos',
                                'ALGO': 'Algorand', 'XLM': 'Stellar', 'FTM': 'Fantom',
                                'NEAR': 'NEAR Protocol'
                            }
                            
                            is_false_positive = False
                            for other_symbol, other_name in other_cryptos.items():
                                if other_symbol != terms['symbol'] and other_name != terms['name']:
                                    other_pattern = r'\b' + other_symbol.lower() + r'\b'
                                    other_name_pattern = r'\b' + other_name.lower() + r'\b'
                                    if re.search(other_pattern, title.lower()) or re.search(other_name_pattern, title.lower()):
                                        is_false_positive = True
                                        break
                            
                            if not is_false_positive:
                                article = {
                                    'title': entry.get('title'),
                                    'description': entry.get('description'),
                                    'url': entry.get('link'),
                                    'publishedAt': pub_date.isoformat(),
                                    'source': {
                                        'name': feed.feed.get('title', feed_url)
                                    },
                                    'asset': asset,
                                    'matched_term': terms['symbol'] if re.search(symbol_pattern, content) else terms['name']
                                }
                                news_items.append(article)
                                break  # One article can only be assigned to one asset
                            
            except Exception as e:
                self.logger.error(f"Error fetching RSS feed {feed_url}: {e}")
                continue

        # Sort by publishedAt and take most recent/relevant articles
        news_items.sort(key=lambda x: x.get('publishedAt', ''), reverse=True)
        return news_items[:20]  # Return top 20 most recent articles

    def _period_to_timedelta(self, period: str) -> timedelta:
        """Convert yfinance period format to timedelta"""
        period = period.lower()
        now = datetime.now()
        
        # Direct day conversions
        if period.endswith('d'):
            return timedelta(days=int(period[:-1]))
        
        # Month conversions (approximate)
        if period.endswith('mo'):
            months = int(period[:-2])
            return timedelta(days=months * 30)
        
        # Year conversions
        if period.endswith('y'):
            years = int(period[:-1])
            return timedelta(days=years * 365)
        
        # Special periods
        if period == 'ytd':
            return now - datetime(now.year, 1, 1)
        if period == 'max':
            return timedelta(days=30)  # Cap at 30 days
        
        # Default to 7 days if unknown period
        self.logger.warning(f"Unknown period format '{period}', defaulting to 7 days")
        return timedelta(days=7)
        
    def _get_full_name(self, symbol: str) -> str:
        """Get the full name of a cryptocurrency"""
        crypto_names = {
            'BTC': 'Bitcoin',
            'ETH': 'Ethereum',
            'SOL': 'Solana',
            'XRP': 'Ripple',
            'ADA': 'Cardano',
            'DOGE': 'Dogecoin',
            'DOT': 'Polkadot',
            'USDT': 'Tether',
            'USDC': 'USD Coin',
            'BNB': 'Binance Coin'
        }
        return crypto_names.get(symbol, symbol)
