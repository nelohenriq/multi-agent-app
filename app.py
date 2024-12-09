# app.py

import streamlit as st
from agents import AgentManager
from utils.logger import logger
import os
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()


def main():
    st.set_page_config(page_title="Multi-Agent AI System", layout="wide")
    st.title("Multi-Agent AI System with Collaboration and Validation")

    st.sidebar.title("Select Task")
    task = st.sidebar.selectbox(
        "Choose a task:",
        [
            "Summarize Medical Text",
            "Write and Refine Research Article",
            "Sanitize Medical Data (PHI)",
            "Financial Digital Assets Analysis",
        ],
    )

    agent_manager = AgentManager(max_retries=2, verbose=True)

    if task == "Summarize Medical Text":
        summarize_section(agent_manager)
    elif task == "Write and Refine Research Article":
        write_and_refine_article_section(agent_manager)
    elif task == "Sanitize Medical Data (PHI)":
        sanitize_data_section(agent_manager)
    elif task == "Financial Digital Assets Analysis":
        financial_analysis_section(agent_manager)


def summarize_section(agent_manager):
    st.header("Summarize Medical Text")
    text = st.text_area("Enter medical text to summarize:", height=200)
    if st.button("Summarize"):
        if text:
            main_agent = agent_manager.get_agent("SummarizeTool")
            validator_agent = agent_manager.get_agent("SummarizeValidatorAgent")
            with st.spinner("Summarizing..."):
                try:
                    summary = main_agent.execute(text)
                    st.subheader("Summary:")
                    st.write(summary)
                except Exception as e:
                    st.error(f"Error: {e}")
                    logger.error(f"SummarizeAgent Error: {e}")
                    return

            with st.spinner("Validating summary..."):
                try:
                    validation = validator_agent.execute(
                        original_text=text, summary=summary
                    )
                    st.subheader("Validation:")
                    st.write(validation)
                except Exception as e:
                    st.error(f"Validation Error: {e}")
                    logger.error(f"SummarizeValidatorAgent Error: {e}")
        else:
            st.warning("Please enter some text to summarize.")


def write_and_refine_article_section(agent_manager):
    st.header("Write and Refine Research Article")
    topic = st.text_input("Enter the topic for the research article:")
    outline = st.text_area("Enter an outline (optional):", height=150)
    if st.button("Write and Refine Article"):
        if topic:
            writer_agent = agent_manager.get_agent("write_article")
            refiner_agent = agent_manager.get_agent("refiner")
            validator_agent = agent_manager.get_agent("validator")
            with st.spinner("Writing article..."):
                try:
                    draft = writer_agent.execute(topic, outline)
                    st.subheader("Draft Article:")
                    st.write(draft)
                except Exception as e:
                    st.error(f"Error: {e}")
                    logger.error(f"WriteArticleAgent Error: {e}")
                    return

            with st.spinner("Refining article..."):
                try:
                    refined_article = refiner_agent.execute(draft)
                    st.subheader("Refined Article:")
                    st.write(refined_article)
                except Exception as e:
                    st.error(f"Refinement Error: {e}")
                    logger.error(f"RefinerAgent Error: {e}")
                    return

            with st.spinner("Validating article..."):
                try:
                    validation = validator_agent.execute(
                        topic=topic, article=refined_article
                    )
                    st.subheader("Validation:")
                    st.write(validation)
                except Exception as e:
                    st.error(f"Validation Error: {e}")
                    logger.error(f"ValidatorAgent Error: {e}")
        else:
            st.warning("Please enter a topic for the research article.")


def sanitize_data_section(agent_manager):
    st.header("Sanitize Medical Data (PHI)")
    medical_data = st.text_area("Enter medical data to sanitize:", height=200)
    if st.button("Sanitize Data"):
        if medical_data:
            main_agent = agent_manager.get_agent("SanitizeDataTool")
            validator_agent = agent_manager.get_agent("SanitizeDataValidatorAgent")
            with st.spinner("Sanitizing data..."):
                try:
                    sanitized_data = main_agent.execute(medical_data)
                    st.subheader("Sanitized Data:")
                    st.write(sanitized_data)
                except Exception as e:
                    st.error(f"Error: {e}")
                    logger.error(f"SanitizeDataAgent Error: {e}")
                    return

            with st.spinner("Validating sanitized data..."):
                try:
                    validation = validator_agent.execute(
                        original_data=medical_data, sanitized_data=sanitized_data
                    )
                    st.subheader("Validation:")
                    st.write(validation)
                except Exception as e:
                    st.error(f"Validation Error: {e}")
                    logger.error(f"SanitizeDataValidatorAgent Error: {e}")
        else:
            st.warning("Please enter medical data to sanitize.")


def financial_analysis_section(agent_manager):
    st.header("Financial Digital Assets Analysis")

    # Input for digital assets
    default_assets = ["BTC-USD", "ETH-USD", "SOL-USD"]
    assets_input = st.text_area(
        "Enter digital assets to analyze (one per line):",
        value="\n".join(default_assets),
        height=100
    )
    assets = [asset.strip() for asset in assets_input.split("\n") if asset.strip()]
    
    # Analysis period
    period = st.selectbox(
        "Analysis period",
        [
            "1d", "5d",           # Days
            "1mo", "3mo", "6mo",  # Months
            "1y", "2y", "5y", "10y",  # Years
            "ytd", "max"          # Special periods
        ],
        index=5,  # Default to "1y"
        format_func=lambda x: {
            "1d": "1 Day",
            "5d": "5 Days",
            "1mo": "1 Month",
            "3mo": "3 Months",
            "6mo": "6 Months",
            "1y": "1 Year",
            "2y": "2 Years",
            "5y": "5 Years",
            "10y": "10 Years",
            "ytd": "Year to Date",
            "max": "Maximum Available"
        }.get(x, x),
        help="Market data will use the full period. News data will be capped at 30 days due to API limits."
    )
    
    if st.button("Generate Report"):
        if assets:
            with st.spinner("Analyzing market data and news..."):
                try:
                    # 1. Fetch market data
                    market_agent = agent_manager.get_agent("MarketDataTool")
                    market_data = market_agent.execute(assets, period)
                    
                    # 2. Fetch news
                    news_agent = agent_manager.get_agent("NewsFetcherTool")
                    news_data = news_agent.execute(assets, period)
                    
                    # 3. Analyze sentiment
                    sentiment_agent = agent_manager.get_agent("SentimentAnalyzerTool")
                    analyzed_news = sentiment_agent.execute(news_data)
                    
                    # 4. Generate report
                    report_agent = agent_manager.get_agent("ReportGeneratorTool")
                    report = report_agent.execute(market_data, analyzed_news)
                    
                    # Display market data
                    st.header("Market Overview")
                    for asset, data in market_data.items():
                        col1, col2 = st.columns([1, 2])
                        with col1:
                            st.metric(
                                label=f"{asset} Price",
                                value=f"${data['current_price']:.2f}",
                                delta=f"{data['price_change']:.2f}%"
                            )
                        with col2:
                            st.markdown(f"""
                            - **Period:** {data['period']}
                            - **High:** ${data['high']:.2f}
                            - **Low:** ${data['low']:.2f}
                            """)
                    
                    # Display news with sentiment
                    st.header("Latest News & Sentiment")
                    for item in analyzed_news:
                        sentiment = item['sentiment_analysis']
                        score = sentiment['score']
                        
                        # Color code based on sentiment
                        if score > 0:
                            color = "green"
                            emoji = "📈"
                        elif score < 0:
                            color = "red"
                            emoji = "📉"
                        else:
                            color = "gray"
                            emoji = "➖"
                        
                        # Create an expander for each news item
                        with st.expander(f"{emoji} {item['title']} ({item['asset']})"):
                            st.markdown(f"""
                            **Source:** {item['source']['name']}  
                            **Published:** {item['publishedAt']}  
                            **Sentiment:** <span style='color: {color}'>{sentiment['explanation']} (Score: {score:.2f})</span>
                            
                            {item['description']}
                            
                            [Read more]({item['url']})
                            """, unsafe_allow_html=True)
                    
                    # Display AI analysis
                    st.header("AI Market Analysis")
                    st.markdown(report)
                    
                except Exception as e:
                    st.error(f"FinancialAnalysis Error: {str(e)}")
        else:
            st.warning("Please enter at least one digital asset to analyze.")


if __name__ == "__main__":
    main()
