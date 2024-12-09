import requests
import json
import logging

logging.basicConfig(level=logging.DEBUG)

def test_ollama_call():
    messages = [
        {
            "role": "system",
            "content": "You are a professional financial analyst. Generate a report based on the provided data."
        },
        {
            "role": "user",
            "content": """Market Data Summary:
BTC:
Current Price: $44,000.00
Price Change (24h): 2.5%
Period High: $45,000.00
Period Low: $43,000.00

News Analysis:
Headline: Bitcoin shows strong momentum
Sentiment: Positive"""
        }
    ]
    
    try:
        # Enable debug logging for requests
        requests_log = logging.getLogger("urllib3")
        requests_log.setLevel(logging.DEBUG)
        
        url = "http://localhost:11434/api/chat"
        payload = {
            "model": "tinyllama",
            "messages": messages,
            "options": {
                "num_predict": 4000,
                "temperature": 0.0,
                "top_k": 10,
                "top_p": 0.9
            },
            "stream": False
        }
        
        print(f"\nSending request to: {url}")
        print(f"Request payload: {json.dumps(payload, indent=2)}")
        
        response = requests.post(url, json=payload)
        print(f"\nStatus Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        
        if response.status_code != 200:
            print(f"Error Response Text: {response.text}")
            try:
                error_json = response.json()
                print(f"Error Response JSON: {json.dumps(error_json, indent=2)}")
            except:
                print("Could not parse error response as JSON")
        else:
            print(f"Response: {json.dumps(response.json(), indent=2)}")
            
    except requests.exceptions.RequestException as e:
        print(f"Request Exception: {str(e)}")
        if hasattr(e, 'response'):
            print(f"Error Response Status: {e.response.status_code}")
            print(f"Error Response Headers: {dict(e.response.headers)}")
            print(f"Error Response Text: {e.response.text}")
    except Exception as e:
        print(f"Other Exception: {str(e)}")

if __name__ == "__main__":
    test_ollama_call()
