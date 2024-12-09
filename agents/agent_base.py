import openai
from abc import ABC, abstractmethod
import os
from loguru import logger
from dotenv import load_dotenv
import requests

load_dotenv()

openai.base_url = os.getenv("GROQ_API_BASE")
openai.api_key = os.getenv("GROQ_API_KEY")


class AgentBase(ABC):
    def __init__(self, name, max_retries=2, verbose=True):
        self.name = name
        self.max_retries = max_retries
        self.verbose = verbose
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.ollama_model = os.getenv("OLLAMA_MODEL", "tinyllama")
        self.logger = logger

    @abstractmethod
    def execute(self, *args, **kwargs):
        pass

    def call_ollama(self, messages, max_tokens=150, temperature=0.7):
        retries = 0
        while retries < self.max_retries:
            try:
                if self.verbose:
                    self.logger.info(f"\n{'='*50}")
                    self.logger.info(f"[{self.name}] Sending message to Ollama:")
                    self.logger.info(f"Model: {self.ollama_model}")
                    self.logger.info(f"Temperature: {temperature}")
                    self.logger.info(f"Max tokens: {max_tokens}")
                    self.logger.info("\nMessages:")
                    for msg in messages:
                        self.logger.info(f"\n[{msg['role'].upper()}]")
                        self.logger.info(f"{msg['content']}")
                    self.logger.info(f"\n{'='*50}")

                response = requests.post(
                    f"{self.ollama_base_url}/api/chat",
                    json={
                        "model": self.ollama_model,
                        "messages": messages,
                        "options": {
                            "num_predict": max_tokens,
                            "temperature": temperature,
                            "top_k": 10,  # Limit token selection to top 10 most likely
                            "top_p": 0.9  # Sample from 90% most likely tokens
                        },
                        "stream": False  # Disable streaming to get a single response
                    }
                )
                response.raise_for_status()
                
                # Get the last message from the response
                reply = response.json()
                content = reply.get('message', {}).get('content', '')
                
                if self.verbose:
                    self.logger.info(f"\n{'='*50}")
                    self.logger.info(f"[{self.name}] Ollama replied:")
                    self.logger.info(f"\n{content}")
                    self.logger.info(f"\n{'='*50}")
                return content
            except Exception as e:
                retries += 1
                self.logger.error(
                    f"[{self.name}] Error calling Ollama: {e}, Retry {retries}/{self.max_retries}"
                )
                continue
        raise Exception(
            f"[{self.name}] Failed to call Ollama after {self.max_retries} retries"
        )

    def call_openai(self, messages, max_tokens=150, temperature=0.7):
        retries = 0
        while retries < self.max_retries:
            try:
                if self.verbose:
                    self.logger.info(f"[{self.name}] Sending message to OpenAi:")
                    for message in messages:
                        self.logger.debug(f"{message['role']}: {message['content']}")

                response = openai.chat.completions.create(
                    model="llama-3.2-3b-preview",
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                reply = response.choices[0].message
                if self.verbose:
                    self.logger.info(f"[{self.name}] OpenAi replied: {reply.content}")
                return reply.content
            except Exception as e:
                retries += 1
                self.logger.error(
                    f"[{self.name}] Error calling OpenAI: {e}, Retry {retries}/{self.max_retries}"
                )
                retries += 1
                continue
        raise Exception(
            f"[{self.name}] Failed to call OpenAI after {self.max_retries} retries"
        )
