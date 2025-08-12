import os
import json
from typing import Dict, List, Optional, Any
import openai
import anthropic
import requests


class LLMClient:
    def __init__(self, provider: str = "openai"):
        self.provider = provider
        self._setup_client()
    
    def _setup_client(self):
        if self.provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")
            openai.api_key = api_key
            
        elif self.provider == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY environment variable not set")
            self.client = anthropic.Anthropic(api_key=api_key)
            
        elif self.provider == "ollama":
            self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def generate(self, prompt: str, model: str, **kwargs) -> str:
        if self.provider == "openai":
            return self._generate_openai(prompt, model, **kwargs)
        elif self.provider == "anthropic":
            return self._generate_anthropic(prompt, model, **kwargs)
        elif self.provider == "ollama":
            return self._generate_ollama(prompt, model, **kwargs)
    
    def _generate_openai(self, prompt: str, model: str, **kwargs) -> str:
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                **kwargs
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"OpenAI API error: {e}")
            return ""
    
    def _generate_anthropic(self, prompt: str, model: str, **kwargs) -> str:
        try:
            response = self.client.messages.create(
                model=model,
                max_tokens=kwargs.get("max_tokens", 1000),
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except Exception as e:
            print(f"Anthropic API error: {e}")
            return ""
    
    def _generate_ollama(self, prompt: str, model: str, **kwargs) -> str:
        try:
            url = f"{self.base_url}/api/generate"
            data = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                **kwargs
            }
            response = requests.post(url, json=data)
            response.raise_for_status()
            return response.json()["response"]
        except Exception as e:
            print(f"Ollama API error: {e}")
            return ""


def get_llm_client(provider: str = "openai") -> LLMClient:
    return LLMClient(provider)
