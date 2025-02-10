import requests
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from langchain_core.embeddings import Embeddings
from tenacity import retry, stop_after_attempt, wait_fixed

class InfinityEmbeddingsComponent(BaseModel, Embeddings):
    api_url: str         # e.g., "http://localhost:7997"
    model: Optional[str] = None   # If None, auto-select model with "embed" capability
    auth_token: Optional[str] = None
    timeout: int = 30

    def _check_health(self, timeout: Optional[int] = None) -> None:
        endpoint = f"{self.api_url.rstrip('/')}/health"
        timeout = timeout if timeout is not None else self.timeout
        
        @retry(stop=stop_after_attempt(timeout), wait=wait_fixed(3))
        def check_healthy():
            res = requests.get(endpoint, timeout=5)
            if res.status_code != 200:
                raise Exception(f"Health check failed, status code: {res.status_code}")
        check_healthy()

    def get_available_models(self) -> List[Dict[str, Any]]:
        endpoint = f"{self.api_url.rstrip('/')}/models"
        headers = {"Content-Type": "application/json"}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        response = requests.get(endpoint, headers=headers)
        response.raise_for_status()
        result = response.json()
        return result.get("data", [])

    def auto_select_model(self, capability: str = "embed") -> str:
        models = self.get_available_models()
        for model_info in models:
            if capability in model_info.get("capabilities", []):
                return model_info.get("id")
        raise ValueError(f"Could not find a model with '{capability}' capability")

    def _embed(self, text: str) -> List[float]:
        self._check_health()
        if not self.model:
            self.model = self.auto_select_model("embed")
        endpoint = f"{self.api_url.rstrip('/')}/embeddings"
        headers = {"Content-Type": "application/json"}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        payload = {"input": text, "model": self.model}
        try:
            response = requests.post(endpoint, json=payload, headers=headers)
            response.raise_for_status()
            result = response.json()
            data = result.get("data")
            if not data or not isinstance(data, list):
                raise ValueError("Invalid response, missing 'data' field")
            embedding = data[0].get("embedding")
            if embedding is None:
                raise ValueError("Response does not contain 'embedding' field")
            return embedding
        except Exception as e:
            raise Exception(f"Error calling Infinity Embeddings API: {e}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        cleaned_texts = [text.replace("\n", " ") for text in texts]
        return [self._embed(text) for text in cleaned_texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)

# LangFlow UI integration
from langflow.io import StrInput, Output
from langflow.base.embeddings.model import LCEmbeddingsModel
from langflow.field_typing import Embeddings

class InfinityEmbeddingsComponentUI(LCEmbeddingsModel):
    display_name = "Local Infinity Embeddings"
    description = (
        "Generate embeddings and perform reranking by calling the locally deployed Infinity API. "
        "If no model is specified, a suitable model will be automatically selected from the /models endpoint "
        "based on the 'embed' capability. You may enter a model name, and the system will verify if that model exists."
    )
    documentation = "https://github.com/michaelfeil/infinity"
    icon = "Infinity"
    inputs = [
        StrInput(
            name="api_url",
            display_name="API URL",
            value="http://rerank-embeding-jinaai:7997",
            required=True,
            advanced=False
        ),
        StrInput(
            name="model_name",
            display_name="Model Name",
            value="",
            info="Enter the model name. The system will verify if the model exists.",
        ),
        StrInput(
            name="auth_token",
            display_name="Auth Token",
            value="",
            advanced=True
        )
    ]
    outputs = [
        Output(
            display_name="Embeddings",
            name="embeddings",
            method="build_embeddings"
        ),
    ]

    def build_embeddings(self) -> Embeddings:
        model_name_input = self.model_name.strip() if hasattr(self, "model_name") else ""
        if model_name_input:
            try:
                endpoint = f"{self.api_url.rstrip('/')}/models"
                headers = {"Content-Type": "application/json"}
                if self.auth_token:
                    headers["Authorization"] = f"Bearer {self.auth_token}"
                response = requests.get(endpoint, headers=headers)
                response.raise_for_status()
                result = response.json()
                available_models = result.get("data", [])
                model_ids = [model_info["id"].strip().lower() for model_info in available_models]
                if model_name_input.lower() not in model_ids:
                    raise ValueError(f"Model '{model_name_input}' does not exist.")
            except Exception as e:
                raise ValueError(f"Unable to validate model: {e}")
        return InfinityEmbeddingsComponent(
            api_url=self.api_url,
            model=model_name_input if model_name_input != "" else None,
            auth_token=self.auth_token,
            timeout=15,
        )