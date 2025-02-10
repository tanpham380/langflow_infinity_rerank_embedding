from typing import Dict, List, Any, Optional
import requests
from langflow.custom import Component
from langflow.io import MessageTextInput, DataInput, StrInput, IntInput, Output
from langflow.schema import Data
from tenacity import retry, stop_after_attempt, wait_fixed

class InfinityReranksComponent(Component):
    display_name = "Local Infinity Rerank"
    description = (
        "Ranks documents based on a query using a cross-encoder model. "
        "This component calls an external API (like a locally deployed Cohere rerank service) "
        "to rank documents and returns the top-ranked documents. "
        "If a model name is provided and is not valid, an error is returned."
    )
    icon = "Infinity"
    documentation = "https://github.com/michaelfeil/infinity"

    inputs = [
        MessageTextInput(
            name="query",
            display_name="Query",
            info="The query string to rank documents against.",
            value="What is FPT?",
        ),
        DataInput(
            name="documents",
            display_name="Documents",
            info="A list of documents to be ranked.",
        ),
        StrInput(
            name="api_url",
            display_name="API URL",
            info="The base URL of the API endpoint (e.g., http://rerank-embeding-jinaai:7997/).",
            value="http://rerank-embeding-jinaai:7997/",
        ),
        StrInput(
            name="model_name",
            display_name="Model Name",
            info="Optional: the model name to use for reranking. If provided, it must be valid.",
            value="",
        ),
        IntInput(
            name="number_of_rerank_return",
            display_name="Number of Documents to Return",
            info="The number of top documents to return after reranking.",
            value=3,
        ),
    ]
    outputs = [
        Output(
            display_name="Top Ranked Documents",
            name="top_ranked_document",
            method="rank_documents",
        )
    ]
    
    # Use a requests session for connection reuse.
    session = requests.Session()
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Cache models lookup to avoid redundant API calls.
        self._cached_models = None
        self._cached_api_url = None

    def _check_health(self, timeout: Optional[int] = None) -> None:
        endpoint = f"{self.api_url.rstrip('/')}/health"
        timeout_val = timeout if timeout is not None else (self.timeout if hasattr(self, "timeout") else 30)
        
        @retry(stop=stop_after_attempt(timeout_val), wait=wait_fixed(3))
        def check_healthy():
            res = requests.get(endpoint, timeout=5)
            if res.status_code != 200:
                raise Exception(f"Health check failed, status code: {res.status_code}")
        check_healthy()
    def get_valid_model(self, api_url: str, model_name: str) -> str:
        """
        Retrieves the list of models from the API and validates the provided model_name.
        If model_name is non-empty but not found, a ValueError is raised.
        If model_name is empty, returns the first model with a 'rerank' capability.
        Implements caching to avoid redundant API calls.
        """
        models_url = api_url.rstrip("/") + "/models"
        if self._cached_models is not None and self._cached_api_url == api_url:
            models_data = self._cached_models
        else:
            try:
                response = self.session.get(models_url, timeout=5)
                response.raise_for_status()
                models_data = response.json().get("data", [])
                self._cached_models = models_data
                self._cached_api_url = api_url
            except Exception as e:
                self.status = f"Error retrieving models: {e}"
                raise ValueError(f"Error retrieving models: {e}")
        
        if model_name:
            model_name_norm = model_name.strip().lower()
            for model in models_data:
                if model.get("id", "").strip().lower() == model_name_norm:
                    return model["id"]
            raise ValueError(f"Model '{model_name}' not found in available models.")
        else:
            for model in models_data:
                if "rerank" in model.get("capabilities", []):
                    return model["id"]
            raise ValueError("No rerank model found in API models.")

    def ranking_function(self, documents: List[str], query: str, api_url: str, model: str, top_n: int) -> List[Dict[str, Any]]:
        """
        Calls the external rerank API with the query, documents, model, and top_n.
        The API is expected to return an object with a "results" field,
        where each result contains an "index" and "relevance_score".
        This function returns a list of dictionaries, each with "document" and "score".
        """
        # Check health of the rerank API before proceeding.
        self._check_health()
        
        rerank_url = api_url.rstrip("/") + "/rerank"
        req_body = {
            "query": query,
            "documents": documents,
            "model": model,
            "top_n": top_n,
        }
        try:
            response = self.session.post(rerank_url, json=req_body, timeout=10)
            response.raise_for_status()
            res_body = response.json()
            if "results" in res_body:
                results = res_body["results"]
                output = []
                for item in results:
                    idx = item.get("index")
                    score = item.get("relevance_score")
                    if idx is not None and idx < len(documents):
                        output.append({"document": documents[idx], "score": score})
                return output
            else:
                return []
        except Exception as e:
            self.status = f"Error calling rerank API: {e}"
            return []
    
    def rank_documents(self) -> List[Data]:
        """
        Extracts document texts, validates the API URL and model name, calls the rerank API,
        sorts the returned results in descending order by score, and returns the top-ranked documents
        as a list of Data objects. If the provided model name is invalid, a ValueError is raised.
        """
        if not self.documents:
            raise ValueError("No documents provided for ranking.")
        if not self.query:
            raise ValueError("Query cannot be empty.")
        if not self.api_url:
            raise ValueError("API URL is required.")
    
        if isinstance(self.documents, list):
            documents_list = [doc.text for doc in self.documents if hasattr(doc, "text")]
        else:
            documents_list = [self.documents.text if hasattr(self.documents, "text") else str(self.documents)]
    
        models_url = self.api_url.rstrip("/") + "/models"
        try:
            test_response = self.session.get(models_url, timeout=5)
            test_response.raise_for_status()
        except Exception as e:
            raise ValueError(f"API URL is not valid or reachable: {e}")
    
        valid_model = self.get_valid_model(self.api_url, self.model_name)
    
        ranked_results = self.ranking_function(documents_list, self.query, self.api_url, valid_model, self.number_of_rerank_return)
    
        if ranked_results:
            top_documents = sorted(ranked_results, key=lambda x: x["score"], reverse=True)[:self.number_of_rerank_return]
            data_objects = [Data(data=item) for item in top_documents]
            return data_objects
    
        return []  # Return an empty list if no results are obtained.