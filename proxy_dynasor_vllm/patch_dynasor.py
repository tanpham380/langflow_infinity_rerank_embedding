import dynasor.cli.openai_server as os
import time

def patched_ensure_model_initialized(self, model: str):
    try:
        model_dict = self.client.models.retrieve(model)
    except Exception:
        model_dict = {
            "id": model,
            "created": int(time.time()),
            "object": "model",
            "owned_by": "vllm",
            "permission": [],
            "root": model,
            "parent": None
        }
    return model, model_dict

# Ghi đè phương thức ensure_model_initialized
os.DynasorOpenAIClient.ensure_model_initialized = patched_ensure_model_initialized
