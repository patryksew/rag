"""Ollama utility functions for model management and downloading."""
import ollama
from tqdm import tqdm


def pull_model_with_indicator(
    model: str,
    client: ollama.Client | None = None,
) -> None:
    """
    Pull an Ollama model with a download progress indicator.

    Args:
        model (str): Name of the model to pull.
        client (ollama.Client | None): Optional Ollama client to use.
            Defaults to the default Ollama client.
    """
    target_client = client if client is not None else ollama.Client()
    progress = target_client.pull(model, stream=True)
    with tqdm(
        unit="B",
        unit_scale=True,
        bar_format=f"Pulling {model} | {{rate_fmt}} | {{elapsed}}",
        leave=False,
    ) as pbar:
        completed_by_digest: dict[str, int] = {}
        for update in progress:
            digest = update.digest
            if digest and update.completed is not None:
                completed_by_digest[digest] = update.completed
                pbar.n = sum(completed_by_digest.values())

            pbar.refresh()


def ensure_model_is_pulled(
    model: str,
    client: ollama.Client | None = None,
) -> None:
    """
    Ensure that the specified model is pulled, downloading it if not present.

    Args:
        model (str): Name of the model to check and pull.
        client (ollama.Client | None): Optional Ollama client to use.
            Defaults to the default Ollama client.
    """
    target_client = client if client is not None else ollama.Client()
    models_list = target_client.list()
    existing_models = [
        m.model for m in models_list.models if m.model is not None
    ]
    existing_models_lower = [
        m.model.lower() for m in models_list.models if m.model is not None
    ]
    model_lower = model.lower()

    is_available = (
        model in existing_models
        or f"{model}:latest" in existing_models
        or model_lower in existing_models_lower
        or f"{model_lower}:latest" in existing_models_lower
    )

    if not is_available:
        pull_model_with_indicator(model, client=target_client)
