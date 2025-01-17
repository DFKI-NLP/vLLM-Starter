import os
from huggingface_hub import snapshot_download

def load_model_deprecated(model_name: str, model_path: str) -> str:
    """
    Loads a model from a given path. If the model does not exist, downloads it.

    Args:
        model_path (str): The directory to check for or download the model.
        model_name (str): The model identifier to download.

    Returns:
        str: The full path to the model directory.
    """

    # Construct the full model directory path
    model_dir = os.path.join(model_path, model_name)

    if not os.path.exists(model_dir):
        print(f"Downloading model '{model_name}' to '{model_dir}'...")
        try:
            snapshot_download(repo_id=model_name, local_dir=model_dir)
        except Exception as e:
            print(f"Error downloading model: {e}")
            raise
    else:
        print(f"Model already exists at '{model_dir}'.")

    return model_dir