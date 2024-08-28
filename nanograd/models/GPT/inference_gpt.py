from pathlib import Path
import torch
from litgpt.generate.base import main
from litgpt.utils import get_default_supported_precision

from nanograd.models.GPT import GPT
from nanograd.models.GPT import BioGPT_config

def use_model():

    checkpoint_dir = Path("nanograd\models\GPT\checkpoints") / "EleutherAI" / "pythia-1b"

    torch.manual_seed(123)

    main(
        prompt= input("your prompt: "),
        max_new_tokens=50,
        temperature=0.5,
        top_k=200,
        top_p=1.0,
        checkpoint_dir=checkpoint_dir,
        precision=get_default_supported_precision(training=False),
        compile=False
    )


if __name__ == "__main__":
    use_model()