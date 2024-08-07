from pathlib import Path
import torch
from litgpt.generate.base import main
from litgpt.utils import get_default_supported_precision

from nanograd.models.GPT import GPT
from nanograd.models.GPT import BioGPT_config

from nanograd.models.llama.inference_llama import LLaMA

def use_GPT():

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

allow_cuda = False
device = 'cuda' if torch.cuda.is_available() and allow_cuda else 'cpu'

def use_llama():
    torch.manual_seed(0)

    user_prompt = input("Prompt: ")    

    model = LLaMA.build(
        checkpoints_dir='nanograd\models\llama\llama-2-7b',
        tokenizer_path='nanograd\models\llama\\tokenizer.model',
        load_model=True,
        max_seq_len=1024,
        max_batch_size=len(user_prompt),
        device=device
    )

    out_tokens, out_texts = (model.text_completion(user_prompt, max_gen_len=64))
    assert len(out_texts) == len(user_prompt)
    for i in range(len(out_texts)):
        print(f'{out_texts[i]}')
        print('-' * 50)

if __name__ == "__main__":
    use_GPT()