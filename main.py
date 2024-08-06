from nanograd import generate_dataset 

from nanograd.models.llama import inference_llama 
from nanograd.models.GPT import inference_gpt

from nanograd.models import ollama
from nanograd.models import chat
import nanograd

from nanograd.optimizers import AdamW
from nanograd.models.GPT import tokenizer
from nanograd.analysis_lab import sentiment_analysis

from nanograd.models.stable_diffusion.sd import sd_inference

from nanograd.RL import Cartpole, car # import reinforcement learning package


# Cartpole.run()
# car.run()

sd_inference.run()

# sentiment_analysis.run()

# generate_dataset.tokenize()

# inference_gpt.use_model()

# inference_llama.use_model()

# tokenizer.run_tokenizer()


# ollama.run() # test the model. 
# chat.chat_with_models()
# chat.chat_models()
###################################################


# if __name__ == "__main__":
#     from nanograd.nn.engine import Value

#     a = Value(-4.0)
#     b = Value(2.0)
#     c = a + b
#     d = a + b + b**3
#     c += c + 1
#     c += 1 + c + (-a)
#     d += d * 2 + (b + a).relu()
#     d += 3 * d + (b - a).relu()
#     d += 3 * d + (b - a).sigmoid(5)
#     e = c - d
#     f = e**2
#     g = f / 2.0
#     g += 10.0 / f
#     print(f'{g.data:.4f}') 
#     g.backward()
#     print(f'{a.grad:.4f}') 
#     print(f'{b.grad:.4f}')  
#     print(f'{e.grad:.4f}')  


# import nanograd.nn.train
