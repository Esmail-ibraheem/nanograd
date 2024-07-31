from nanograd.models.llama import inference_llama 
from nanograd.models.GPT import inference_gpt 
from nanograd.models import ollama
from nanograd.models import chat
   
from nanograd.optimizers import AdamW  
from nanograd.models.GPT import tokenizer 
from nanograd.analysis_lab import sentiment_analysis


# sentiment_analysis.run()


inference_gpt.use_model()

# inference_llama.use_model()

# tokenizer.tokenize()


# ollama.run() # test the model. 
# chat.chat_models()

