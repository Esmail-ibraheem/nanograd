# omniGPT                                                   
The fastest and most efficient repository for fine-tuning GPTs, implementing various PEFT techniques like LoRA and adapters, as well as quantization, FlashAttention, and sentiment analysis using PPO.
omniGPT is an ambitious project aimed at building a GPT (Generative Pre-trained Transformer) model from scratch in Python, complemented by state-of-the-art fine-tuning techniques and optimized for performance with CUDA. This project encompasses everything from tokenization to model training, making it a robust toolkit for developing and fine-tuning large language models.

**_Key Features_**
- GPT Model from Scratch:
  Implement a GPT model from the ground up using Python, offering full transparency and flexibility.

- Efficient Fine-Tuning with PEFT Techniques:
  LoRA (Low-Rank Adaptation): Fine-tune models efficiently by reducing the number of trainable parameters.
  Adapters: Insert small trainable modules within the model to enable rapid fine-tuning on new tasks.
  Quantization: Optimize the model for faster inference by reducing the precision of weights without significant loss in accuracy.

- Custom Tokenization:
  Implement a Byte Pair Encoding (BPE) tokenizer from scratch, tailored for the specific needs of your model and dataset.

- CUDA-Optimized Kernels:
  Integrate custom CUDA kernels for critical GPT operations to leverage GPU acceleration, significantly improving training and inference speed.
  
## omniGPT overview:
<p align="center"> <img src="https://github.com/Esmail-ibraheem/omniGPT/blob/main/assets/omniGPT-architecture.drawio.svg" alt="omniGPT overview" ></p> 

## GPT architecture:
<p align="center"> <img src="https://github.com/Esmail-ibraheem/omniGPT/blob/main/assets/GPT.jpeg" alt="GPT architecture overview" ></p> 

## fine-tuning (PEFT):
<p align="center"> <img src="https://github.com/Esmail-ibraheem/omniGPT/blob/main/assets/LoRAoverview.jpeg" alt="LoRAs" ></p> 
