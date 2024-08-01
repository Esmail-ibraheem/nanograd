# nanograd🧬: 
![nanograd](https://github.com/user-attachments/assets/09cca76c-cd3a-4335-9a34-c026854f5124) 

nanograd is a neural net engine (**_command line interface and python library_**) inspired by **[micrograd](https://github.com/karpathy/micrograd)** and **[tinygrad](https://github.com/tinygrad/tinygrad)**, built upon a PyTorch-like API. It aims to provide users with easy-to-use tools for creating and utilizing various **neural network architectures**, including `GPT`, `llama`, `stable diffusion`, and `transformers` from scratch in Python. It is optimized for performance with `CUDA` and utilizes state-of-the-art `pre-training` and `fine-tuning` techniques.  This library also offers `comprehensive data processing capabilities`. from model training to providing a robust toolkit for developing and fine-tuning large language models. As the fastest and most efficient repository for continued pre-training and fine-tuning GPTs, nanograd sets a new standard in AI research and development. 


 
## nano Features

| ✔️ | LlAMa                       | ✔️ | GPT                          | ✔️ | Stable diffusion            |
|---|------------------------------|---|------------------------------|---|------------------------------|
| ✔️ | Transformers                 | ✔️ | ollama | ✔️ | Dataset generator               |
| ✔️ | multiple checkpoints             | ✔️ | BioGPT | ✔️ | From scratch implementations|
| ✔️ | Beginner friendly             | ✔️ | No Abstraction                        | ✔️ | pretraining and fine-tuning           |
   






**_Key Features highlight_**
- **`Transformers from Scratch:`**
  Implement GPT and llama [models](#models-implementation) from the ground up using Pytorch, offering full transparency and flexibility.

- **`Efficient Pre-training and Fine-tuning:`**
  Used the instructions from LitGPT repo for pre-training and fine-tuning the model for Both datasets: `Arabic` and `English`, here are the [instructions](#Continued-Pre-training) section

- **`Custom Tokenization:`**
  Implement a Byte Pair Encoding (BPE) tokenizer from scratch, tailored for the specific needs of your model and dataset.

- **`CUDA-Optimized Kernels:`**
  Integrate custom CUDA kernels for critical GPT operations to leverage GPU acceleration, significantly improving training and inference speed.
  


## models implementation:
1. GPT
2. LlAma
3. Bio GPT
<p align="center"> <img src="https://github.com/Esmail-ibraheem/omniGPT/blob/main/assets/GPT.jpeg" alt="GPT architecture overview" ></p> 

## Continued-Pre-training: 
### 1- Continue Pretraining an LLM
In this section we're going to learn how to continue pretraining an LLM with LitGPT. Here's the full code, which we'll break down into steps:
```Bash
# 1) Download the model (or use your own)
litgpt download \
  --repo_id EleutherAI/pythia-160m \
  --tokenizer_only True

# 2) Continue pretraining the model
litgpt pretrain \
  --model_name pythia-160m \
  --tokenizer_dir checkpoints/EleutherAI/pythia-160m \
  --initial_checkpoint_dir checkpoints/EleutherAI/pythia-160m \
  --data TextFiles \
  --data.train_data_path "custom_texts" \
  --out_dir out/custom_model

# 3) Chat with the model
litgpt chat \
  --checkpoint_dir out/custom_model/final

# 4) Deploy the model
litgpt serve \
  --checkpoint_dir out/custom_model/final
```
### 2- Choose the model (checkpoints)

you can choose a model(checkpoints) from the models that litgpt support it: `litgpt download list`
| Model                   | Model size                 | Author                    | Reference                               |
|-------------------------|----------------------------|---------------------------|-----------------------------------------|
| CodeGemma               | 7B                         | Google                    | [Google Team, Google Deepmind](https://ai.google.dev/gemma/docs/codegemma) |
| Code Llama              | 7B, 13B, 34B, 70B          | Meta AI                   | [Rozière et al. 2023](https://arxiv.org/abs/2308.12950) |
| Danube2                 | 1.8B                       | H2O.ai                    | [H2O.ai](https://h2o.ai/platform/danube-1-8b/)                |
| Dolly                   | 3B, 7B, 12B                | Databricks                | [Conover et al. 2023](https://www.databricks.com/blog/2023/04/12/dolly-first-open-commercially-viable-instruction-tuned-llm) |
| Falcon                  | 7B, 40B, 180B              | TII UAE                   | [TII 2023](https://falconllm.tii.ae/)         |
| FreeWilly2 (Stable Beluga 2) | 70B                | Stability AI              | [Stability AI 2023](https://stability.ai/blog/stable-beluga-large-instruction-fine-tuned-models)|
| Function Calling Llama 2 | 7B                        | Trellis                   | [Trellis et al. 2023](https://huggingface.co/Trelis/Llama-2-7b-chat-hf-function-calling-v2) |
| Gemma                   | 2B, 7B                     | Google                    | [Google Team, Google Deepmind](https://storage.googleapis.com/deepmind-media/gemma/gemma-report.pdf) |
| Llama 2                 | 7B, 13B, 70B               | Meta AI                   | [Touvron et al. 2023](https://arxiv.org/abs/2307.09288) |
| Llama 3                 | 88B, 70B                   | Meta AI                   | [Meta AI 2024](https://github.com/meta-llama/llama3)     |
| LongChat                | 7B, 13B                    | LMSYS                     | [LongChat Team 2023](https://lmsys.org/blog/2023-06-29-longchat/) |
| MicroLlama              | 300M                       | Ken Wang                  | [MicroLlama repo](https://github.com/keeeeenw/MicroLlama)  |
| Mixtral MoE             | 8x7B                       | Mistral AI                | [Mistral AI 2023](https://mistral.ai/news/mixtral-of-experts/)  |
| Mistral                 | 7B                         | Mistral AI                | [Mistral AI 2023](https://mistral.ai/news/announcing-mistral-7b/)  |
| Nous-Hermes             | 7B, 13B, 70B               | NousResearch              | [Org page](https://huggingface.co/NousResearch)         |
| OpenLLaMA               | 3B, 7B, 13B                | OpenLM Research           | [Geng & Liu 2023](https://github.com/openlm-research/open_llama)  |
| Phi 1.5 & 2             | 1.3B, 2.7B                 | Microsoft Research        | [Li et al. 2023](https://arxiv.org/abs/2309.05463)   |
| Phi 3                   | 3.8B                       | Microsoft Research        | [Abdin et al. 2024](https://arxiv.org/abs/2404.14219)|
| Platypus                | 7B, 13B, 70B               | Lee et al.                | [Lee, Hunter, and Ruiz 2023](https://arxiv.org/abs/2308.07317) |
| Pythia                  | (14, 31, 70, 160, 410)M, (1.1, 4.2, 8.6, 9, 12)B | EleutherAI | [Biderman et al. 2023](https://arxiv.org/abs/2304.01373) |
| RedPajama-INCITE        | 3B, 7B                     | Together                  | [Together 2023](https://together.ai/blog/redpajama-models-v1)    |
| StableCode              | 3B                         | Stability AI              | [Stability AI 2023](https://stability.ai/blog/stablecode-llm-generative-ai-coding)|
| StableLM                | 3B, 7B                     | Stability AI              | [Stability AI 2023](https://github.com/Stability-AI/StableLM)|
| StableLM Zephyr         | 3B                         | Stability AI              | [Stability AI 2023](https://stability.ai/blog/stablecode-llm-generative-ai-coding)|
| TinyLlama               | 1.1B                       | Zhang et al.              | [Zhang et al. 2023](https://github.com/jzhang38/TinyLlama)|
| Vicuna                  | 7B, 13B, 33B               | LMSYS                     | [Li et al. 2023](https://lmsys.org/blog/2023-03-30-vicuna/)   |

to download a pre-trained model, write this: `litgpt download --repo_id microsoft/phi-2`

### 3- Prepare your data
this command will create a folder called "custom_texts" which has 2 text files, you can replace the text files, with your dataset:
```Bash
# 2) Pretain the model
litgpt pretrain \
  --model_name pythia-160m \
  --tokenizer_dir checkpoints/EleutherAI/pythia-160m \
  --initial_checkpoint_dir checkpoints/EleutherAI/pythia-160m \
  --data TextFiles \
  --data.train_data_path "custom_texts" \
  --out_dir out/custom_model
```

> [!WARNING]  
> Using this approach is only recommended for small datasets. Since text data is highly compressible, it is often stored in compressed format, and often in file formats where documents can be loaded row by row without having to load entire files at once. In other words, this `TextFiles` approach is only feasible to store the data in plain text files due to the limited size. For datasets that take up multiple gigabytes, we recommend preprocessing it with another way...

### 4- Deploy the model

First, verify the model works well:
```Bash
# 3) Chat with the model
litgpt chat \
  --checkpoint_dir out/custom_model/final
```
Now deploy the model:
```Bash
# 4) Deploy the model
litgpt serve \
  --checkpoint_dir out/custom_model/final
```

**_for more check this lightning ai studio: [pre-training LLMs](https://lightning.ai/lightning-ai/studios/litgpt-continue-pretraining) _**

---


## Contributing
We welcome contributions from the community. Please read our contributing guidelines and submit your pull requests.

## License
This project is licensed under the MIT License. See the [LICENSE file](https://github.com/Esmail-ibraheem/omniGPT?tab=MIT-1-ov-file#) for more details.

---

## Citations
```BibTex 
@misc{Gumaan2024-nanograd,
  title   = "nanograd",
  author  = "Gumaan, Esmail",
  howpublished = {\url{https://github.com/Esmail-ibraheem/nanograd}},
  year    = "2024",
  month   = "",
  note    = "[Online; accessed 2024-05-24]",
}
```


---

## Notes and Acknowledgments
- Hugging-Face Transformers
- Lightning AI
- NVIDIA FasterTransformer
- OpenAI for pioneering the GPT architecture

> [!NOTE]
> I developed the project 'nanograd' to enhance my skills in developing, pre-training, and fine-tuning models with LitGPT and LitData. This project is based on the Transformers architecture, specifically utilizing the Generative Pre-trained Transformer (GPT) and LLaMA models. Notably, I worked on developing GPT and LLaMA models using both Python and CUDA, facilitating training on either GPU or CPU and building the tokenizers. This project integrates LitGPT and Transformers' GPT-2.

---

## Contact
For any inquiries or feedback, please open an issue on GitHub or reach out to the project maintainer at esm.agumaan@gmail.com.
