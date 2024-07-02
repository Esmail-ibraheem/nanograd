# OmniGPT (Transformers):      
OmniGPT is an ambitious project inspired by [LitGPT](https://github.com/Lightning-AI/litgpt) from [Lightning AI](https://github.com/Lightning-AI), designed to build Generative Pre-trained Transformers (GPT) models from scratch in Python. It is optimized for performance with CUDA and utilizes state-of-the-art pre-training and fine-tuning techniques. This project covers everything from tokenization to model training, providing a robust toolkit for developing and fine-tuning large language models. As the fastest and most efficient repository for continued pre-training and fine-tuning GPTs, OmniGPT sets a new standard in AI research and development.

<p align="center">
  <a href="#GPT">
    <img src="https://img.shields.io/badge/Model-GPT%20model-%23FFD700" alt="GPT">
  </a>
  <a href="#Pretraining">
    <img src="https://img.shields.io/badge/Feature-Pretraining-%23FF7F50" alt="Pretraining">
  </a>
  <a href="#Fine-tuning">
    <img src="https://img.shields.io/badge/Feature-Fine_tuning-%23FF7F50" alt="Fine-tuning">
  </a>
  <a href="https://github.com/Esmail-ibraheem/omniGPT?tab=readme-ov-file#fine-tuning-peft">
    <img src="https://img.shields.io/badge/Feature-Quantization-%23FF7F50" alt="Quantization">
  </a>
  <a href="#Cuda-GPT-Llama">
    <img src="https://img.shields.io/badge/Cuda-GPT_Llama%20for%20GPUs-%23228B22" alt="Model Cuda">
  </a>
  <a href="#Llama">
    <img src="https://img.shields.io/badge/Model-Llama%20model-%23FFD700" alt="Llama model">
  </a>
</p>





**_Key Features_**
- **`Transformers from Scratch:`**
  Implement a GPT and llama models from the ground up using Pytorch, offering full transparency and flexibility.

- **`Efficient Pre-training and Fine-tuning:`**
  Used the instructions from LitGPT repo for pre-training and fine-tuning the model for Both datasets: `Arabic` and `English`, here are the [instructions](#Continued-Pre-training) section

- **`Custom Tokenization:`**
  Implement a Byte Pair Encoding (BPE) tokenizer from scratch, tailored for the specific needs of your model and dataset.

- **`CUDA-Optimized Kernels:`**
  Integrate custom CUDA kernels for critical GPT operations to leverage GPU acceleration, significantly improving training and inference speed.
  


## GPT architecture:
<p align="center"> <img src="https://github.com/Esmail-ibraheem/omniGPT/blob/main/assets/GPT.jpeg" alt="GPT architecture overview" ></p> 

## Continued-Pre-training: Continue Pretraining an LLM
In this section we're going to learn how to continue pretraining an LLM with LitGPT. Here's the full code, which we'll break down into steps:
```
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

---


## Contributing
We welcome contributions from the community. Please read our contributing guidelines and submit your pull requests.

## License
This project is licensed under the MIT License. See the [LICENSE file](https://github.com/Esmail-ibraheem/omniGPT?tab=MIT-1-ov-file#) for more details.

---

## Citations
```BibTex
@misc{Gumaan2024-omniGPT,
  title   = "omniGPT",
  author  = "Gumaan, Esmail",
  howpublished = {\url{https://github.com/Esmail-ibraheem/omniGPT}},
  year    = "2024",
  month   = "",
  note    = "[Online; accessed 2024-05-24]",
}
```

```BibTex
@article{Gumaan2024omniGPT,
  title={omniGPT: Comprehensive Survey of Generative Pre-Trained Transformers with PEFT and CUDA Optimization},
  author={Esmail Gumaan},
  year={2024},
  journal={حاليا مدري},
}

```

---

## Notes and Acknowledgments
- Hugging-Face Transformers
- litgpt 
- NVIDIA FasterTransformer
- OpenAI for pioneering the GPT architecture

---

## Contact
For any inquiries or feedback, please open an issue on GitHub or reach out to the project maintainer at esm.agumaan@gmail.com.
