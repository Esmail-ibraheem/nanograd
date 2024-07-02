# OmniGPT (Transformers):      
The fastest and most efficient repository for continued-pre-training, and fine-tuning GPTs inspired from litgpt from lightning AI,
omniGPT is an ambitious project aimed at building a GPT (Generative Pre-trained Transformers) models from scratch in Python, complemented by state-of-the-art Pre-training and fine-tuning techniques and optimized for performance with CUDA. This project encompasses everything from tokenization to model training, making it a robust toolkit for developing and fine-tuning large language models.

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
- **`GPT Model from Scratch:`**
  Implement a GPT model from the ground up using Python, offering full transparency and flexibility.

- **`Efficient Fine-Tuning with PEFT Techniques:`**
  LoRA (Low-Rank Adaptation): Fine-tune models efficiently by reducing the number of trainable parameters.
  Adapters: Insert small trainable modules within the model to enable rapid fine-tuning on new tasks.
  Quantization: Optimize the model for faster inference by reducing the precision of weights without significant loss in accuracy.

- **`Custom Tokenization:`**
  Implement a Byte Pair Encoding (BPE) tokenizer from scratch, tailored for the specific needs of your model and dataset.

- **`CUDA-Optimized Kernels:`**
  Integrate custom CUDA kernels for critical GPT operations to leverage GPU acceleration, significantly improving training and inference speed.
  


## GPT architecture:
<p align="center"> <img src="https://github.com/Esmail-ibraheem/omniGPT/blob/main/assets/GPT.jpeg" alt="GPT architecture overview" ></p> 

## fine-tuning (PEFT):
<p align="center"> <img src="https://github.com/Esmail-ibraheem/omniGPT/blob/main/assets/LoRAoverview.jpeg" alt="LoRAs" ></p> 


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
