# 🚀omniGPT                                                   
The fastest and most efficient repository for fine-tuning GPTs, implementing various PEFT techniques like LoRA and adapters, as well as quantization, FlashAttention, and sentiment analysis using PPO.
omniGPT is an ambitious project aimed at building a GPT (Generative Pre-trained Transformer) model from scratch in Python, complemented by state-of-the-art fine-tuning techniques and optimized for performance with CUDA. This project encompasses everything from tokenization to model training, making it a robust toolkit for developing and fine-tuning large language models.
<p align="center">
✅ flash attention    ✅ LoRA                  ✅ Adapter<br>
✅ Quantization       ✅ GPT_kernels for GPUs  ✅ sentiment analysis     
</p>

[![Python](https://img.shields.io/badge/Language-Python-blue)](https://en.wikipedia.org/wiki/Python_(programming_language))
[![CUDA](https://img.shields.io/badge/Language-CUDA-green)](https://en.wikipedia.org/wiki/CUDA)
[![Jupyter Notebook](https://img.shields.io/badge/Tool-Jupyter%20Notebook-orange)](https://en.wikipedia.org/wiki/Project_Jupyter)
[![C++](https://img.shields.io/badge/Language-C++-red)](https://en.wikipedia.org/wiki/C%2B%2B)


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
  
## omniGPT overview:
<p align="center"> <img src="https://github.com/Esmail-ibraheem/omniGPT/blob/main/assets/omniGPT-architecture.drawio.svg" alt="omniGPT overview" ></p> 

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

## Citation
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

---

## Notes and Acknowledgments
- Hugging-Face Transformers
- litgpt 
- NVIDIA FasterTransformer
- OpenAI for pioneering the GPT architecture

---

## Contact
For any inquiries or feedback, please open an issue on GitHub or reach out to the project maintainer at esm.agumaan@gmail.com.
