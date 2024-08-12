# nanograd 
![nanograd](https://github.com/user-attachments/assets/09cca76c-cd3a-4335-9a34-c026854f5124) 

nanograd is a neural net engine (**_command line interface and python library_**) inspired by **[micrograd](https://github.com/karpathy/micrograd)** and **[tinygrad](https://github.com/tinygrad/tinygrad)**, built upon a PyTorch-like API. It aims to provide users with easy-to-use tools for creating and utilizing various **neural network architectures**, including `GPT`, `llama`, `stable diffusion`, and `transformers` from scratch in Python. This library also offers `comprehensive data processing capabilities` from model training to providing a robust toolkit for developing and fine-tuning large language models. As the fastest and most efficient repository for continued pre-training and fine-tuning GPTs, nanograd sets a new standard in AI research and development. 


 
## nano Features

| ✔️ | LlAMa                       | ✔️ | GPT                          | ✔️ | Stable diffusion            |
|---|------------------------------|---|------------------------------|---|------------------------------|
| ✔️ | Transformers                 | ✔️ | ollama | ✔️ | Dataset generator(ngram model)               |
| ✔️ | Vision Transformer             | ✔️ | BioGPT | ✔️ | From scratch implementations|
| ✔️ | Beginner friendly             | ✔️ | No Abstraction                        | ✔️ | pretraining and fine-tuning           |
   






**_Key Features highlight_**
- **`Transformers from Scratch:`**
  Implement GPT and llama [models](#models-implementation) from the ground up using Pytorch, offering full transparency and flexibility.

- **`Neural Network engine for scalar:`**
  Implementing mathematical operations similar to PyTorch, from addition to matrix multiplication (matmul), including ReLU and sigmoid activation functions, and training a simple feedforward neural network architecture

- **`Custom Tokenization:`**
  Implement a Byte Pair Encoding (BPE) tokenizer from scratch, tailored for the specific needs of your model and dataset.

- **`Command-Line Interface:`**
  Providing a command line interface (CLI) to easily access built-in models without writing any code, making it accessible for both programmers and regular users (:.

- **`Reinforcement Learning with gym environments:`**
  Currently providing some cool applications such as Carpole, and Mountain car, BipedalWalker with gym environments from OpenAI, and this feature is built upon tinygrad.
  



---

# nanograd/models

Whether you're working on text generation, image synthesis, image classification, or any other AI-driven task, nanograd models are designed to help you achieve your goals efficiently and effectively.

## Models Included

### 1. **GPT**
   The GPT (Generative Pre-trained Transformer) model implementation in nanograd is based on the transformer architecture. This model is designed to generate coherent and contextually relevant text based on a given input. It is a powerful tool for tasks such as text generation, completion, and more.

### 2. **LLaMA**
   LLaMA (Language Learning Model Architecture) is a sophisticated model designed to handle a variety of natural language processing (NLP) tasks. This implementation within nanograd provides an efficient and scalable way to work with large-scale language models, focusing on ease of use and performance.

### 3. **Stable Diffusion**
   The Stable Diffusion model in nanograd is tailored for generative tasks, particularly in creating images from textual descriptions. This model is optimized for both creativity and performance, enabling users to generate high-quality images in a streamlined process.

### 4. **Vision Transformer (ViT)**
   The Vision Transformer (ViT) is an innovative model that applies transformer architecture to image data. Unlike traditional convolutional neural networks (CNNs), ViT treats image patches as sequences, similar to how GPT processes text. This allows for powerful image classification capabilities, particularly in scenarios where large datasets are available for training.

### 5. **Custom Models**
   In addition to the predefined models like GPT, LLaMA, Stable Diffusion, and Vision Transformer, this directory also serves as a repository for custom models built using the nanograd framework. Users are encouraged to contribute their models or adapt existing ones to fit their specific use cases.

<p align="center"> <img src="https://github.com/Esmail-ibraheem/omniGPT/blob/main/assets/GPT.jpeg" alt="GPT architecture overview" ></p> 


---

## Autograd implementation:
The nanograd [autograd engine](https://github.com/Esmail-ibraheem/nanograd/tree/main/nanograd/nn) provides a robust backpropagation mechanism over a dynamically constructed Directed Acyclic Graph (DAG). This tiny yet powerful engine mimics a PyTorch-like API, ensuring ease of use for those familiar with PyTorch.

### Key Features

- **Backpropagation over DAG**: The engine dynamically builds a DAG during the forward pass and performs backpropagation through this graph. The DAG operates over scalar values, breaking down each neuron into individual adds and multiplies.
- **Minimalistic Design**: The autograd engine consists of approximately 100 lines of code, and the accompanying neural network library is around 50 lines, emphasizing simplicity and clarity.
- **Educational Value**: This implementation, though minimal, is capable of constructing and training entire deep neural networks for tasks like binary classification. It's particularly useful for educational purposes, offering insights into the inner workings of backpropagation and neural networks.


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
