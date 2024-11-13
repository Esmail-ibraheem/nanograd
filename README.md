<h1 align="center">Nanograd Engine - AI Ecosystem</h1>



nanograd is a neural net engine (**_command line interface and python library_**) inspired by **[micrograd](https://github.com/karpathy/micrograd)** and **[tinygrad](https://github.com/tinygrad/tinygrad)**, built upon a PyTorch-like API. It aims to provide users with easy-to-use tools for creating and utilizing various **neural network architectures**, including `GPT`, `llama`, `stable diffusion`, and `transformers` from scratch in Python. This library also offers `comprehensive data processing capabilities` from model training to providing a robust toolkit for developing and fine-tuning large language models. As the fastest and most efficient repository for continued pre-training and fine-tuning GPTs, nanograd sets a new standard in AI research and development. 




                      +----------------------------------------------------------------------------+
                      |     +----------------------------------------------------------------+     |
                      |     | Developers: Esmail Gumaan/ ML Engineer.                        |     |
                      |     | ( your Unreal Engine, but for AI, essentially making it an AI engine)|                
                      |     |     +----------------------------------------------------+     |     |
                      |     |     | Contributors: Those who wants to contribute        |     |     |
                      |     |     | (You make PR to this repo)                         |     |     |
                      |     |     +----------------------------------------------------+     |     |
                      |     +----------------------------------------------------------------+     |
                      +----------------------------------------------------------------------------+

 
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

- **`nanograd engine:`**
  An interface using Gradio that allows users to easily use any model they want without writing a single line of code—similar to Unreal Engine, but for AI, essentially making it an AI engine.

- **`Reinforcement Learning with gym environments:`**
  Currently providing some cool applications such as Carpole, and Mountain car, BipedalWalker with gym environments from OpenAI, and this feature is built upon tinygrad.
  
![Windows PowerShell 2024-09-18 09-41-04](https://github.com/user-attachments/assets/11f24f8a-7ed9-4fdc-89b5-eb17e953ec74)




---

## [nano models](https://github.com/Esmail-ibraheem/nanograd/tree/main/nanograd/models):

Whether you're working on text generation, image synthesis, image classification, or any other AI-driven task, nanograd models are designed to help you achieve your goals efficiently and effectively.

### 1. **GPT**
   The GPT (Generative Pre-trained Transformer) model implementation in nanograd is based on the transformer architecture. This model is designed to generate coherent and contextually relevant text based on a given input. It is a powerful tool for tasks such as text generation, completion, and more.

### 2. **LLaMA**
   LLaMA (Language Learning Model Architecture) is a sophisticated model designed to handle a variety of natural language processing (NLP) tasks. This implementation within nanograd provides an efficient and scalable way to work with large-scale language models, focusing on ease of use and performance.

### 3. **Stable Diffusion**
   The Stable Diffusion model in nanograd is tailored for generative tasks, particularly in creating images from textual descriptions. This model is optimized for both creativity and performance, enabling users to generate high-quality images in a streamlined process.

### 4. **Vision Transformer (ViT)**
   The Vision Transformer (ViT) is an innovative model that applies transformer architecture to image data. Unlike traditional convolutional neural networks (CNNs), ViT treats image patches as sequences, similar to how GPT processes text. This allows for powerful image classification capabilities, particularly in scenarios where large datasets are available for training.

### 5. **Custom Models (Ollama models)**
   In addition to the predefined models like GPT, LLaMA, Stable Diffusion, and Vision Transformer, this directory also serves as a repository for custom models built using the nanograd framework. Users are encouraged to contribute their models or adapt existing ones to fit their specific use cases.




---

## [nano nn engine](https://github.com/Esmail-ibraheem/nanograd/tree/main/nanograd/nn)

The `nn` directory in the **nanograd** library provides key components for building, training, and managing neural networks. This module is designed to offer a foundational setup for neural network operations, drawing inspiration from similar modules in established frameworks like PyTorch but tailored for the nanograd engine.

### Key Components

### 1. **Tensor Operations**
   - **`tensor.py`**: This file includes core functionalities related to tensor operations within the nanograd framework. It defines the data structures and operations for tensors, which are the fundamental building blocks for any neural network model. This component is responsible for handling tensor creation, manipulation, and arithmetic operations, which are crucial for performing forward and backward passes in neural networks.

### 2. **Neural Network Engine**
   - **`engine.py`**: The `engine.py` file manages the neural network's computational engine, including the implementation of the forward and backward passes. It is responsible for orchestrating the training process, handling gradient calculations, and updating model parameters. This engine provides the infrastructure needed to perform efficient training and evaluation of neural network models, focusing on performance and scalability within the nanograd framework.

### 3. **Training Utilities**
   - **`train_nn.py`**: This file contains utilities and functions to facilitate the training of neural network models. It includes training loops, evaluation metrics, and optimization routines. `train_nn.py` is designed to simplify the process of training models, making it easier to experiment with different architectures and hyperparameters. It serves as a practical guide for setting up and executing training sessions.

## Comparison to PyTorch

The `nn` module in nanograd provides similar functionalities to PyTorch’s `torch.nn` but with a streamlined and educational approach. Here’s how they compare:

- **Tensor Operations:** Like PyTorch, nanograd’s `tensor.py` handles the core tensor operations, but it aims to offer a more transparent and simplified implementation for educational purposes.

- **Neural Network Engine:** The `engine.py` file parallels PyTorch's `torch.autograd` and `torch.optim`, handling the essential computations and optimizations needed for training models. Nanograd’s engine focuses on efficiency and integration within the framework.

- **Training Utilities:** `train_nn.py` offers training utilities similar to PyTorch’s training utilities but is designed to be straightforward and easy to understand, making it suitable for learning and experimenting with neural network training processes.


-  **[autograd engine](https://github.com/Esmail-ibraheem/nanograd/tree/main/nanograd/nn)** provides a robust backpropagation mechanism over a dynamically constructed Directed Acyclic Graph (DAG). This tiny yet powerful engine mimics a PyTorch-like API, ensuring ease of use for those familiar with PyTorch.

### Key Features

- **Backpropagation over DAG**: The engine dynamically builds a DAG during the forward pass and performs backpropagation through this graph. The DAG operates over scalar values, breaking down each neuron into individual adds and multiplies.
- **Minimalistic Design**: The autograd engine consists of approximately 100 lines of code, and the accompanying neural network library is around 50 lines, emphasizing simplicity and clarity.
- **Educational Value**: This implementation, though minimal, is capable of constructing and training entire deep neural networks for tasks like binary classification. It's particularly useful for educational purposes, offering insights into the inner workings of backpropagation and neural networks.


---

## Get Started:
### 1- as a library, And CLI:
**clone the repo**
```
git clone https://github.com/Esmail-ibraheem/nanograd.git
```

**installing it on your computer:**
```
./install.bat
```

- **_using it in your main file:_**
  - **_nano models_**
  
  ```python
  import nanograd

  from nanograd.RL import Cartpole, car # import reinforcement learning package
  # Cartpole.run()
  # car.run()
  
  ###############################################################
  from nanograd.models.stable_diffusion import sd_inference
  sd_inference.run()
  
  ##############################################################
  from nanograd.analysis_lab import sentiment_analysis
  # sentiment_analysis.run()
  
  ############################################################
  from nanograd import generate_dataset 
  
  # generate_dataset.tokenize()
  
  ###########################################################
  
  from nanograd.models.llama import inference_llama 
  from nanograd.models.GPT import inference_gpt
  from nanograd.models.GPT import tokenizer
  
  # inference_gpt.use_model()
  
  # inference_llama.use_model()
  
  # tokenizer.run_tokenizer()
  ###########################################################
  from nanograd.models import ollama
  from nanograd.models import chat
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
  
  
  # import nanograd.nn.train_nn
  ```

   - **_nanograd and Pytorch Comparison:_**
     
     - **nanograd**
  
     ```python
     from nanograd.nn.tensor import Tensor

     # Create tensors with gradient tracking enabled
     a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
     b = Tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)
     
     # Perform some tensor operations
     c = a + b  # Element-wise addition
     d = a * b  # Element-wise multiplication
     e = c.sum()  # Sum all elements of the result of addition
     
     # Compute gradients
     e.backward()
     
     # Print the results and gradients
     print("Tensor a:")
     print(a.numpy())
     print("Tensor b:")
     print(b.numpy())
     print("Result of a + b:")
     print(c.numpy())
     print("Result of a * b:")
     print(d.numpy())
     print("Gradient of a:")
     print(a.grad.numpy())
     print("Gradient of b:")
     print(b.grad.numpy())

     ```
     - **Pytorch**
     
     ```python
     import torch

     # Create tensors with gradient tracking enabled
     a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
     b = torch.tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)
     
     # Perform some tensor operations
     c = a + b  # Element-wise addition
     d = a * b  # Element-wise multiplication
     e = c.sum()  # Sum all elements of the result of addition
     
     # Compute gradients
     e.backward()
     
     # Print the results and gradients
     print("Tensor a:")
     print(a)
     print("Tensor b:")
     print(b)
     print("Result of a + b:")
     print(c)
     print("Result of a * b:")
     print(d)
     print("Gradient of a:")
     print(a.grad)
     print("Gradient of b:")
     print(b.grad)

     ```   


- **_using it in your terminal:_**

**then type this command on your terminal: `nanograd`**
**_the output should be something like this_**
```
██     ██     ███     ██      ██  ████████     ██████   ████████       ███       ████████
███    ██   ██   ██   ███     ██ ██      ██   ██   ██   ██     ██    ██   ██     ██      ██
████   ██  ██     ██  ████    ██ ██      ██  ██         ██     ██   ██      ██   ██       ██
██ ██  ██ ██       ██ ██  ██  ██ ██      ██  ██  ████   ████████   ██        ██  ██       ██
██   ████ ███████████ ██    ████ ██      ██  ██    ██   ██    ██   ████████████  ██       ██
██    ███ ██       ██ ██      ██ ██      ██  ██    ██   ██    ██   ██        ██  ██      ██
██     ██ ██       ██ ██      ██  ████████    ██████    ██     ██  ██        ██  ████████
+----------------------------------------------------------------------------+
|     +----------------------------------------------------------------+     |
|     | Developers: Esmail Gumaan/ ML Engineer.                        |     |
|     | ( your Unreal Engine, but for AI, essentially making it an AI engine)|
|     |     +----------------------------------------------------+     |     |
|     |     | Contributors: Those who wants to contribute        |     |     |
|     |     | (You make PR to this repo)                         |     |     |
|     |     +----------------------------------------------------+     |     |
|     +----------------------------------------------------------------+     |
+----------------------------------------------------------------------------+
Nanograd-Engine🧠 Creator: Esmail Gumaan (The Mozart of AI and Mathematics (: ))
usage: nanograd [-h] {install,generate,download,pretrain,run_gpt,run_llama,run_engine,run_diffusion,run} ...

Nanograd CLI

positional arguments:
  {install,generate,download,pretrain,run_gpt,run_llama,run_engine,run_diffusion,run}
                        Sub-commands
    install             Install dependencies
    generate            Generate datasets
    download            Download checkpoints or llama
    pretrain            Pretrain a model
    run_gpt             Run GPT inference
    run_llama           Run LLaMA inference
    run_engine          Run engine interface
    run_diffusion       Run Stable Diffusion
    run                 Run model inference

options:
  -h, --help            show this help message and exit
```



> [!NOTE]
> You can use the nanograd library in your CLI, after installing it on your computer using `pip install -e`. then you can run any model you want,
> here are some commands to run in the terminal

| command | Sub-command | type this |
|---------|-------------|-----------|
| install | install dependencies | `nanograd install ollama` |
| download | download checkpoints or llama | `nanograd download checkpoint micorosoft/phi-2` or `nanograd download llama` | 
| run_gpt | run gpt inference | `nanograd run_gpt` | 
| run_llama | run llama inference | `nanograd run_llama` |
| run | run model inference (ollama models) | `nanograd run llama3.1` |
| run_diffusion | run stable diffusion | `nanograd run_diffusion stable_diffusion` |
| run_engine | run engine interface | `nanograd run_engine` |


## 2- [as an engine](https://github.com/Esmail-ibraheem/nanograd-engine/blob/main/nano_engine.py):
in this feature you can do all you can do in the CLI feature but with an interface, you do not need to type anything just some clicks.
- run stable diffusion
- run ollama models
- blueprints to generate different stories each time for the LLMs and stable diffusion
- download checkpoints using litgpt
- install ollama
  run this command\
   `python nano_engine.py`
- generating story themes from pre-made blueprints
![image](https://github.com/user-attachments/assets/1f27cc7c-c7d6-4290-8aa4-3798aafa99dc)


- multimodal chatbot
    - prompted-chatbot 
    - run BPE tokenizer
    - RAG chatbot
    - vision transformer

![image](https://github.com/user-attachments/assets/830d2961-f5b1-4633-acc2-97372055740e)
![image](https://github.com/user-attachments/assets/7084e557-fb84-4732-ab7e-83de4d8f9887)




- AutoTrainer (**`LLama-Factory`**) with bilingual supported languages: Arabic and English
  ![image](https://github.com/user-attachments/assets/30a696c8-501f-47ed-9f78-a0c0a5a217ee)

- AutoCoder (**Copilot**) code editor with AI for helping in coding
  ![image](https://github.com/user-attachments/assets/a99932d2-1107-4567-9eac-95c74f84c7bd)




### 3- as an API:**still in developing process**
run this command `python nanograd_API.py`;.;.

> [!IMPORTANT]
> you can't run any of this unless you do downloaded the checkpoints before, for the litgpt you can download their models(Checkpoints) after you installed the library run this `nanograd download checkpoints`, for the ollama models also the same, for the stable diffusion download them from huggingface.  

---

## ToDo:

- [ ] Path Management for other OSs.
- [ ] Parallelization: multiprocessing.
- [ ] Cuda Support.
     - [ ] rebuild the engine using C++. 
- [ ] vision transformer.
- [ ] 3D Tobia XL.
- [ ] memory efficient weight loading.
- [ ] AI Mathematician.
- [ ] Gamengen: games generation.
- [x] stable diffusion.
- [x] ollama models support.
- [x] llama.
     - [x] transformer architecture 
- [x] GPT.
     - [x] litGPT checkpoints installation
     - [x] GPT Architecture
     - [x] transformer architecture   
- [x] AutoTrainer.
- [x] AutoCoder (**Copilot**).



---

## Contributing
We welcome contributions from the community. if you want to contribute just by:
- adding models built from scratch to make them accessible without the internet needed (offline).
- Fix some issues, if there are any, or fix something you discovered, and explain what you are trying to fix and why.
- solving or building one of the ToDos list 

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
  month   = "May",
  note    = "[Online; accessed 2024-05-24]",
}
```


---

## Notes and Acknowledgments
- [Hugging-Face Transformers](https://github.com/huggingface/transformers)
- [Lightning AI](https://github.com/Lightning-AI)
- [Karpathy's miniGPT](https://github.com/karpathy/minGPT)
- [GeoHotz's tinygrad](https://github.com/tinygrad/tinygrad)
- [OpenAI for pioneering the GPT architecture](https://www.bing.com/ck/a?!&&p=9a9de5d499443c51JmltdHM9MTcyNzkxMzYwMCZpZ3VpZD0zMWRiYzRjZS1mZDI5LTY0Y2QtMmIxNy1kNTM1ZmM3YjY1ZjYmaW5zaWQ9NTQ4MQ&ptn=3&ver=2&hsh=3&fclid=31dbc4ce-fd29-64cd-2b17-d535fc7b65f6&psq=gpt+model+paper&u=a1aHR0cHM6Ly9jZG4ub3BlbmFpLmNvbS9yZXNlYXJjaC1jb3ZlcnMvbGFuZ3VhZ2UtdW5zdXBlcnZpc2VkL2xhbmd1YWdlX3VuZGVyc3RhbmRpbmdfcGFwZXIucGRm&ntb=1)
- [Meta for pioneering the Llama architecture](https://www.bing.com/ck/a?!&&p=a6ef05b712210c58JmltdHM9MTcyNzkxMzYwMCZpZ3VpZD0zMWRiYzRjZS1mZDI5LTY0Y2QtMmIxNy1kNTM1ZmM3YjY1ZjYmaW5zaWQ9NTIwNQ&ptn=3&ver=2&hsh=3&fclid=31dbc4ce-fd29-64cd-2b17-d535fc7b65f6&psq=llama+paper&u=a1aHR0cHM6Ly9hcnhpdi5vcmcvYWJzLzIzMDIuMTM5NzE&ntb=1)
- [hiyouga-PhD](https://github.com/hiyouga)

> [!NOTE]
> **nanograd** was developed with a specific vision and purpose in mind. This section provides insight into the motivations behind its creation and the approach taken during its development. The primary motivation for developing nanograd was to create a framework that emphasizes simplicity and transparency in neural network design and training. While many existing frameworks, such as PyTorch and TensorFlow, offer extensive features and optimizations, they can also be complex and challenging to fully understand, particularly for those new to deep learning or for educational purposes. nanograd aims to simplify access to models like GPT, LLaMA, Stable Diffusion, and others, making these technologies more approachable and easier to work with.

---

## Contact
For any inquiries or feedback, please open an issue on GitHub or reach out to the project maintainer at esm.agumaan@gmail.com.
