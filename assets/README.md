
# Nanograd: Your Ultimate Neural Net Engine

## Introduction
**Nanograd** is a neural net engine inspired by micrograd and tinygrad, built upon a PyTorch-like API. It aims to provide users with easy-to-use tools for creating and utilizing various neural network architectures, including GPT, llama, stable diffusion, RNN, CNN, and transformers. This library also offers comprehensive data processing capabilities.

## Features
1. **GPT Model with Checkpoints**
   - Pre-trained GPT models available for immediate use.
   - Customizable checkpoints for fine-tuning and continued training.

2. **Llama Model with Checkpoints**
   - Includes pre-trained Llama models.
   - Checkpoints for Llama-7B and others available for download.

3. **Stable Diffusion**
   - Implements stable diffusion models.
   - Based on work from Umar Jamil and the user's DDPM repository.

4. **Reinforcement Learning Techniques**
   - Integration of various RL algorithms and techniques.

5. **Data Cleaning and Pipeline Operations**
   - Data processing tools inspired by litdata from Lightning AI.
   - End-to-end data operations from cleaning to pipeline execution.

## Team Roles
- **AI Development**: Esmail Gumaan
  - Responsible for neural network, RL, and model implementations.
- **Data Engineering**: Ibraheem Sultan
  - Handles data pipeline, checkpoints, and processing tasks.
- **Low-Level Programming**: Ahmed AL-Kateeb
  - Focuses on CUDA, C++ optimizations, and OS-level programming.
- **Website Design**: Ibraheem Al-Hitari
  - Develops the project website and manages its visual profile.

## Planned Models and Functions
- **GPT Architecture**: Derived from litGPT and biogpt from Hugging Face Transformers.
- **Llama Architecture**: Based on litLlama and Xllama.
- **Optimizers**:
  - `from nanograd.optimizers import Adam`
  - `from nanograd.optimizers import Sophia-G`
- **Normalization Techniques**:
  - `from nanograd.norm import batch_norm`
  - `from nanograd.norm import layer_norm`

## Sample Code Snippets
```python
from nanograd.llama import Llama
llama = Llama(prompt="", dataset="")

from nanograd.nn import CNN, RNN, GPT, Transformer
cnn = CNN(input_neurons=, output_neurons=)
rnn = RNN()
gpt = GPT()
transformer = Transformer()

from nanograd.sd import StableDiffusion
sd = StableDiffusion()
sd.generate()

from nanograd.nn import modules, optimizers
rnn_module = modules.RNN()
cnn_module = modules.CNN()
transformer_module = modules.Transformer()

adam_optimizer = optimizers.Adam()
adamw_optimizer = optimizers.AdamW()

from nanograd.models import llama, stable_diffusion
llama.generate()
sd.generate()

from nanograd.data import pipeline, checkpoints
from nanograd.RL import QLearning
```

## Usage of Data and Checkpoints
```python
from nanograd.data import pipeline, checkpoints
# Data processing and loading checkpoints for models
```

## Reinforcement Learning Components
```python
from nanograd.RL import QLearning, other_algorithms
```

## Web Interface and Deployment
- Exploring web UI options like Gradio or Streamlit for user interaction.

## Nanograd Computer (Nano Computer)
- Building small computers using Raspberry Pi.
- Installing Ubuntu OS and testing the nanograd library on them.
- **Logo Command**: `nanograd on nano computer`

## Project Development Plan
### Initial Steps
1. Understand the principles of building a library to integrate into Python.
2. Discuss and explain various ideas within the team to get a comprehensive understanding of the project.

---

