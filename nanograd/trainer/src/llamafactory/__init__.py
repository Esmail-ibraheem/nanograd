

r"""
Efficient fine-tuning of large language models.

Level:
  api, webui > chat, eval, train > data, model > hparams > extras

Dependency graph:
  main:
    transformers>=4.41.2,<=4.45.0
    datasets>=2.16.0,<=2.21.0
    accelerate>=0.30.1,<=0.34.2
    peft>=0.11.1,<=0.12.0
    trl>=0.8.6,<=0.9.6
  attention:
    transformers>=4.42.4 (gemma+fa2)
  longlora:
    transformers>=4.41.2,<=4.45.0
  packing:
    transformers>=4.41.2,<=4.45.0

Disable version checking: DISABLE_VERSION_CHECK=1
Enable VRAM recording: RECORD_VRAM=1
Force check imports: FORCE_CHECK_IMPORTS=1
Force using torchrun: FORCE_TORCHRUN=1
Set logging verbosity: LLAMAFACTORY_VERBOSITY=WARN
Use modelscope: USE_MODELSCOPE_HUB=1
"""

from .extras.env import VERSION


__version__ = VERSION
