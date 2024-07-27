from setuptools import setup, find_packages

setup(
    name='nanograd',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'torch', 
        'argparse',
        'tensorboard',
        'wget',
        'transformers',
        'litgpt',
        'tiktoken',
        'sentencepiece',
        'tqdm',
        'regex',
    ],
    entry_points={
        'console_scripts': [
            'nanograd=nanograd.nanograd_CLI:main',
        ],
    },
)
