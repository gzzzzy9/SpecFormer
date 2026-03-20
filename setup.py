from setuptools import setup, find_packages

setup(
    name="specformer",
    version="0.1.0",
    description="Transformer-based protein language model for antigen-specific BCR sequences",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "einops>=0.7.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "wandb>=0.16.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
    ],
)
