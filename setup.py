from setuptools import setup

setup(
    name="starcoder-safety-eval",
    description="Code for toxicity and bias evaluations in StarCoder paper.",
    version="0.0.1",
    url="https://github.com/McGill-NLP/StarCoderSafetyEval",
    python_requires=">=3.8",
    install_requires=[
        "torch",
        "transformers",
        "accelerate",
        "evaluate",
        "datasets",
        "parlai",
    ],
    include_package_data=True,
    zip_safe=False,
)
