from distutils.core import setup

setup(
    name="pytorch-neat",
    version="1.0",
    license="Apache License 2.0",
    description="An extension of NEAT-Python using PyTorch",
    long_description=(
        "PyTorch NEAT builds upon NEAT-Python by providing some functions which can"
        " turn a NEAT-Python genome into either a recurrent PyTorch network or a"
        " PyTorch CPPN for use in HyperNEAT or Adaptive HyperNEAT."
    ),
    author="Alex Gajewsky",
    maintainer_email="joel.lehman@uber.com",
    url="https://github.com/uber-research/PyTorch-NEAT",
    packages=["pytorch_neat"],
    install_requires=[
        "neat-python>=0.92",
        "numpy>=1.14.3",
        "gym>=0.10.5",
        "click>=6.7",
        "torch>=0.4.0",
    ]
)
