# TensorFuzz: Coverage Guided Fuzzing for Neural Networks

This repository contains a library for performing coverage guided fuzzing of neural networks,
as was described in [this paper](https://arxiv.org/abs/1807.10875).
It's still a prototype, but the ultimate goal is for people to actually use this to test real software.
Any suggestions about how to make it more useful for that purpose would be appreciated.

## Installation

You ought to be able to run the code in this repository by doing the following:

```
pip install -r requirements.txt
```

Then do:

```
export PYTHONPATH="$PYTHONPATH:$HOME/tensorfuzz"
```

## The structure of this repository

Broadly speaking, this repository contains a core fuzzing library, examples of how 
to use the fuzzer, a list of bugs found with the fuzzer, and some utilities.

### /bugs

This directory contains bugs or weird behaviors that we've found by using this tool.

### /examples

This directory contains examples of how to use the fuzzer in several different ways.
It contains examples of looking for numerical errors, finding broken loss functions
in publicly available code, and checking for disagreements between trained classifiers
and their quantized versions.

### /lib

This directoy contains the fuzzing engine and all the necessary utils.

### /third_party

This directory contains code written by other people and the (potentially updated) 
LICENSES for that code. 


## Disclaimers

This is not an officially supported Google product.
