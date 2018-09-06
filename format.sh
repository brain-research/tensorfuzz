black -l 79 bugs/
black -l 79 examples/
black -l 79 lib/
pylint bugs/*.py
pylint lib/*.py
pylint examples/dcgan/*.py
pylint examples/nans/*.py
pylint examples/quantize/*.py
