# Copyright 2018 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Functions for choosing the next input to fuzz."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import random


def uniform_sample_function(input_corpus):
    """Samples uniformly from all the elements in the input corpus.

  Args:
    input_corpus: an InputCorpus object.

  Returns:
    A CorpusElement object.
  """
    corpus = input_corpus.corpus
    choice = random.choice(corpus)
    return choice


def recent_sample_function(input_corpus):
    """Samples from the corpus with a bias toward more recently created inputs.

  Args:
    input_corpus: an InputCorpus object.

  Returns:
    A CorpusElement object.
  """
    corpus = input_corpus.corpus
    reservoir = corpus[-5:] + [random.choice(corpus)]
    choice = random.choice(reservoir)
    return choice
