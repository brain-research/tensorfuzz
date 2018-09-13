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
"""Defines the actual Fuzzer object."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from lib.corpus import CorpusElement
from lib.corpus import InputCorpus
from lib.corpus import seed_corpus_from_numpy_arrays
from lib.fuzz_utils import build_fetch_function
import tensorflow as tf


# pylint: disable=too-few-public-methods
# pylint: disable=too-many-arguments
class Fuzzer(object):
    """Class representing the fuzzer itself."""

    def __init__(
        self,
        sess,
        seed_inputs,
        input_tensors,
        coverage_tensors,
        metadata_tensors,
        coverage_function,
        metadata_function,
        objective_function,
        mutation_function,
        sample_function,
        threshold,
        algorithm="kdtree",
    ):
        """Init the class.

    Args:
      sess: a TF session
      seed_inputs: np arrays of initial inputs, to seed the corpus with.
      input_tensors: TF tensors to which we feed batches of input.
      coverage_tensors: TF tensors we fetch to get coverage batches.
      metadata_tensors: TF tensors we fetch to get metadata batches.
      coverage_function: a function that does coverage batches -> coverage object.
      metadata_function: a function that does metadata batches -> metadata object.
      objective_function: a function that checks if a CorpusElement satisifies
        the fuzzing objective (e.g. find a NaN, find a misclassification, etc).
      mutation_function: a function that does CorpusElement -> mutated data.
      fetch_function: grabs numpy arrays from the TF runtime using the relevant
        tensors, to produce coverage_batches and metadata_batches
    Returns:
      Initialized object.
    """
        self.coverage_function = coverage_function
        self.metadata_function = metadata_function
        self.objective_function = objective_function
        self.mutation_function = mutation_function

        # create a single fetch function (to sess.run the tensors)
        self.fetch_function = build_fetch_function(
            sess,
            input_tensors,
            coverage_tensors,
            metadata_tensors
        )

        # set up seed corpus
        seed_corpus = seed_corpus_from_numpy_arrays(
            seed_inputs,
            self.coverage_function, self.metadata_function, self.fetch_function
        )
        self.corpus = InputCorpus(
            seed_corpus, sample_function, threshold, algorithm
        )


    def loop(self, iterations):
        """Fuzzes a machine learning model in a loop, making *iterations* steps."""

        for iteration in range(iterations):
            if iteration % 100 == 0:
                tf.logging.info("fuzzing iteration: %s", iteration)
            parent = self.corpus.sample_input()

            # Get a mutated batch for each input tensor
            mutated_data_batches = self.mutation_function(parent)

            # Grab the coverage and metadata for mutated batch from the TF runtime.
            coverage_batches, metadata_batches = self.fetch_function(
                mutated_data_batches
            )

            # Get the coverage - one from each batch element
            mutated_coverage_list = self.coverage_function(coverage_batches)

            # Get the metadata objects - one from each batch element
            mutated_metadata_list = self.metadata_function(metadata_batches)

            # Check for new coverage and create new corpus elements if necessary.
            # pylint: disable=consider-using-enumerate
            for idx in range(len(mutated_coverage_list)):
                new_element = CorpusElement(
                    [batch[idx] for batch in mutated_data_batches],
                    mutated_metadata_list[idx],
                    mutated_coverage_list[idx],
                    parent,
                )
                if self.objective_function(new_element):
                    return new_element
                self.corpus.maybe_add_to_corpus(new_element)

        return None
