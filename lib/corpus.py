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
"""Defines a set of objects that together describe the fuzzing input corpus."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import numpy as np
import tensorflow as tf
import pyflann

_BUFFER_SIZE = 50

# pylint: disable=too-few-public-methods
class CorpusElement(object):
    """Class representing a single element of a corpus."""

    def __init__(self, data, metadata, coverage, parent):
        """Inits the object.

        Args:
          data: a list of numpy arrays representing the mutated data.
          metadata: arbitrary python object to be used by the fuzzer for
            computing the objective function during the fuzzing loop.
          coverage: an arbitrary hashable python object that guides fuzzing process.
          parent: a reference to the CorpusElement this element is a mutation of.
          iteration: the fuzzing iteration (number of CorpusElements sampled to
            mutate) that this CorpusElement was created at.
        Returns:
          Initialized object.
        """
        self.data = data
        self.metadata = metadata
        self.parent = parent
        self.coverage = coverage

    def oldest_ancestor(self):
        """Returns the least recently created ancestor of this corpus item."""
        current_element = self
        generations = 0
        while current_element.parent is not None:
            current_element = current_element.parent
            generations += 1
        return current_element, generations


def seed_corpus_from_numpy_arrays(
    numpy_arrays, coverage_function, metadata_function, fetch_function
):
    """Constructs a starting corpus, given numpy_arrays of inputs,
    by passing them through the system to produce outputs (i.e. coverage
    representations)

    We only use the first element of the batch that we fetch, because
    we're only trying to create one corpus element, and we may end up
    getting back a whole batch of coverage due to the need to tile our
    inputs to fit the static shape of certain feed_dicts.
    Args:
      numpy_arrays: multiple lists of input_arrays, each list with as many
        arrays as there are input tensors.
      coverage_function: a function that does coverage batches -> coverage object.
      metadata_function: a function that does metadata batches -> metadata object.
      fetch_function: grabs output from tensorflow runtime.
    Returns:
      List of CorpusElements.
    """
    seed_corpus = []
    for input_array_list in numpy_arrays:
        input_batches = []
        for input_array in input_array_list:
            input_batches.append(np.expand_dims(input_array, axis=0))
        coverage_batches, metadata_batches = fetch_function(input_batches)
        coverage_list = coverage_function(coverage_batches)
        metadata_list = metadata_function(metadata_batches)
        new_element = CorpusElement(
            input_array_list, metadata_list[0], coverage_list[0], None
        )
        seed_corpus.append(new_element)
    return seed_corpus


class Updater(object):
    """Class holding the state of the update function."""

    def __init__(self, threshold, algorithm):
        """Inits the object.

        Args:
          threshold: Float distance at which coverage is considered new.
          algorithm: Algorithm used to get approximate neighbors.
        Returns:
          Initialized object.
        """
        self.flann = pyflann.FLANN()
        self.threshold = threshold
        self.algorithm = algorithm
        self.corpus_buffer = []
        self.lookup_array = []

    def build_index_and_flush_buffer(self, corpus_object):
        """Builds the nearest neighbor index and flushes buffer of examples.

        This method first empties the buffer of examples that have not yet
        been added to the nearest neighbor index.
        Then it rebuilds that index using the contents of the whole corpus.
        Args:
          corpus_object: InputCorpus object.
        """
        self.corpus_buffer[:] = []
        self.lookup_array = np.array(
            [element.coverage for element in corpus_object.corpus]
        )
        self.flann.build_index(self.lookup_array, algorithm=self.algorithm)
        tf.logging.info("Flushing buffer and building index.")

    def update_function(self, corpus_object, element):
        """Checks if coverage is new and updates corpus if so.

        The updater maintains both a corpus_buffer and a lookup_array.
        When the corpus_buffer reaches a certain size, we empty it out
        and rebuild the nearest neighbor index.
        Whenever we check for neighbors, we get exact neighbors from the
        buffer and approximate neighbors from the index.
        This stops us from building the index too frequently.
        FLANN supports incremental additions to the index, but they require
        periodic rebalancing anyway, and so far this method seems to be
        working OK.
        Args:
          corpus_object: InputCorpus object.
          element: CorpusElement object to maybe be added to the corpus.
        """
        if corpus_object.corpus is None:
            corpus_object.corpus = [element]
            self.build_index_and_flush_buffer(corpus_object)
        else:
            _, approx_distances = self.flann.nn_index(
                np.array([element.coverage]), 1, algorithm=self.algorithm
            )
            exact_distances = [
                np.sum(np.square(element.coverage - buffer_elt))
                for buffer_elt in self.corpus_buffer
            ]
            nearest_distance = min(exact_distances + approx_distances.tolist())
            if nearest_distance > self.threshold:
                tf.logging.info(
                    "corpus_size %s mutations_processed %s",
                    len(corpus_object.corpus),
                    corpus_object.mutations_processed,
                )
                tf.logging.info(
                    "coverage: %s, metadata: %s",
                    element.coverage,
                    element.metadata,
                )
                corpus_object.corpus.append(element)
                self.corpus_buffer.append(element.coverage)
                if len(self.corpus_buffer) >= _BUFFER_SIZE:
                    self.build_index_and_flush_buffer(corpus_object)


class InputCorpus(object):
    """Class that holds inputs and associated coverage."""

    def __init__(self, seed_corpus, sample_function, threshold, algorithm):
        """Init the class.

        Args:
          seed_corpus: a list of numpy arrays, one for each input tensor in the
            fuzzing process.
          sample_function: a function that looks at the whole current corpus and
            samples the next element to mutate in the fuzzing loop.
        Returns:
          Initialized object.
        """
        self.mutations_processed = 0
        self.corpus = None
        self.sample_function = sample_function
        self.start_time = time.time()
        self.current_time = time.time()
        self.log_time = time.time()
        self.updater = Updater(threshold, algorithm)

        for corpus_element in seed_corpus:
            self.maybe_add_to_corpus(corpus_element)

    def maybe_add_to_corpus(self, element):
        """Adds item to corpus if it exercises new coverage."""
        self.updater.update_function(self, element)
        self.mutations_processed += 1
        current_time = time.time()
        if current_time - self.log_time > 10:
            self.log_time = current_time
            tf.logging.info(
                "mutations_per_second: %s",
                float(self.mutations_processed)
                / (current_time - self.start_time),
            )

    def sample_input(self):
        """Grabs new input from corpus according to sample_function."""
        choice = self.sample_function(self)
        return choice
