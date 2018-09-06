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
"""Fuzz functions from a public DCGAN implementation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
from lib.fuzz_utils import build_fetch_function
from lib.corpus import InputCorpus
from lib.corpus import seed_corpus_from_numpy_arrays
from lib.coverage_functions import raw_logit_coverage_function
from lib.fuzzer import Fuzzer
from lib.mutation_functions import do_basic_mutations
from lib.sample_functions import uniform_sample_function
from third_party.dcgan.ops import binary_cross_entropy_with_logits

tf.flags.DEFINE_integer("total_inputs_to_fuzz", 100, "Number of mutations.")
tf.flags.DEFINE_integer(
    "mutations_per_corpus_item", 64, "Number of times to mutate corpus item."
)
tf.flags.DEFINE_float(
    "ann_threshold",
    1.0,
    "Distance below which we consider something new coverage.",
)
FLAGS = tf.flags.FLAGS


def metadata_function(metadata_batches):
    """Gets the metadata."""
    loss_batch, grad_batch = metadata_batches
    metadata_list = [
        {"loss": loss_batch[idx], "grad": grad_batch[idx]}
        for idx in range(loss_batch.shape[0])
    ]
    return metadata_list


def objective_function(corpus_element):
    """Checks if the grad is bad, man."""
    loss = corpus_element.metadata["loss"]
    grad = corpus_element.metadata["grad"]
    if loss > 0.01 and abs(grad) < 0.0001:
        tf.logging.info(
            "SUCCESS: Loss %s w/ grad %s on input %s",
            loss,
            grad,
            corpus_element.data,
        )
        return True
    return False


# pylint: disable=too-many-locals
def main(_):
    """Configures and runs the fuzzer."""

    # Log more
    tf.logging.set_verbosity(tf.logging.INFO)

    coverage_function = raw_logit_coverage_function
    target_seed = np.random.uniform(low=0.0, high=1.0, size=(1,))
    numpy_arrays = [[target_seed]]

    targets_tensor = tf.placeholder(tf.float32, [64, 1])
    coverage_tensor = tf.identity(targets_tensor)
    loss_batch_tensor, _ = binary_cross_entropy_with_logits(
        tf.zeros_like(targets_tensor), tf.nn.sigmoid(targets_tensor)
    )
    grads_tensor = tf.gradients(loss_batch_tensor, targets_tensor)[0]
    tensor_map = {
        "input": [targets_tensor],
        "coverage": [coverage_tensor],
        "metadata": [loss_batch_tensor, grads_tensor],
    }

    with tf.Session() as sess:

        fetch_function = build_fetch_function(sess, tensor_map)
        size = FLAGS.mutations_per_corpus_item
        mutation_function = lambda elt: do_basic_mutations(
            elt, size, a_min=-1000, a_max=1000
        )
        seed_corpus = seed_corpus_from_numpy_arrays(
            numpy_arrays, coverage_function, metadata_function, fetch_function
        )
        corpus = InputCorpus(
            seed_corpus, uniform_sample_function, FLAGS.ann_threshold, "kdtree"
        )
        fuzzer = Fuzzer(
            corpus,
            coverage_function,
            metadata_function,
            objective_function,
            mutation_function,
            fetch_function,
        )
        result = fuzzer.loop(FLAGS.total_inputs_to_fuzz)
        if result is not None:
            tf.logging.info("Fuzzing succeeded.")
            tf.logging.info(
                "Generations to make satisfying element: %s.",
                result.oldest_ancestor()[1],
            )
        else:
            tf.logging.info("Fuzzing failed to satisfy objective function.")


if __name__ == "__main__":
    tf.app.run()
