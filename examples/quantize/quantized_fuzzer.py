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
"""Fuzz a neural network to find disagreements between normal and quantized."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
from lib import fuzz_utils
from lib.coverage_functions import sum_coverage_function
from lib.fuzzer import Fuzzer
from lib.mutation_functions import do_basic_mutations
from lib.sample_functions import recent_sample_function


tf.flags.DEFINE_string(
    "checkpoint_dir", None, "Dir containing checkpoints of model to fuzz."
)
tf.flags.DEFINE_string(
    "output_path", None, "Where to write the satisfying output."
)
tf.flags.DEFINE_integer(
    "total_inputs_to_fuzz", 100, "Loops over the whole corpus."
)
tf.flags.DEFINE_integer(
    "mutations_per_corpus_item", 100, "Number of times to mutate corpus item."
)
tf.flags.DEFINE_float(
    "perturbation_constraint", None, "Constraint on norm of perturbations."
)
tf.flags.DEFINE_float(
    "ann_threshold",
    1.0,
    "Distance below which we consider something new coverage.",
)
tf.flags.DEFINE_boolean(
    "random_seed_corpus", False, "Whether to choose a random seed corpus."
)
FLAGS = tf.flags.FLAGS


if FLAGS.checkpoint_dir is None:
    raise ValueError('checkpoint_dir flag must be specified')



def metadata_function(metadata_batches):
    """Gets the metadata, for computing the objective function."""
    logit_32_batch = metadata_batches[0]
    logit_16_batch = metadata_batches[1]
    metadata_list = []
    for idx in range(logit_16_batch.shape[0]):
        metadata_list.append({
            "logits_32": logit_32_batch[idx],
            "logits_16": logit_16_batch[idx]})
    return metadata_list


def objective_function(corpus_element):
    """Checks if the element is misclassified."""
    logits_32 = corpus_element.metadata["logits_32"]
    logits_16 = corpus_element.metadata["logits_16"]
    prediction_16 = np.argmax(logits_16)
    prediction_32 = np.argmax(logits_32)
    if prediction_16 == prediction_32:
        return False

    tf.logging.info(
        "Objective function satisfied: 32: %s, 16: %s",
        prediction_32,
        prediction_16,
    )
    return True


def mutation_function(elt):
    """Mutates the element in question."""
    return do_basic_mutations(
        elt, FLAGS.mutations_per_corpus_item, FLAGS.perturbation_constraint)


# pylint: disable=too-many-locals
def main(_):
    """Constructs the fuzzer and fuzzes."""

    # Log more
    tf.logging.set_verbosity(tf.logging.INFO)

    # Set up initial seed inputs
    image, label = fuzz_utils.basic_mnist_input_corpus(
        choose_randomly=FLAGS.random_seed_corpus
    )
    seed_inputs = [[image, label]]
    image_copy = image[:]

    with tf.Session() as sess:
        # Specify input, coverage, and metadata tensors
        input_tensors, coverage_tensors, metadata_tensors = \
          fuzz_utils.get_tensors_from_checkpoint(
              sess, FLAGS.checkpoint_dir
          )

        # Construct and run fuzzer
        fuzzer = Fuzzer(
            sess=sess,
            seed_inputs=seed_inputs,
            input_tensors=input_tensors,
            coverage_tensors=coverage_tensors,
            metadata_tensors=metadata_tensors,
            coverage_function=sum_coverage_function,
            metadata_function=metadata_function,
            objective_function=objective_function,
            mutation_function=mutation_function,
            sample_function=recent_sample_function,
            threshold=FLAGS.ann_threshold,
        )
        result = fuzzer.loop(FLAGS.total_inputs_to_fuzz)

        if result is not None:
            # Double check that there is persistent disagreement
            for idx in range(10):
                logits, quantized_logits = sess.run(
                    [coverage_tensors[0], coverage_tensors[1]],
                    feed_dict={
                        input_tensors[0]: np.expand_dims(
                            result.data[0], 0
                        )
                    },
                )
                if np.argmax(logits, 1) != np.argmax(quantized_logits, 1):
                    tf.logging.info("disagreement confirmed: idx %s", idx)
                else:
                    tf.logging.info("SPURIOUS DISAGREEMENT!!!")
            tf.logging.info("Fuzzing succeeded.")
            tf.logging.info(
                "Generations to make satisfying element: %s.",
                result.oldest_ancestor()[1],
            )
            max_diff = np.max(result.data[0] - image_copy)
            tf.logging.info(
                "Max difference between perturbation and original: %s.",
                max_diff,
            )
        else:
            tf.logging.info("Fuzzing failed to satisfy objective function.")


if __name__ == "__main__":
    tf.app.run()
