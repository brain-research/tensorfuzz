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
"""Utilities for the fuzzer library."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import random as random
import numpy as np
import scipy
import tensorflow as tf
import lib.dataset as mnist


def basic_mnist_input_corpus(choose_randomly=False, data_dir="/tmp/mnist"):
    """Returns the first image and label from MNIST.

    Args:
      choose_randomly: a boolean indicating whether to choose randomly.
      data_dir: a string giving the location of the original MNIST data.
    Returns:
      A single image and a single label.
    """

    dataset = mnist.train(data_dir)
    dataset = dataset.cache().shuffle(buffer_size=50000).batch(100).repeat()
    iterator = dataset.make_one_shot_iterator()
    images, integer_labels = iterator.get_next()
    images = tf.reshape(images, [-1, 28, 28, 1])
    # labels = tf.one_hot(integer_labels, 10)
    labels = integer_labels

    with tf.train.MonitoredTrainingSession() as sess:
        image_batch, label_batch = sess.run([images, labels])

    if choose_randomly:
        idx = random.choice(range(image_batch.shape[0]))
    else:
        idx = 0
    tf.logging.info("Seeding corpus with element at idx: %s", idx)
    return image_batch[idx], label_batch[idx]


def imsave(image, path):
    """Saves an image to a given path.

    This function has the side-effect of writing to disk.

    Args:
        image: The Numpy array representing the image.
        path: A Filepath.
    """
    image = np.squeeze(image)
    with tf.gfile.Open(path, mode="w") as fptr:
        scipy.misc.imsave(fptr, image)


def build_feed_dict(input_tensors, input_batches):
    """Constructs a feed_dict to pass to the run method of TensorFlow Session.

    In the logic we assume all tensors should have the same batch size.
    However, we have to do some crazy stuff to deal with the case when
    some of the tensors have concrete shapes and some don't, especially
    when we're constructing the seed corpus.

    Args:
        input_tensors: The TF tensors into which we will feed the fuzzed inputs.
        input_batches: Numpy arrays that will be fed into the input tensors.

    Returns:
        The feed_dict described above.
    """
    feed_dict = {}

    # If the tensor has concrete shape and we are feeding in something that has a
    # non-matching shape, we will need to tile it to make it work.
    tensor_bszs = [x.get_shape().as_list()[0] for x in input_tensors]
    should_tile = any([x is not None for x in tensor_bszs])
    if should_tile:
        max_tensor_bsz = max([x for x in tensor_bszs if x is not None])
    for idx in range(len(list(zip(input_tensors, input_batches)))):
        np_bsz = input_batches[idx].shape[0]
        if should_tile and np_bsz != max_tensor_bsz:
            tf.logging.info(
                "Tiling feed_dict inputs due to concrete batch sizes."
            )
            this_shape = [max_tensor_bsz // np_bsz] + [
                1 for _ in range(len(input_batches[idx].shape[1:]))
            ]
            input_batches[idx] = np.tile(input_batches[idx], this_shape)

    # Note that this will truncate one of input_tensors or input_batches
    # if either of them is longer. This is WAI right now, because we sometimes
    # want to store the label for an image classifier for which we don't have
    # a label placeholder in the checkpoint.
    for input_tensor, input_batch in list(zip(input_tensors, input_batches)):
        feed_dict[input_tensor] = input_batch
    return feed_dict


def get_tensors_from_checkpoint(sess, checkpoint_dir):
    """Loads and returns the fuzzing tensors given a session and a directory.

    It's assumed that the checkpoint directory has checkpoints from a TensorFlow
    model, and moreoever that those checkpoints have 3 collections:
    1. input_tensors: The tensors into which we will feed the fuzzed inputs.
    2. coverage_tensors: The tensors from which we will fetch information needed
      to compute the coverage. The coverage will be used to guide the fuzzing
      process.
    3. metadata_tensors: The tensors from which we will fetch information needed
      to compute the metadata. The metadata can be used for computing the fuzzing
      objective or just to track the progress of fuzzing.

    Args:
      sess: a TensorFlow Session object.
      checkpoint_dir: a directory containing the TensorFlow checkpoints.

    Returns:
        The 3 lists of tensorflow tensors described above.
    """
    potential_files = tf.gfile.ListDirectory(checkpoint_dir)
    meta_files = [f for f in potential_files if f.endswith(".meta")]

    # Sort the meta files by global step
    meta_files.sort(key=lambda f: int(f[: -len(".meta")].split("-")[-1]))
    meta_file = meta_files[-1]

    explicit_meta_path = os.path.join(checkpoint_dir, meta_file)
    explicit_checkpoint_path = explicit_meta_path[: -len(".meta")]
    tf.logging.info("Visualizing checkpoint: %s", explicit_checkpoint_path)

    new_saver = tf.train.import_meta_graph(
        explicit_meta_path, clear_devices=True
    )
    new_saver.restore(sess, explicit_checkpoint_path)

    input_tensors = tf.get_collection("input_tensors")
    coverage_tensors = tf.get_collection("coverage_tensors")
    metadata_tensors = tf.get_collection("metadata_tensors")

    return input_tensors, coverage_tensors, metadata_tensors


def fetch_function(
    sess, input_tensors, coverage_tensors, metadata_tensors, input_batches
):
    """Fetches from the TensorFlow runtime given inputs.

    Args:
      sess: a TensorFlow Session object.
      input_tensors: TF tensors to which we feed input_batches.
      coverage_tensors: TF tensors we fetch for coverage.
      metadata_tensors: TF tensors we fetch for metadata.
      input_batches: numpy arrays we feed to input_tensors.

    Returns:
        Coverage and metadata as lists of numpy arrays.
    """
    feed_dict = build_feed_dict(input_tensors, input_batches)
    fetched_data = sess.run(
        coverage_tensors + metadata_tensors, feed_dict=feed_dict
    )
    idx = len(coverage_tensors)
    coverage_batches = fetched_data[:idx]
    metadata_batches = fetched_data[idx:]
    return coverage_batches, metadata_batches


def build_fetch_function(sess, input_tensors, coverage_tensors, metadata_tensors):
    """Constructs fetch function given session and tensors."""

    def func(input_batches):
        """The fetch function."""
        return fetch_function(
            sess,
            input_tensors,
            coverage_tensors,
            metadata_tensors,
            input_batches,
        )

    return func
