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
"""Train a model and its quantized counterpart."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import lib.dataset as mnist
import tensorflow as tf


tf.flags.DEFINE_string(
    "checkpoint_dir",
    "/tmp/quantizefuzzer",
    "The overall dir in which we store experiments",
)
tf.flags.DEFINE_string(
    "data_dir", "/tmp/mnist", "The directory in which we store the MNIST data"
)
tf.flags.DEFINE_integer(
    "training_steps", 35000, "Number of mini-batch gradient updates to perform"
)

FLAGS = tf.flags.FLAGS


def weight_variable(shape):
    """Construct variable for fully connected weights."""
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)


def bias_variable(shape):
    """Construct variable for fully connected biases."""
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial)


# pylint: disable=too-many-locals
# pylint: disable=too-many-statements
def main(_):
    """Train a model and a sort-of-quantized version."""

    dataset = mnist.train(FLAGS.data_dir)
    dataset = dataset.cache().shuffle(buffer_size=50000).batch(100).repeat()
    iterator = dataset.make_one_shot_iterator()
    images, integer_labels = iterator.get_next()
    images = tf.reshape(images, [-1, 28, 28, 1])
    images = tf.identity(images)

    # Now we construct the model in kind of a goofy way, because this makes
    # quantization easier?

    # Sizes of hidden layers
    h_0 = 784
    h_1 = 200
    h_2 = 100
    h_3 = 10

    # Declaring the weight variables
    images_flattened = tf.reshape(images, [-1, h_0])

    w_fc1 = weight_variable([h_0, h_1])
    b_fc1 = bias_variable([h_1])

    w_fc2 = weight_variable([h_1, h_2])
    b_fc2 = bias_variable([h_2])

    w_fc3 = weight_variable([h_2, h_3])
    b_fc3 = bias_variable([h_3])

    # Constructing the classifier from the weight variables
    h_fc1 = tf.nn.relu(tf.matmul(images_flattened, w_fc1) + b_fc1)
    h_fc2 = tf.nn.relu(tf.matmul(h_fc1, w_fc2) + b_fc2)
    h_fc3 = tf.matmul(h_fc2, w_fc3) + b_fc3

    logits = h_fc3

    # Now I want to construct another classifier w/ quantized weights
    images_quantized = tf.cast(images_flattened, tf.float16)

    w_fc1_quantized = tf.cast(w_fc1, tf.float16)
    b_fc1_quantized = tf.cast(b_fc1, tf.float16)

    w_fc2_quantized = tf.cast(w_fc2, tf.float16)
    b_fc2_quantized = tf.cast(b_fc2, tf.float16)

    w_fc3_quantized = tf.cast(w_fc3, tf.float16)
    b_fc3_quantized = tf.cast(b_fc3, tf.float16)

    # Constructing the classifier from the weight variables
    h_fc1_quantized = tf.nn.relu(
        tf.matmul(images_quantized, w_fc1_quantized) + b_fc1_quantized
    )
    h_fc2_quantized = tf.nn.relu(
        tf.matmul(h_fc1_quantized, w_fc2_quantized) + b_fc2_quantized
    )
    h_fc3_quantized = (
        tf.matmul(h_fc2_quantized, w_fc3_quantized) + b_fc3_quantized
    )

    logits_quantized = h_fc3_quantized

    labels = tf.one_hot(integer_labels, 10)
    equality = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.to_float(equality))

    cross_entropies = tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=labels
    )
    loss = tf.reduce_mean(cross_entropies)
    optimizer = tf.train.GradientDescentOptimizer(0.01)

    tf.add_to_collection("input_tensors", images)
    tf.add_to_collection("coverage_tensors", logits)
    tf.add_to_collection("coverage_tensors", logits_quantized)
    tf.add_to_collection("metadata_tensors", logits)
    tf.add_to_collection("metadata_tensors", logits_quantized)

    train_op = optimizer.minimize(loss)

    saver = tf.train.Saver(keep_checkpoint_every_n_hours=1)
    sess = tf.Session()
    sess.run(tf.initialize_all_tables())
    sess.run(tf.global_variables_initializer())

    # train classifier on these images and labels
    for idx in range(FLAGS.training_steps):
        sess.run(train_op)
        if idx % 100 == 0:
            loss_val, accuracy_val = sess.run([loss, accuracy])
            print("loss: {}, accuracy: {}".format(loss_val, accuracy_val))
            saver.save(
                sess,
                os.path.join(FLAGS.checkpoint_dir, "fuzz_checkpoint"),
                global_step=idx,
            )


if __name__ == "__main__":
    tf.app.run()
