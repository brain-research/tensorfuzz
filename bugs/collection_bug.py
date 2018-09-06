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
"""Minimal reproduction of mysterious get_collection issue."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf


def main(_):
    """The main function, which reproduces the issue."""
    with tf.Graph().as_default():
        var_one = tf.Variable(2)
        id_one = tf.identity(var_one)
        tf.add_to_collection("input_tensors", id_one)
        saver = tf.train.Saver(keep_checkpoint_every_n_hours=1)
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        saver.save(sess, "/tmp/bug_checkpoint", global_step=0)
        sess.close()

    with tf.Graph().as_default():
        var_two = tf.Variable(3)
        id_two = tf.identity(var_two)
        del var_two
        del id_two
        new_sess = tf.Session()
        new_sess.run(tf.global_variables_initializer())
        new_saver = tf.train.import_meta_graph("/tmp/bug_checkpoint-0.meta")
        new_saver.restore(new_sess, "/tmp/bug_checkpoint-0")
        input_tensor = tf.get_collection("input_tensors")[0]
        batch_one = new_sess.run(input_tensor)
        print(batch_one)  # I expect a 2 here, but I get 3.


if __name__ == "__main__":
    tf.app.run()
