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
"""Functions that compute neural network coverage."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np


def all_logit_coverage_function(coverage_batches):
    """Computes coverage based on the sum of the absolute values of the logits.

    Args:
        coverage_batches: Numpy arrays containing coverage information pulled from
          a call to sess.run. In this case, we assume that these correspond to a
          batch of logits.

    Returns:
        A python integer corresponding to the sum of the absolute values of the
        logits.
    """
    coverage_batch = coverage_batches[0]
    coverage_list = []
    for idx in range(coverage_batch.shape[0]):
        elt = coverage_batch[idx]
        elt = np.expand_dims(np.sum(np.abs(elt)), 0)
        coverage_list.append(elt)
    return coverage_list


def raw_logit_coverage_function(coverage_batches):
    """The coverage in this case is just the actual logits.

    This coverage function is intended for use with a nearest neighbor method.

    Args:
        coverage_batches: Numpy arrays containing coverage information pulled from
          a call to sess.run. In this case, we assume that these correspond to a
          batch of logits.

    Returns:
        A numpy array of logits.
    """
    # For our purpose, we only need the first coverage element
    coverage_batch = coverage_batches[0]
    coverage_list = []
    for idx in range(coverage_batch.shape[0]):
        elt = coverage_batch[idx]
        coverage_list.append(elt)
    return coverage_list
