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


def sum_coverage_function(coverage_batches):
    """Computes coverage as the sum of the absolute values of coverage_batches.

    Args:
        coverage_batches: Numpy arrays containing coverage information pulled from
          a call to sess.run.

    Returns:
        A list of python integers corresponding to the sum of the absolute
        values of the entries in coverage_batches.
    """
    coverage_batch = coverage_batches[0]
    coverage_list = []
    for idx in range(coverage_batch.shape[0]):
        elt = coverage_batch[idx]
        elt = np.expand_dims(np.sum(np.abs(elt)), 0)
        coverage_list.append(elt)
    return coverage_list


def raw_coverage_function(coverage_batches):
    """The coverage in this case is just the actual values of coverage_batches.

    This coverage function is intended for use with a nearest neighbor method.

    Args:
        coverage_batches: Numpy arrays containing coverage information pulled from
          a call to sess.run.

    Returns:
        A list of numpy arrays corresponding to the entries in coverage_batches.
    """
    # For our purpose, we only need the first coverage element
    coverage_batch = coverage_batches[0]
    coverage_list = []
    for idx in range(coverage_batch.shape[0]):
        elt = coverage_batch[idx]
        coverage_list.append(elt)
    return coverage_list
