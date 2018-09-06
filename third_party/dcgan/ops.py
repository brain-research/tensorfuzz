import tensorflow as tf
from tensorflow.python.framework import ops


def binary_cross_entropy_with_logits(logits, targets, name=None):
    """Computes binary cross entropy given `logits`.

    For brevity, let `x = logits`, `z = targets`.  The logistic loss is

        loss(x, z) = - sum_i (x[i] * log(z[i]) + (1 - x[i]) * log(1 - z[i]))

    Args:
        logits: A `Tensor` of type `float32` or `float64`.
        targets: A `Tensor` of the same type and shape as `logits`.
    """
    eps = 1e-12
    with ops.op_scope([logits, targets], name, "bce_loss") as name:
        logits = ops.convert_to_tensor(logits, name="logits")
        targets = ops.convert_to_tensor(targets, name="targets")
        loss_batch = -(
            logits * tf.log(targets + eps) + (1. - logits) * tf.log(1. - targets + eps)
        )
        loss_mean = tf.reduce_mean(loss_batch)
        return loss_batch, loss_mean
