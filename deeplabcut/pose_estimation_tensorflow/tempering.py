"""
DeepLabCut2.0 Toolbox (deeplabcut.org)
© A. & M. Mathis Labs
https://github.com/AlexEMG/DeepLabCut

Please see AUTHORS for contributors.
https://github.com/AlexEMG/DeepLabCut/blob/master/AUTHORS
Licensed under GNU Lesser General Public License v3.0

Authored by Evan Shelhamer
during the DeepLabCut Hackathon at the Rowland Insitute on 03/06/19.

"""
import tensorflow as tf
import tensorflow.contrib.slim as slim


def tempered_sigmoid_cross_entropy(logits, targets, temperature):
    """
    Temper the sigmoid distribution by scaling the logit by temperature.
    Temperature is inversely proportion to the entropy:
    - as temperature goes up, the distribution converges to {0, 1},
      and the gradient norm increases
    - as temperature goes down, the distribution converges to 0.5,
      and the gradient norm decreases

    note: the same temperature is shared across channels/sigmoids for now,
          but each sigmoid could have its own temperature.
    """
    tempered_logits = logits / temperature
    return tf.losses.sigmoid_cross_entropy(logits=logits, labels=targets)


def tempered_softmax_cross_entropy(logits, targets, temperature):
    """
    Temper the softmax distribution by scaling the logits by temperature.
    Temperature is inversely proportion to the entropy:
    - as temperature goes up, the distribution converges to the argmax,
      and gradient norm increases
    - as temperature goes down, the distribution converges to uniform,
      and gradient norm decreases
    """
    tempered_logits = logits / temperature
    return tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=targets)


def constrain_temperature(raw_temperature, lo=0.01, ref=1.0):
    """
    Map unconstrained `raw_temperature` into (0, +Inf)
    by the affine softplus, which we found more stable than log/exp.
    (See the utilities below for other choices.)
    """
    return affine_softplus(raw_temperature, lo, ref)


def define_temperature_regressor(cfg, feature, reuse=None):
    """
    Make graph for simple temperature regressor on given feature.
    """
    rate = cfg.deconvolutionstride
    with tf.variable_scope("uncertainty"):
        with slim.arg_scope(
            [slim.conv2d, slim.conv2d_transpose],
            padding="SAME",
            activation_fn=None,
            normalizer_fn=None,
            weights_regularizer=slim.l2_regularizer(cfg.weight_decay),
        ):
            with tf.variable_scope("raw_temperature", reuse=reuse):
                raw_temperature = slim.conv2d(
                    feature, 1, kernel_size=[3, 3], stride=1, scope="block4",
                )
                # TODO(shelhamer) replace learned conv. tr.
                # with bilinear interpolation for simplicity
                raw_temperature_up = slim.conv2d_transpose(
                    raw_temperature,
                    1,
                    kernel_size=[3, 3],
                    stride=rate,
                    scope="block4",
                )
                temperature = constrain_temperature(raw_temperature_up)
                return temperature


"""
The following utilities are borrowed from Jon Barron's tensorflow impl. of his
generalized robust loss rho:
https://github.com/google-research/google-research/blob/master/robust_loss/util.py
"""


def logit(y):
    """The inverse of tf.nn.sigmoid()."""
    return -tf.math.log(1.0 / y - 1.0)


def inv_softplus(y):
    """The inverse of tf.nn.softplus()."""
    return tf.where(y > 87.5, y, tf.math.log(tf.math.expm1(y)))


def affine_sigmoid(real, lo=0, hi=1):
    """Maps reals to (lo, hi), where 0 maps to (lo+hi)/2."""
    if not lo < hi:
        raise ValueError("`lo` (%g) must be < `hi` (%g)" % (lo, hi))
    alpha = tf.sigmoid(real) * (hi - lo) + lo
    return alpha


def inv_affine_sigmoid(alpha, lo=0, hi=1):
    """The inverse of affine_sigmoid(., lo, hi)."""
    if not lo < hi:
        raise ValueError("`lo` (%g) must be < `hi` (%g)" % (lo, hi))
    real = logit((alpha - lo) / (hi - lo))
    return real


def affine_softplus(real, lo=0, ref=1):
    """Maps real numbers to (lo, infinity), where 0 maps to ref."""
    if not lo < ref:
        raise ValueError("`lo` (%g) must be < `ref` (%g)" % (lo, ref))
    shift = inv_softplus(tf.cast(1.0, real.dtype))
    scale = (ref - lo) * tf.nn.softplus(real + shift) + lo
    return scale


def inv_affine_softplus(scale, lo=0, ref=1):
    """The inverse of affine_softplus(., lo, ref)."""
    if not lo < ref:
        raise ValueError("`lo` (%g) must be < `ref` (%g)" % (lo, ref))
    shift = inv_softplus(tf.cast(1.0, scale.dtype))
    real = inv_softplus((scale - lo) / (ref - lo)) - shift
    return real
