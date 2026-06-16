from typing import Union, Optional, Sequence

import tensorflow as tf
from sonnet.src import types
from sonnet.src import utils
from sonnet.optimizers import Adam
from sonnet.src.optimizers import optimizer_utils
import sonnet as snt
from sonnet.src.optimizers.adam import adam_update


def adam_clip_update(g, alpha, beta_1, beta_2, epsilon, t, m, v, mc, vc, nu):
  """Implements 'Algorithm 1' from :cite:`kingma2014adam`."""
  # m = beta_1 * m + (1. - beta_1) * g      # Biased first moment estimate.
  # v = beta_2 * v + (1. - beta_2) * g * g  # Biased second raw moment estimate.
  # v_hat = v / (1. - tf.pow(beta_2, t))    # Bias corrected 2nd moment estimate.
  #
  # clip = nu * tf.sqrt(v_hat + epsilon)
  # gc = tf.clip_by_value(g, -clip, clip)
  #
  # mc = beta_1 * mc + (1. - beta_1) * gc
  # vc = beta_2 * vc + (1. - beta_2) * gc * gc
  # vc_hat = vc / (1. - tf.pow(beta_2, t))
  # mc_hat = mc / (1. - tf.pow(beta_1, t))    # Bias corrected 1st moment estimate.
  #
  # update = alpha * mc_hat / (tf.sqrt(vc_hat) + epsilon)
  return *adam_update(g, alpha, beta_1, beta_2, epsilon, t, m, v), mc, vc


class AdamClip(Adam):

        def __init__(self,
                     learning_rate: Union[types.FloatLike, tf.Variable] = 0.001,
                     beta1: Union[types.FloatLike, tf.Variable] = 0.9,
                     beta2: Union[types.FloatLike, tf.Variable] = 0.999,
                     epsilon: Union[types.FloatLike, tf.Variable] = 1e-8,
                     nu: Union[types.FloatLike, tf.Variable] = 5.,
                     name: Optional[str] = None):
          super().__init__(learning_rate, beta1, beta2, epsilon, name)
          self.m_clipped = []
          self.v_clipped = []
          self.nu = nu

        @snt.once
        def _initialize(self, parameters: Sequence[tf.Variable]):
            """First and second order moments are initialized to zero."""
            zero_var = lambda p: utils.variable_like(p, trainable=False)
            with tf.name_scope("m"):
                self.m.extend(zero_var(p) for p in parameters)
            with tf.name_scope("v"):
                self.v.extend(zero_var(p) for p in parameters)
            with tf.name_scope("m_clipped"):
                self.m_clipped.extend(zero_var(p) for p in parameters)
            with tf.name_scope("v_clipped"):
                self.v_clipped.extend(zero_var(p) for p in parameters)

        def apply(self, updates: Sequence[types.ParameterUpdate],
                  parameters: Sequence[tf.Variable]):

            optimizer_utils.check_distribution_strategy()
            optimizer_utils.check_updates_parameters(updates, parameters)
            self._initialize(parameters)
            self.step.assign_add(1)
            for update, param, m_var, v_var, mc_var, vc_var in zip(updates, parameters, self.m, self.v, self.m_clipped, self.v_clipped):
                if update is None:
                    continue

                optimizer_utils.check_same_dtype(update, param)
                learning_rate = tf.cast(self.learning_rate, update.dtype)
                beta_1 = tf.cast(self.beta1, update.dtype)
                beta_2 = tf.cast(self.beta2, update.dtype)
                epsilon = tf.cast(self.epsilon, update.dtype)
                step = tf.cast(self.step, update.dtype)

                if isinstance(update, tf.IndexedSlices):
                    # Sparse read our state.
                    update, indices = optimizer_utils.deduplicate_indexed_slices(update)
                    m = m_var.sparse_read(indices)
                    v = v_var.sparse_read(indices)
                    mc = mc_var.sparse_read(indices)
                    vc = vc_var.sparse_read(indices)

                    # Compute and apply a sparse update to our parameter and state.
                    update, m, v, mc, vc = adam_clip_update(
                        g=update, alpha=learning_rate, beta_1=beta_1, beta_2=beta_2,
                        epsilon=epsilon, t=step, m=m, v=v, mc=mc, vc=vc, nu=self.nu)
                    param.scatter_sub(tf.IndexedSlices(update, indices))
                    m_var.scatter_update(tf.IndexedSlices(m, indices))
                    v_var.scatter_update(tf.IndexedSlices(v, indices))
                    mc_var.scatter_update(tf.IndexedSlices(mc, indices))
                    vc_var.scatter_update(tf.IndexedSlices(vc, indices))

                else:
                    # Compute and apply a dense update to our parameter and state.
                    update, m, v, mc, vc = adam_clip_update(
                        g=update, alpha=learning_rate, beta_1=beta_1, beta_2=beta_2,
                        epsilon=epsilon, t=step, m=m_var, v=v_var, mc=mc_var, vc=vc_var, nu=self.nu)
                    param.assign_sub(update)
                    m_var.assign(m)
                    v_var.assign(v)
                    mc_var.assign(mc)
                    vc_var.assign(vc)