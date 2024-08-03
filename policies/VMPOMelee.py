import time

import tree
from melee import Action
from polaris.policies.VMPO import VMPO

from polaris.policies.parametrised import ParametrisedPolicy
from polaris.experience import SampleBatch, get_epochs
from polaris.models.utils import EpsilonCategorical


import numpy as np
import tensorflow as tf

from polaris.policies.utils.return_based_scaling import ReturnBasedScaling
from polaris.policies.utils.popart import Popart

tf.compat.v1.enable_eager_execution()

from polaris.policies.utils.misc import explained_variance
from polaris.policies.utils.vtrace import compute_vtrace





def get_annealed_softmax_advantages(top_half_advantages, temperature, top_half_clipped_rhos):
    exp_adv = tf.exp(tf.minimum(top_half_advantages / temperature, 20.)) * top_half_clipped_rhos
    return tf.stop_gradient(
        exp_adv / tf.reduce_sum(exp_adv)
    )

def compute_temperature_loss(temperature, temperature_eps, temperature_kl):
    """

    :param temperature:  temperature
    :param temperature_eps:  eps_temperature
    :param temperature_kl:  temperature_kl
    :return: temperature loss

    """
    return temperature * tf.stop_gradient(temperature_eps - temperature_kl)

def compute_trust_region_loss(trust_region_coeff, trust_region_eps, kl_offline_to_online):
    """
    Whole batch of data should be used here
    :param trust_region_coeff: trust_region_coeff
    :param trust_region_eps:  trust_region_eps
    :param kl_offline_to_online:  kl divergence between online version and behaviour distribution
    :return: trust region loss
    """
    return tf.reduce_mean(trust_region_coeff * (trust_region_eps - tf.stop_gradient(kl_offline_to_online)) +
                          tf.stop_gradient(trust_region_coeff) * kl_offline_to_online)

def compute_policy_loss(top_half_online_logp, annealed_top_half_exp_advantage):
    #psi = tf.stop_gradient(annealed_top_half_exp_advantage / tf.reduce_sum(annealed_top_half_exp_advantage))
    # seedrl google says mean... , but we do sum
    return - tf.reduce_sum(top_half_online_logp * annealed_top_half_exp_advantage)

class VMPOMelee(VMPO):
    # TODO: take care of the loss, tophalf adv, not much more i think


    def init_model(self):
        super().init_model()

        if self.offline_policy is not None:
            self.offline_policy.init_model()
            # set weights to offline policy
            self.update_offline_model()
        else:
            # TODO: add epsilon param, add vmpo config toggle
            self.model.action_dist = EpsilonCategorical


