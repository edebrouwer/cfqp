import pytorch_lightning as pl
import sys
from condgen.utils import DATA_DIR
#from causalode.datagen import cancer_simulation
import condgen.utils
from condgen.utils import str2bool
import torch
from torch.utils.data import Dataset, DataLoader, Subset, TensorDataset
import os
import argparse
import numpy as np
from scipy.integrate import odeint
import pandas as pd


"""
CODE ADAPTED FROM: https://github.com/sjblim/rmsn_nips_2018

[Treatment Effects with RNNs] cancer_simulation
Created on 2/4/2018 8:14 AM

Medically realistic data simulation for small-cell lung cancer based on Geng et al 2017.
URL: https://www.nature.com/articles/s41598-017-13646-z

Notes:
- Simulation time taken to be in days

"""

import logging
import numpy as np
import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import truncnorm  # we need to sample from truncated normal distributions


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Simulation Constants

# Spherical calculations - tumours assumed to be spherical per Winer-Muram et al 2002.
# URL: https://pubs.rsna.org/doi/10.1148/radiol.2233011026?url_ver=Z39.88-2003&rfr_id=ori%3Arid%3Acrossref.org&rfr_dat=cr_pub%3Dpubmed
def calc_volume(diameter):
    return 4.0 / 3.0 * np.pi * (diameter / 2.0) ** 3.0


def calc_diameter(volume):
    return ((volume / (4.0 / 3.0 * np.pi)) ** (1.0 / 3.0)) * 2.0


# Tumour constants per
tumour_cell_density = 5.8 * 10.0 ** 8.0  # cells per cm^3
tumour_death_threshold = calc_volume(13)  # assume spherical

# Patient cancer stage. (mu, sigma, lower bound, upper bound) - for lognormal dist
tumour_size_distributions = {'I': (1.72, 4.70, 4.0, 8.0),
                             'II': (1.96, 1.63, 4.0, 13.0),
                             'IIIA': (1.91, 9.40, 4.0, 13.0),
                             'IIIB': (2.76, 6.87, 4.0, 13.0),
                             'IV': (3.86, 8.82, 4.0, 13.0)}  # 13.0 is the death condition

# Observations of stage proportions taken from Detterbeck and Gibson 2008
# - URL: http://www.jto.org/article/S1556-0864(15)33353-0/fulltext#cesec50\
cancer_stage_observations = {'I': 1432,
                             "II": 128,
                             "IIIA": 1306,
                             "IIIB": 7248,
                             "IV": 12840}


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Simulation Functions

def get_confounding_params(num_patients, chemo_coeff, radio_coeff):
    """

    Get original simulation parameters, and add extra ones to control confounding

    :param num_patients:
    :param chemo_coeff: Bias on action policy for chemotherapy assignments
    :param radio_activation_group: Bias on action policy for chemotherapy assignments
    :return:
    """

    basic_params = get_standard_params(num_patients)
    patient_types = basic_params['patient_types']
    tumour_stage_centres = [s for s in cancer_stage_observations if 'IIIA' not in s]
    tumour_stage_centres.sort()

    d_max = calc_diameter(tumour_death_threshold)
    basic_params['chemo_sigmoid_intercepts'] = np.array([d_max / 1 for i in patient_types])
    basic_params['radio_sigmoid_intercepts'] = np.array([d_max / 1 for i in patient_types])

    basic_params['chemo_sigmoid_betas'] = np.array([chemo_coeff / d_max for i in patient_types])
    basic_params['radio_sigmoid_betas'] = np.array([radio_coeff / d_max for i in patient_types])

    return basic_params


def get_standard_params(num_patients):  # additional params
    """
    Simulation parameters from the Nature article + adjustments for static variables

    :param num_patients:
    :return: simulation_parameters
    """

    # Adjustments for static variables
    possible_patient_types = [1, 2, 3]
    patient_types = np.random.choice(possible_patient_types,
                                     num_patients)
    chemo_mean_adjustments = np.array([0.0 if i < 3 else 0.1 for i in patient_types])
    radio_mean_adjustments = np.array([0.0 if i > 1 else 0.1 for i in patient_types])

    total = 0
    for k in cancer_stage_observations:
        total += cancer_stage_observations[k]
    cancer_stage_proportions = {k: float(cancer_stage_observations[k]) / float(total) for k in
                                cancer_stage_observations}

    # remove possible entries
    possible_stages = list(tumour_size_distributions.keys())
    possible_stages.sort()

    initial_stages = np.random.choice(possible_stages, num_patients,
                                      p=[cancer_stage_proportions[k] for k in possible_stages])

    # Get info on patient stages and initial volumes
    output_initial_diam = []
    patient_sim_stages = []
    for stg in possible_stages:
        count = np.sum((initial_stages == stg) * 1)

        mu, sigma, lower_bound, upper_bound = tumour_size_distributions[stg]

        # Convert lognorm bounds in to standard normal bounds
        lower_bound = (np.log(lower_bound) - mu) / sigma
        upper_bound = (np.log(upper_bound) - mu) / sigma

        logging.info(("Simulating initial volumes for stage {} " +
                      " with norm params: mu={}, sigma={}, lb={}, ub={}").format(
            stg,
            mu,
            sigma,
            lower_bound,
            upper_bound))

        norm_rvs = truncnorm.rvs(lower_bound, upper_bound,
                                 size=count)  # truncated normal for realistic clinical outcome

        initial_volume_by_stage = np.exp((norm_rvs * sigma) + mu)
        output_initial_diam += list(initial_volume_by_stage)
        patient_sim_stages += [stg for i in range(count)]

    # Fixed params
    K = calc_volume(30)  # carrying capacity given in cm, so convert to volume
    alpha_beta_ratio = 10
    alpha_rho_corr = 0.87

    # Distributional parameters for dynamics
    parameter_lower_bound = 0.0
    parameter_upper_bound = np.inf
    rho_params = (7 * 10 ** -5, 7.23 * 10 ** -3)
    alpha_params = (0.0398, 0.168)
    beta_c_params = (0.028, 0.0007)

    # Get correlated simulation paramters (alpha, beta, rho) which respects bounds
    alpha_rho_cov = np.array([[alpha_params[1] ** 2,
                               alpha_rho_corr * alpha_params[1] * rho_params[1]],
                              [alpha_rho_corr * alpha_params[1] * rho_params[1],
                               rho_params[1] ** 2]])

    alpha_rho_mean = np.array([alpha_params[0], rho_params[0]])

    simulated_params = []

    while len(simulated_params) < num_patients:  # Keep on simulating till we get the right number of params

        param_holder = np.random.multivariate_normal(alpha_rho_mean, alpha_rho_cov, size=num_patients)

        for i in range(param_holder.shape[0]):

            # Ensure that all params fulfill conditions
            if param_holder[i, 0] > parameter_lower_bound and param_holder[i, 1] > parameter_lower_bound:
                simulated_params.append(param_holder[i, :])

        logging.info("Got correlated params for {} patients".format(len(simulated_params)))

    simulated_params = np.array(simulated_params)[:num_patients, :]  # shorten this back to normal
    alpha_adjustments = alpha_params[0] * radio_mean_adjustments
    alpha = simulated_params[:, 0] + alpha_adjustments
    rho = simulated_params[:, 1]
    beta = alpha / alpha_beta_ratio

    # Get the remaining indep params
    logging.info("Simulating beta c parameters")
    beta_c_adjustments = beta_c_params[0] * chemo_mean_adjustments
    beta_c = beta_c_params[0] + beta_c_params[1] * truncnorm.rvs(
        (parameter_lower_bound - beta_c_params[0]) / beta_c_params[1],
        (parameter_upper_bound - beta_c_params[0]) / beta_c_params[1],
        size=num_patients) + beta_c_adjustments

    output_holder = {'patient_types': patient_types,
                     'initial_stages': np.array(patient_sim_stages),
                     'initial_volumes': calc_volume(np.array(output_initial_diam)),  # assumed spherical with diam
                     'alpha': alpha,
                     'rho': rho,
                     'beta': beta,
                     'beta_c': beta_c,
                     'K': np.array([K for i in range(num_patients)]),
                     }
    # np.random.exponential(expected_treatment_delay, num_patients),

    # Randomise output params
    logging.info("Randomising outputs")
    idx = [i for i in range(num_patients)]
    np.random.shuffle(idx)

    output_params = {}
    for k in output_holder:
        output_params[k] = output_holder[k][idx]

    return output_params


def simulate(simulation_params, num_time_steps, assigned_actions=None):
    """
    Core routine to generate simulation paths

    :param simulation_params:
    :param num_time_steps:
    :param assigned_actions:
    :return:
    """

    total_num_radio_treatments = 1
    total_num_chemo_treatments = 1

    radio_amt = np.array([2.0 for i in range(total_num_radio_treatments)])  # Gy
    radio_days = np.array([i + 1 for i in range(total_num_radio_treatments)])
    chemo_amt = [5.0 for i in range(total_num_chemo_treatments)]
    chemo_days = [(i + 1) * 7 for i in range(total_num_chemo_treatments)]

    # sort this
    chemo_idx = np.argsort(chemo_days)
    chemo_amt = np.array(chemo_amt)[chemo_idx]
    chemo_days = np.array(chemo_days)[chemo_idx]

    drug_half_life = 1  # one day half life for drugs

    # Unpack simulation parameters
    initial_stages = simulation_params['initial_stages']
    initial_volumes = simulation_params['initial_volumes']
    alphas = simulation_params['alpha']
    rhos = simulation_params['rho']
    betas = simulation_params['beta']
    beta_cs = simulation_params['beta_c']
    Ks = simulation_params['K']
    patient_types = simulation_params['patient_types']
    window_size = simulation_params['window_size']  # controls the lookback of the treatment assignment policy

    # Coefficients for treatment assignment probabilities
    chemo_sigmoid_intercepts = simulation_params['chemo_sigmoid_intercepts']
    radio_sigmoid_intercepts = simulation_params['radio_sigmoid_intercepts']
    chemo_sigmoid_betas = simulation_params['chemo_sigmoid_betas']
    radio_sigmoid_betas = simulation_params['radio_sigmoid_betas']

    num_patients = initial_stages.shape[0]

    # Commence Simulation
    cancer_volume = np.zeros((num_patients, num_time_steps))
    chemo_dosage = np.zeros((num_patients, num_time_steps))
    radio_dosage = np.zeros((num_patients, num_time_steps))
    chemo_application_point = np.zeros((num_patients, num_time_steps))
    radio_application_point = np.zeros((num_patients, num_time_steps))
    sequence_lengths = np.zeros(num_patients)
    death_flags = np.zeros((num_patients, num_time_steps))
    recovery_flags = np.zeros((num_patients, num_time_steps))
    chemo_probabilities = np.zeros((num_patients, num_time_steps))
    radio_probabilities = np.zeros((num_patients, num_time_steps))

    noise_terms = 0.01 * np.random.randn(num_patients,
                                         num_time_steps)  # 5% cell variability
    recovery_rvs = np.random.rand(num_patients, num_time_steps)

    chemo_application_rvs = np.random.rand(num_patients, num_time_steps)
    radio_application_rvs = np.random.rand(num_patients, num_time_steps)

    # Run actual simulation
    for i in range(num_patients):
        if (i%200 == 0):
            logging.info("Simulating patient {} of {}".format(i, num_patients))
        noise = noise_terms[i]

        # initial values
        cancer_volume[i, 0] = initial_volumes[i]
        alpha = alphas[i]
        beta = betas[i]
        beta_c = beta_cs[i]
        rho = rhos[i]
        K = Ks[i]

        # Setup cell volume
        b_death = False
        b_recover = False
        for t in range(0, num_time_steps - 1):

            current_chemo_dose = 0.0
            previous_chemo_dose = 0.0 if t == 0 else chemo_dosage[i, t - 1]

            # Action probabilities + death or recovery simulations
            cancer_volume_used = cancer_volume[i, max(t - window_size, 0):t + 1]
            cancer_diameter_used = np.array(
                [calc_diameter(vol) for vol in cancer_volume_used]).mean()  # mean diameter over 15 days
            cancer_metric_used = cancer_diameter_used

            # probabilities
            if assigned_actions is not None:
                chemo_prob = assigned_actions[i, t, 0]
                radio_prob = assigned_actions[i, t, 1]
            else:

                radio_prob = (1.0 / (1.0 + np.exp(- radio_sigmoid_betas[i]
                                                  * (cancer_metric_used - radio_sigmoid_intercepts[i]))))
                chemo_prob = (1.0 / (1.0 + np.exp(- chemo_sigmoid_betas[i] *
                                                  (cancer_metric_used - chemo_sigmoid_intercepts[i]))))
            chemo_probabilities[i, t] = chemo_prob
            radio_probabilities[i, t] = radio_prob

            # Action application
            if radio_application_rvs[i, t] < radio_prob:
                radio_application_point[i, t] = 1
                radio_dosage[i, t] = radio_amt[0]

            if chemo_application_rvs[i, t] < chemo_prob:
                # Apply chemo treatment
                chemo_application_point[i, t] = 1
                current_chemo_dose = chemo_amt[0]

            # Update chemo dosage
            chemo_dosage[i, t] = previous_chemo_dose * np.exp(-np.log(2) / drug_half_life) + current_chemo_dose

            cancer_volume[i, t + 1] = cancer_volume[i, t] * (1 +
                                                             + rho * np.log(K / cancer_volume[i, t])
                                                             - beta_c * chemo_dosage[i, t]
                                                             - (alpha * radio_dosage[i, t] + beta * radio_dosage[
                        i, t] ** 2)
                                                             + noise[t])  # add noise to fit residuals

            if cancer_volume[i, t + 1] > tumour_death_threshold:
                cancer_volume[i, t + 1] = tumour_death_threshold
                break  # patient death

            # recovery threshold as defined by the previous stuff
            if recovery_rvs[i, t + 1] < np.exp(-cancer_volume[i, t + 1] * tumour_cell_density):
                cancer_volume[i, t + 1] = 0
                break

        # Package outputs
        sequence_lengths[i] = int(t + 1)

    outputs = {'cancer_volume': cancer_volume,
               'chemo_dosage': chemo_dosage,
               'radio_dosage': radio_dosage,
               'chemo_application': chemo_application_point,
               'radio_application': radio_application_point,
               'chemo_probabilities': chemo_probabilities,
               'radio_probabilities': radio_probabilities,
               'sequence_lengths': sequence_lengths,
               'patient_types': patient_types
               }

    return outputs


def simulate_counterfactual_test_data(simulation_params, num_time_steps, assigned_actions=None):
    """
    Core routine to generate simulation test paths to asses all of the counterfactuals.

    :param simulation_params:
    :param num_time_steps:
    :param assigned_actions:
    :return:
    """
    np.random.seed(100)

    total_num_radio_treatments = 1
    total_num_chemo_treatments = 1

    num_treatments = 4  # No treatment/Chemotherapy/Radiotherapy/Chemotherapy + Radiotherapy

    radio_amt = np.array([2.0 for i in range(total_num_radio_treatments)])  # Gy
    radio_days = np.array([i + 1 for i in range(total_num_radio_treatments)])
    chemo_amt = [5.0 for i in range(total_num_chemo_treatments)]
    chemo_days = [(i + 1) * 7 for i in range(total_num_chemo_treatments)]

    # sort this
    chemo_idx = np.argsort(chemo_days)
    chemo_amt = np.array(chemo_amt)[chemo_idx]
    chemo_days = np.array(chemo_days)[chemo_idx]

    drug_half_life = 1  # one day half life for drugs

    # Unpack simulation parameters
    initial_stages = simulation_params['initial_stages']
    initial_volumes = simulation_params['initial_volumes']
    alphas = simulation_params['alpha']
    rhos = simulation_params['rho']
    betas = simulation_params['beta']
    beta_cs = simulation_params['beta_c']
    Ks = simulation_params['K']
    patient_types = simulation_params['patient_types']
    window_size = simulation_params['window_size']  # controls the lookback of the treatment assignment policy

    # Coefficients for treatment assignment probabilities
    chemo_sigmoid_intercepts = simulation_params['chemo_sigmoid_intercepts']
    radio_sigmoid_intercepts = simulation_params['radio_sigmoid_intercepts']
    chemo_sigmoid_betas = simulation_params['chemo_sigmoid_betas']
    radio_sigmoid_betas = simulation_params['radio_sigmoid_betas']

    num_patients = initial_stages.shape[0]

    num_test_points = num_patients * num_time_steps * num_treatments

    # Commence Simulation
    cancer_volume = np.zeros((num_test_points, num_time_steps))
    chemo_application_point = np.zeros((num_test_points, num_time_steps))
    radio_application_point = np.zeros((num_test_points, num_time_steps))
    sequence_lengths = np.zeros(num_test_points)
    patient_types_all_trajectories = np.zeros(num_test_points)

    test_idx = 0

    # Run actual simulation
    for i in range(num_patients):

        if (i%200 == 0):
            logging.info("Simulating patient {} of {}".format(i, num_patients))

        noise = 0.01 * np.random.randn(num_time_steps)  # 5% cell variability
        recovery_rvs = np.random.rand(num_time_steps)

        # initial values
        factual_cancer_volume = np.zeros(num_time_steps)
        factual_chemo_dosage = np.zeros(num_time_steps)
        factual_radio_dosage = np.zeros(num_time_steps)
        factual_chemo_application_point = np.zeros(num_time_steps)
        factual_radio_application_point = np.zeros(num_time_steps)
        factual_chemo_probabilities = np.zeros(num_time_steps)
        factual_radio_probabilities = np.zeros(num_time_steps)

        chemo_application_rvs = np.random.rand(num_time_steps)
        radio_application_rvs = np.random.rand(num_time_steps)

        factual_cancer_volume[0] = initial_volumes[i]

        alpha = alphas[i]
        beta = betas[i]
        beta_c = beta_cs[i]
        rho = rhos[i]
        K = Ks[i]

        for t in range(0, num_time_steps - 1):

            # Factual treatments and outcomes
            current_chemo_dose = 0.0
            previous_chemo_dose = 0.0 if t == 0 else factual_chemo_dosage[t - 1]

            # Action probabilities + death or recovery simulations
            cancer_volume_used = cancer_volume[i, max(t - window_size, 0):t + 1]
            cancer_diameter_used = np.array(
                [calc_diameter(vol) for vol in cancer_volume_used]).mean()  # mean diameter over 15 days
            cancer_metric_used = cancer_diameter_used

            # probabilities
            radio_prob = (1.0 / (1.0 + np.exp(-radio_sigmoid_betas[i]
                                              * (cancer_metric_used - radio_sigmoid_intercepts[i]))))
            chemo_prob = (1.0 / (1.0 + np.exp(- chemo_sigmoid_betas[i] *
                                              (cancer_metric_used - chemo_sigmoid_intercepts[i]))))

            factual_chemo_probabilities[t] = chemo_prob
            factual_radio_probabilities[t] = radio_prob

            # Action application
            if radio_application_rvs[t] < radio_prob:
                factual_radio_application_point[t] = 1
                factual_radio_dosage[t] = radio_amt[0]

            if chemo_application_rvs[t] < chemo_prob:
                factual_chemo_application_point[t] = 1
                current_chemo_dose = chemo_amt[0]

            # Update chemo dosage
            factual_chemo_dosage[t] = previous_chemo_dose * np.exp(-np.log(2) / drug_half_life) + current_chemo_dose

            # Factual treatments and outcomes
            factual_cancer_volume[t + 1] = factual_cancer_volume[t] * (1 + rho * np.log(K / factual_cancer_volume[t])
                                                                       - beta_c * factual_chemo_dosage[t] - (
                                                                               alpha * factual_radio_dosage[
                                                                           t] + beta * factual_radio_dosage[t] ** 2)
                                                                       + noise[t + 1])  # add noise to fit residuals

            factual_cancer_volume[t + 1] = np.clip(factual_cancer_volume[t + 1], 0, tumour_death_threshold)

            # Popoulate arrays
            cancer_volume[test_idx] = factual_cancer_volume
            chemo_application_point[test_idx] = factual_chemo_application_point
            radio_application_point[test_idx] = factual_radio_application_point
            patient_types_all_trajectories[test_idx] = patient_types[i]
            sequence_lengths[test_idx] = int(t) + 1
            test_idx = test_idx + 1

            # Counterfactual treatments and outcomes
            treatment_options = [(0, 0), (0, 1), (1, 0), (1, 1)]  # First = chemo; second = radio

            for treatment_option in treatment_options:
                if (factual_chemo_application_point[t] == treatment_option[0] and factual_radio_application_point[t] ==
                        treatment_option[1]):
                    # This represents the factual treatment which was already considered
                    continue
                current_chemo_dose = 0.0
                counterfactual_radio_dosage = 0.0
                counterfactual_chemo_application_point = 0
                counterfactual_radio_application_point = 0

                if treatment_option[0] == 1:
                    counterfactual_chemo_application_point = 1
                    current_chemo_dose = chemo_amt[0]

                if treatment_option[1] == 1:
                    counterfactual_radio_application_point = 1
                    counterfactual_radio_dosage = radio_amt[0]

                counterfactual_chemo_dosage = previous_chemo_dose * np.exp(
                    -np.log(2) / drug_half_life) + current_chemo_dose

                counterfactual_cancer_volume = factual_cancer_volume[t] * (
                        1 + rho * np.log(K / factual_cancer_volume[t])
                        - beta_c * counterfactual_chemo_dosage - (
                                alpha * counterfactual_radio_dosage + beta * counterfactual_radio_dosage ** 2)
                        + noise[t + 1])

                cancer_volume[test_idx][:t + 2] = np.append(factual_cancer_volume[:t + 1],
                                                            [counterfactual_cancer_volume])
                chemo_application_point[test_idx][:t + 1] = np.append(factual_chemo_application_point[:t],
                                                                      [counterfactual_chemo_application_point])
                radio_application_point[test_idx][:t + 1] = np.append(factual_radio_application_point[:t],
                                                                      [counterfactual_radio_application_point])
                patient_types_all_trajectories[test_idx] = patient_types[i]
                sequence_lengths[test_idx] = int(t) + 1
                test_idx = test_idx + 1

            if (factual_cancer_volume[t + 1] >= tumour_death_threshold) or \
                    recovery_rvs[t] <= np.exp(-factual_cancer_volume[t + 1] * tumour_cell_density):
                break

    outputs = {'cancer_volume': cancer_volume[:test_idx],
               'chemo_application': chemo_application_point[:test_idx],
               'radio_application': radio_application_point[:test_idx],
               'sequence_lengths': sequence_lengths[:test_idx],
               'patient_types': patient_types_all_trajectories[:test_idx]
               }

    print("Call to simulate counterfactuals data")

    return outputs


def simulate_sequence_test(simulation_params, num_time_steps, projection_horizon, treatment_options):
    """
    Core routine to generate simulation test paths to asses all of the counterfactuals.

    :param simulation_params:
    :param num_time_steps:
    :param assigned_actions:
    :return:
    """

    np.random.seed(100)

    total_num_radio_treatments = 1
    total_num_chemo_treatments = 1

    radio_amt = np.array([2.0 for i in range(total_num_radio_treatments)])  # Gy
    radio_days = np.array([i + 1 for i in range(total_num_radio_treatments)])
    chemo_amt = [5.0 for i in range(total_num_chemo_treatments)]
    chemo_days = [(i + 1) * 7 for i in range(total_num_chemo_treatments)]

    # sort this
    chemo_idx = np.argsort(chemo_days)
    chemo_amt = np.array(chemo_amt)[chemo_idx]
    chemo_days = np.array(chemo_days)[chemo_idx]

    drug_half_life = 1  # one day half life for drugs

    # Unpack simulation parameters
    initial_stages = simulation_params['initial_stages']
    initial_volumes = simulation_params['initial_volumes']
    alphas = simulation_params['alpha']
    rhos = simulation_params['rho']
    betas = simulation_params['beta']
    beta_cs = simulation_params['beta_c']
    Ks = simulation_params['K']
    patient_types = simulation_params['patient_types']
    window_size = simulation_params['window_size']  # controls the lookback of the treatment assignment policy

    # Coefficients for treatment assignment probabilities
    chemo_sigmoid_intercepts = simulation_params['chemo_sigmoid_intercepts']
    radio_sigmoid_intercepts = simulation_params['radio_sigmoid_intercepts']
    chemo_sigmoid_betas = simulation_params['chemo_sigmoid_betas']
    radio_sigmoid_betas = simulation_params['radio_sigmoid_betas']

    num_patients = initial_stages.shape[0]

    num_test_points = 2 * num_patients * num_time_steps * projection_horizon

    # Commence Simulation
    cancer_volume = np.zeros((num_test_points, num_time_steps + projection_horizon))
    chemo_application_point = np.zeros((num_test_points, num_time_steps + projection_horizon))
    radio_application_point = np.zeros((num_test_points, num_time_steps + projection_horizon))
    sequence_lengths = np.zeros(num_test_points)
    patient_types_all_trajectories = np.zeros(num_test_points)
    patient_ids_all_trajectories = np.zeros(num_test_points)
    patient_current_t = np.zeros(num_test_points)

    test_idx = 0

    # Run actual simulation
    for i in range(num_patients):

        if (i%200 == 0):
            logging.info("Simulating patient {} of {}".format(i, num_patients))

        noise = 0.01 * np.random.randn(num_time_steps + 20)  # 5% cell variability
        recovery_rvs = np.random.rand(num_time_steps)

        # initial values
        factual_cancer_volume = np.zeros(num_time_steps)
        factual_chemo_dosage = np.zeros(num_time_steps)
        factual_radio_dosage = np.zeros(num_time_steps)
        factual_chemo_application_point = np.zeros(num_time_steps)
        factual_radio_application_point = np.zeros(num_time_steps)
        factual_chemo_probabilities = np.zeros(num_time_steps)
        factual_radio_probabilities = np.zeros(num_time_steps)

        chemo_application_rvs = np.random.rand(num_time_steps)
        radio_application_rvs = np.random.rand(num_time_steps)

        factual_cancer_volume[0] = initial_volumes[i]

        alpha = alphas[i]
        beta = betas[i]
        beta_c = beta_cs[i]
        rho = rhos[i]
        K = Ks[i]

        for t in range(0, num_time_steps - 1):

            # Factual treatments and outcomes
            current_chemo_dose = 0.0
            previous_chemo_dose = 0.0 if t == 0 else factual_chemo_dosage[t - 1]

            # Action probabilities + death or recovery simulations
            cancer_volume_used = cancer_volume[i, max(t - window_size, 0):t + 1]
            cancer_diameter_used = np.array(
                [calc_diameter(vol) for vol in cancer_volume_used]).mean()  # mean diameter over 15 days
            cancer_metric_used = cancer_diameter_used

            # probabilities
            radio_prob = (1.0 / (1.0 + np.exp(-radio_sigmoid_betas[i]
                                              * (cancer_metric_used - radio_sigmoid_intercepts[i]))))
            chemo_prob = (1.0 / (1.0 + np.exp(- chemo_sigmoid_betas[i] *
                                              (cancer_metric_used - chemo_sigmoid_intercepts[i]))))

            factual_chemo_probabilities[t] = chemo_prob
            factual_radio_probabilities[t] = radio_prob

            # Action application
            if radio_application_rvs[t] < radio_prob:
                factual_radio_application_point[t] = 1
                factual_radio_dosage[t] = radio_amt[0]

            if chemo_application_rvs[t] < chemo_prob:
                factual_chemo_application_point[t] = 1
                current_chemo_dose = chemo_amt[0]

            # Update chemo dosage
            factual_chemo_dosage[t] = previous_chemo_dose * np.exp(-np.log(2) / drug_half_life) + current_chemo_dose

            # Factual treatments and outcomes
            factual_cancer_volume[t + 1] = factual_cancer_volume[t] * (1 + rho * np.log(K / factual_cancer_volume[t])
                                                                       - beta_c * factual_chemo_dosage[t] - (
                                                                               alpha * factual_radio_dosage[
                                                                           t] + beta * factual_radio_dosage[t] ** 2)
                                                                       + noise[t + 1])  # add noise to fit residuals

            factual_cancer_volume[t + 1] = np.clip(factual_cancer_volume[t + 1], 0, tumour_death_threshold)

            for treatment_option in treatment_options:

                counterfactual_cancer_volume = np.zeros(shape=(t + 1 + projection_horizon + 1))
                counterfactual_chemo_application_point = np.zeros(shape=(t + 1 + projection_horizon))
                counterfactual_radio_application_point = np.zeros(shape=(t + 1 + projection_horizon))
                counterfactual_chemo_dosage = np.zeros(shape=(t + 1 + projection_horizon))
                counterfactual_radio_dosage = np.zeros(shape=(t + 1 + projection_horizon))

                counterfactual_cancer_volume[:t + 2] = factual_cancer_volume[:t + 2]
                counterfactual_chemo_application_point[:t + 1] = factual_chemo_application_point[:t + 1]
                counterfactual_radio_application_point[:t + 1] = factual_radio_application_point[:t + 1]
                counterfactual_chemo_dosage[:t + 1] = factual_chemo_dosage[:t + 1]
                counterfactual_radio_dosage[:t + 1] = factual_radio_dosage[:t + 1]

                for projection_time in range(0, projection_horizon):

                    current_t = t + 1 + projection_time
                    previous_chemo_dose = counterfactual_chemo_dosage[current_t - 1]

                    current_chemo_dose = 0.0
                    counterfactual_radio_dosage[current_t] = 0.0
                    if treatment_option[projection_time][0] == 1:
                        counterfactual_chemo_application_point[current_t] = 1
                        current_chemo_dose = chemo_amt[0]

                    if treatment_option[projection_time][1] == 1:
                        counterfactual_radio_application_point[current_t] = 1
                        counterfactual_radio_dosage[current_t] = radio_amt[0]

                    counterfactual_chemo_dosage[current_t] = previous_chemo_dose * np.exp(
                        -np.log(2) / drug_half_life) + current_chemo_dose

                    counterfactual_cancer_volume[current_t + 1] = counterfactual_cancer_volume[current_t] * \
                                                                  (1 + rho * np.log(
                                                                      K / (counterfactual_cancer_volume[
                                                                               current_t] + 1e-07) + 1e-07)
                                                                   - beta_c * counterfactual_chemo_dosage[current_t] - (
                                                                           alpha * counterfactual_radio_dosage[
                                                                       current_t] + beta *
                                                                           counterfactual_radio_dosage[
                                                                               current_t] ** 2)
                                                                   + noise[current_t + 1])

                if (np.isnan(counterfactual_cancer_volume).any()):
                    continue

                cancer_volume[test_idx][:t + 1 + projection_horizon + 1] = counterfactual_cancer_volume
                chemo_application_point[test_idx][:t + 1 + projection_horizon] = counterfactual_chemo_application_point
                radio_application_point[test_idx][:t + 1 + projection_horizon] = counterfactual_radio_application_point
                patient_types_all_trajectories[test_idx] = patient_types[i]
                patient_ids_all_trajectories[test_idx] = i
                patient_current_t[test_idx] = t

                sequence_lengths[test_idx] = int(t) + 2
                test_idx = test_idx + 1

            if (factual_cancer_volume[t + 1] >= tumour_death_threshold) or \
                    recovery_rvs[t] <= np.exp(-factual_cancer_volume[t + 1] * tumour_cell_density):
                break

    outputs = {'cancer_volume': cancer_volume[:test_idx],
               'chemo_application': chemo_application_point[:test_idx],
               'radio_application': radio_application_point[:test_idx],
               'sequence_lengths': sequence_lengths[:test_idx],
               'patient_types': patient_types_all_trajectories[:test_idx],
               'patient_ids_all_trajectories': patient_ids_all_trajectories[:test_idx],
               'patient_current_t': patient_current_t[:test_idx],
               }

    print("Call to simulate counterfactuals data")

    return outputs


def get_scaling_params(sim):
    real_idx = ['cancer_volume', 'chemo_dosage', 'radio_dosage']

    # df = pd.DataFrame({k: sim[k] for k in real_idx})
    means = {}
    stds = {}
    seq_lengths = sim['sequence_lengths']
    for k in real_idx:
        active_values = []
        for i in range(seq_lengths.shape[0]):
            end = int(seq_lengths[i])
            active_values += list(sim[k][i, :end])

        means[k] = np.mean(active_values)
        stds[k] = np.std(active_values)

    # Add means for static variables`
    means['patient_types'] = np.mean(sim['patient_types'])
    stds['patient_types'] = np.std(sim['patient_types'])

    return pd.Series(means), pd.Series(stds)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Plotting Functions


def plot_treatments(patient):
    df = pd.DataFrame({'N(t)': outputs['cancer_volume'][patient],
                       'C(t)': outputs['chemo_application'][patient],
                       'd(t)': outputs['radio_application'][patient],
                       })
    df = df[['N(t)', "C(t)", "d(t)"]]
    df.plot(secondary_y=['C(t)', 'd(t)'])
    plt.xlabel("$t$")
    plt.show()


def plot_sigmoid_function():
    """
    Simple plots to visualise probabilities of treatment assignments

    :return:
    """

    # Profile of treatment application sigmoid
    for coeff in [i for i in range(11)]:
        tumour_death_threshold = calc_volume(13)
        assigned_beta = coeff / tumour_death_threshold
        assigned_interp = tumour_death_threshold / 2
        idx = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        volumes = idx * tumour_death_threshold

        def sigmoid_fxn(volume, beta, intercept):
            return (1.0 / (1.0 + np.exp(-beta * (volume - intercept))))

        outputs[coeff] = pd.Series(sigmoid_fxn(volumes, assigned_beta, assigned_interp), index=idx)

    df = pd.DataFrame(outputs)
    df.plot()
    plt.show()


def get_cancer_sim_data(chemo_coeff, radio_coeff, b_load, b_save=False, seed=100, model_root='results', window_size=15):
    if window_size == 15:
        pickle_file = os.path.join(model_root, 'new_cancer_sim_{}_{}.p'.format(chemo_coeff, radio_coeff))
    else:
        pickle_file = os.path.join(model_root,
                                   'new_cancer_sim_{}_{}_{}.p'.format(chemo_coeff, radio_coeff, window_size))

    def _generate():
        num_time_steps = 60  # about half a year
        np.random.seed(seed)
        num_patients = 10000

        params = get_confounding_params(num_patients, chemo_coeff=chemo_coeff,
                                            radio_coeff=radio_coeff)
        params['window_size'] = window_size
        training_data = simulate(params, num_time_steps)

        params = get_confounding_params(int(num_patients / 10), chemo_coeff=chemo_coeff,
                                            radio_coeff=radio_coeff)
        params['window_size'] = window_size
        validation_data = simulate(params, num_time_steps)

        params = get_confounding_params(int(num_patients / 10), chemo_coeff=chemo_coeff,
                                            radio_coeff=radio_coeff)
        params['window_size'] = window_size
        test_data_factuals = simulate(params, num_time_steps)
        test_data_counterfactuals = simulate_counterfactual_test_data(params, num_time_steps)

        params = get_confounding_params(int(num_patients / 10), chemo_coeff=chemo_coeff,
                                            radio_coeff=radio_coeff)
        params['window_size'] = window_size
        treatment_options = np.array(
            [[(1, 0), (0, 0), (0, 0), (0, 0), (0, 0)],
             [(0, 0), (1, 0), (0, 0), (0, 0), (0, 0)],
             [(0, 0), (0, 0), (1, 0), (0, 0), (0, 0)],
             [(0, 0), (0, 0), (0, 0), (1, 0), (0, 0)],
             [(0, 0), (0, 0), (0, 0), (0, 0), (1, 0)],
             [(0, 1), (0, 0), (0, 0), (0, 0), (0, 0)],
             [(0, 0), (0, 1), (0, 0), (0, 0), (0, 0)],
             [(0, 0), (0, 0), (0, 1), (0, 0), (0, 0)],
             [(0, 0), (0, 0), (0, 0), (0, 1), (0, 0)],
             [(0, 0), (0, 0), (0, 0), (0, 0), (0, 1)]
             ])
        test_data_seq = simulate_sequence_test(params, num_time_steps, 5, treatment_options)

        scaling_data = get_scaling_params(training_data)

        pickle_map = {'chemo_coeff': chemo_coeff,
                      'radio_coeff': radio_coeff,
                      'num_time_steps': num_time_steps,
                      'training_data': training_data,
                      'validation_data': validation_data,
                      'test_data': test_data_counterfactuals,
                      'test_data_factuals': test_data_factuals,
                      'test_data_seq': test_data_seq,
                      'scaling_data': scaling_data,
                      'window_size': window_size}

        if b_save:
            logging.info("Saving pickle map to {}".format(pickle_file))
            pickle.dump(pickle_map, open(pickle_file, 'wb'))
        return pickle_map

    # Controls whether to regenerate the data, or load from a persisted file
    if not b_load:
        pickle_map = _generate()

    else:
        logging.info("Loading pickle map from {}".format(pickle_file))

        try:
            pickle_map = pickle.load(open(pickle_file, "rb"))

        except IOError:
            logging.info("Pickle file does not exist, regenerating: {}".format(pickle_file))
            pickle_map = _generate()

    return pickle_map


def create_cancer_data(N,gamma, T_cond = 0, T_horizon = 0, random_seed = 42, counterfactual = False):

    np.random.seed(random_seed)
    num_time_steps = 60  # 6 month followup
    num_patients = N

    simulation_params = get_confounding_params(num_patients, chemo_coeff=gamma, radio_coeff=gamma)
    simulation_params['window_size'] = 15

    if counterfactual:
        outputs = simulate_sequence_test(simulation_params, num_time_steps)
    else:
        outputs = simulate(simulation_params, num_time_steps)
    
    t = np.arange(1,1+T_cond+T_horizon)
    t_x = t[t<=T_cond]
    t_y = t[t>T_cond]
    
    X = outputs["cancer_volume"][:,:len(t)]
    A = np.stack((outputs["radio_application"],outputs["chemo_application"]),-1)[:,:len(t)]
   
    Y = X[:,t>T_cond]
    X = X[:,t<=T_cond]
   
    A_x = A[:,t<=T_cond]
    A_y = A[:,t>T_cond]
    B = torch.nn.functional.one_hot(torch.LongTensor(outputs["patient_types"]-1),3)
    
    X = X[...,None]
    Y = Y[...,None]

    M_before = torch.ones(X.shape)
    M_after = torch.ones(Y.shape)
    
    times_X = torch.Tensor(t_x[None,:]).repeat(Y.shape[0],1)
    times_Y = torch.Tensor(t_y[None,:]).repeat(Y.shape[0],1)

    return torch.Tensor(X), None, torch.Tensor(Y), B.float(), None, times_X, times_Y, torch.Tensor(A_x), torch.Tensor(A_y), M_before, M_after

class CancerDataset(Dataset):
    def __init__(self,N, gamma,noise_std, seed, T_cond, T_horizon, counterfactual = False, dmm_order = False):

        X, T, Y, X_0,_ , times_X, times_Y, A_x, A_y, M_before, M_after = create_cancer_data(N, gamma, T_cond = T_cond, T_horizon = T_horizon, random_seed = seed, counterfactual = counterfactual)
        self.X = X.permute(0,2,1)
        self.T = torch.zeros(X.shape[0],1)
        self.Y = Y.permute(0,2,1)
        self.p = torch.zeros(X.shape[0],3)
        self.X_0 = X_0
        self.A_x = A_x.permute(0,2,1)
        self.A_y = A_y.permute(0,2,1)
        self.M_before = M_before.permute(0,2,1)
        self.M_after = M_after.permute(0,2,1)
        
        self.conditional_dim = self.X.shape[1] + self.M_before.shape[1] + self.A_x.shape[1]
        self.output_dim = self.Y.shape[1]
        self.conditional_len = self.X.shape[-1]
        self.baseline_size = self.X_0.shape[-1]
        self.treatment_dim = self.A_x.shape[1]
        self.ts_dim = self.X.shape[1]
       
        means_x = self.X.mean(dim=(0,2))[None,:,None]
        stds_x = self.X.std(dim=(0,2))[None,:,None]

        #means_y = self.Y.mean(dim=(0,2))[None,:,None]
        #stds_y = self.Y.std(dim=(0,2))[None,:,None]

        self.X = (self.X-means_x)/stds_x
        self.Y = (self.Y-means_x)/stds_x

        self.times_X = times_X
        self.times_Y = times_Y

        self.CE = torch.zeros(self.X.shape[0])
        self.Y_countdown = torch.zeros(self.X.shape[0])
        
        self.dmm_order = dmm_order
        if self.dmm_order:
            self.B = self.X_0
            self.X = torch.cat((self.X,self.Y),-1).permute(0,2,1)
            self.A = torch.cat((self.A_x,self.A_y),-1).permute(0,2,1)
            self.M = torch.cat((self.M_before,self.M_after),-1).permute(0,2,1)
            self.CE = torch.zeros(self.X.shape[0])
            self.events = torch.zeros(self.X.shape[0],self.X.shape[1])

    def __getitem__(self,idx):
        if self.dmm_order:
            return self.B[idx], self.X[idx], self.A[idx], self.M[idx], self.events[idx], self.CE[idx]
        else:
            return self.X[idx], self.Y[idx], self.T[idx], self.Y[idx], self.p[idx], self.X_0[idx], self.A_x[idx], self.A_y[idx], self.M_before[idx], self.M_after[idx], self.times_X[idx], self.times_Y[idx], self.Y_countdown[idx], self.CE[idx]
    def __len__(self):
        return self.X.shape[0]

class CancerDataModule(pl.LightningDataModule):
    def __init__(self,batch_size, seed, N_ts, gamma, noise_std, num_workers = 4,  T_cond = 0, T_horizon = 0, **kwargs):
        
        super().__init__()
        self.batch_size = batch_size
        self.seed = seed
        self.num_workers = num_workers

        self.train_shuffle = True

        self.input_dim = 1
        self.output_dim = 1

        self.N = N_ts
        self.gamma = gamma
        self.noise_std = noise_std

        self.T_cond = T_cond
        self.T_horizon = T_horizon

        if kwargs["model_type"] == "DMM":
            self.dmm_order = True
        else:
            self.dmm_order = False

    def prepare_data(self):

        self.dataset = CancerDataset(self.N, self.gamma, self.noise_std, self.seed,  T_cond = self.T_cond, T_horizon = self.T_horizon, dmm_order = self.dmm_order)       
        self.cf_dataset = CancerDataset(self.N, self.gamma, self.noise_std, self.seed,  T_cond = self.T_cond, T_horizon = self.T_horizon, counterfactual = False, dmm_order = self.dmm_order)       
        
        train_idx = np.arange(len(self.dataset))[:int(0.5*len(self.dataset))]
        val_idx = np.arange(len(self.dataset))[int(0.5*len(self.dataset)):]
        test_idx = val_idx[int(len(val_idx)/2):]
        val_idx = val_idx[:int(len(val_idx)/2)]

        if self.batch_size==0:
            self.train_batch_size = len(train_idx)
            self.val_batch_size = len(val_idx)
            self.test_batch_size = len(test_idx)
        else:
            self.train_batch_size = self.batch_size
            self.val_batch_size = self.batch_size
            self.test_batch_size = self.batch_size

        self.train = Subset(self.dataset,train_idx)
        self.val = Subset(self.dataset,val_idx)
        self.test = Subset(self.dataset,test_idx)

        self.conditional_dim = self.dataset.conditional_dim
        self.conditional_len = self.dataset.conditional_len
        self.output_dim = self.dataset.output_dim
        self.baseline_size = self.dataset.baseline_size
        self.treatment_dim = self.dataset.treatment_dim
        self.ts_dim = self.dataset.ts_dim

        self.output_dims = [i for i in range(self.output_dim)]

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.train_batch_size,
            shuffle=self.train_shuffle,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.val_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True
        )

    def test_dataloader(self, shuffle = False):
        return DataLoader(
            self.test,
            batch_size=self.test_batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True
            )
    def cf_test_dataloader(self, shuffle = False):
        return DataLoader(
            self.cf_dataset,
            batch_size=self.test_batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True
            )

    @classmethod
    def add_dataset_specific_args(cls, parent):
        import argparse
        parser = argparse.ArgumentParser(parents=[parent], add_help=False)
        parser.add_argument('--seed', type=int, default=42)
        parser.add_argument('--batch_size', type=int, default=128)
        parser.add_argument('--N_ts', type=int, default=1000)
        parser.add_argument('--gamma', type=float, default=0)
        parser.add_argument('--noise_std', type=float, default=0)
        parser.add_argument('--T_cond', type=int, default=10)
        parser.add_argument('--T_horizon', type=int, default=5)
        parser.add_argument('--num_workers', type=int, default=4)
        return parser



   
if __name__=="__main__":
    dataset = CancerDataModule(batch_size = 32, N_ts = 1000, noise_std = 0., gamma = 0., seed = 42, T_cond = 10, T_horizon = 7)
    dataset.prepare_data()

    import ipdb;  ipdb.set_trace()
    #dataset = Dataset(path="/home/edward/Data/pre_processed_mimic/models/")
    #datam = FluidDataModule(path = "/home/edward/Data/pre_processed_mimic/models/")

    #datam.prepare_data()

    #for i,b in enumerate(datam.train_dataloader()):
    #    print(b)
    #    import ipdb; ipdb.set_trace()
    #    #break

    #dataset = CancerDataModule(batch_size = 32, seed = 23, chemo_coeff = 2, radio_coeff = 2, window_size = 15, num_time_steps = 20, t_limit = 10, num_workers = 4, )
    #dataset.prepare_data()
