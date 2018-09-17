#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import pickle

from data_processing.loading import load_save_all
from model_plan.pre_processing import pre_processing
from model_plan.gp_model_approx import model_approx_continuous_example, model_approx_create_interp_data
from model_plan.movement_model import save_default_model
from model_plan.policies import generate_mdp_policies


def process_user(root_path, usr, training_protocol, modalities):
    for modality in modalities:
        load_save_all(root_path + "/user" + str(usr) + "/" + modality,
                      training_protocol, flag_transform_to_body=True)

        pre_processing(root_path, "user" + str(usr), modality)

        model_approx_continuous_example(root_path + "/user" + str(usr), modality, flag_train_model=True)

    model_approx_create_interp_data(root_path + "/user" + str(usr))


if __name__ == "__main__":
    data_path = "/home/yuhang/Documents/proactive_guidance/training_data"
    training_protocol = "../resources/protocols/random_continuous_protocol_5rep2.txt"
    # modalities = ["haptic", "audio"]
    modalities = ["haptic"]

    # process_user(data_path, 2, training_protocol, modalities)

    # save_default_model(2, "../resources/pretrained_models/human_models")

    generate_mdp_policies("../resources/protocols/free_space_exp_protocol_7targets_mdp.txt",
                          "/home/yuhang/Documents/proactive_guidance/training_data/user2/pretrained_model",
                          "haptic", 2)
