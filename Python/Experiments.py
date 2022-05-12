#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 16:09:21 2022

@author: yunhui, xinyi
"""

# cd "/Users/yunhui/Library/Mobile Documents/com~apple~CloudDocs/PhD/Interns/2021 Summer/Generative Models/DGMmicroRNA/Python"
# %% Import libraries
import torch
import pandas as pd
import re
import seaborn as sns
import numpy as np
from pathlib import Path
from helper_utils import preprocessinglog2, create_labels, draw_pilot
from helper_training import training_AEs, training_GANs, training_iter, training_flows

sns.set()


# %% Define pilot experiments functions
def PilotExperiment(dataname, pilot_size, model, batch_frac, learning_rate, AE_head):
    # train GLOW, RealNVP with several pilot size given data, model, batch_size, learning rate
    path = "../RealData/" + dataname + ".csv"
    dat_pd = pd.read_csv(path, header=0)
    if dataname == "SKCMLAMLPositive_3":
        dat_pd_1 = dat_pd[dat_pd['groups'] == "SKCM"]
        dat_pd_2 = dat_pd[dat_pd['groups'] == "LAML"]
    data_pd = dat_pd.select_dtypes(include=np.number)
    print(data_pd)
    oridata = torch.from_numpy(data_pd.to_numpy()).to(torch.float32)
    oridata = preprocessinglog2(oridata)
    n_samples = oridata.shape[0]
    if "groups" in dat_pd.columns:
        groups = dat_pd['groups']
    else:
        groups = None

    orilabels, oriblurlabels = create_labels(n_samples=n_samples, groups=groups)

    print("1. Read data, path is " + path)

    # get model name and kl_weight if modelname is some autoencoder
    if len(re.split("([A-Z]+)(\d)([-+])(\d+)", model)) > 1:
        kl_weight = int(re.split("([A-Z]+)(\d)([-+])(\d+)", model)[4])
        modelname = re.split("([A-Z]+)(\d)([-+])(\d+)", model)[1]
    else:
        modelname = model
        kl_weight = 1

    print("2. Determine the model is " + model + " with kl-weight = " + str(kl_weight))
    # hyperparameters
    random_seed = 123
    num_epochs = 150
    repli = 5
    if (len(torch.unique(orilabels)) > 1) & (int(sum(orilabels == 0)) != int(sum(orilabels == 1))):
        new_size = [int(sum(orilabels == 0)), int(sum(orilabels == 1)), repli]
    else:
        new_size = [repli * n_samples]
    print("3. Pilot experiments start ... ")
    for n_pilot in pilot_size:
        for rand_pilot in [1, 2, 3, 4, 5]:
            print(
                "Training for data=" + dataname + ", model=" + model + ", pilot size=" + str(n_pilot) + ", for " + str(
                    rand_pilot) + "-th draw")
            # get pilot_size real samples as seeds for DGM
            rawdata, rawlabels, rawblurlabels = draw_pilot(dataset=oridata, labels=orilabels, blurlabels=oriblurlabels,
                                                           n_pilot=n_pilot, seednum=rand_pilot)

            savepath = "../ReconsData/" + dataname + "_" + model + "_" + str(n_pilot) + "_Draw" + str(
                rand_pilot) + ".csv"
            savepathnew = "../GeneratedData/" + dataname + "_" + model + "_" + str(n_pilot) + "_Draw" + str(
                rand_pilot) + ".csv"
            losspath = "../Loss/" + dataname + "_" + model + "_" + str(n_pilot) + "_Draw" + str(rand_pilot) + ".csv"

            if (modelname != "CVAE") and (torch.unique(rawlabels).shape[0] > 1):
                rawdata = torch.cat((rawdata, rawblurlabels), dim=1)

            # if AE_head = True, for each pilot size, 2 iterative AE reconstruction will be conducted first
            # resulting in n_pilot * 4 samples, and the extended samples will be input to the model specified by modelname
            if AE_head:
                savepath = "../ReconsData/" + dataname + "_AEhead_" + model + "_" + str(n_pilot) + "_Draw" + str(
                    rand_pilot) + ".csv"
                savepathnew = "../GeneratedData/" + dataname + "_AEhead_" + model + "_" + str(n_pilot) + "_Draw" + str(
                    rand_pilot) + ".csv"
                savepathextend = "../ExtendData/" + dataname + "_AEhead_" + model + "_" + str(n_pilot) + "_Draw" + str(
                    rand_pilot) + ".csv"
                losspath = "../Loss/" + dataname + "_AEhead_" + model + "_" + str(n_pilot) + "_Draw" + str(
                    rand_pilot) + ".csv"
                print("AE reconstruction head is added, reconstruction starting ...")
                feed_data, feed_labels = training_iter(iter_times=2,
                                                       # how many times to iterative, will get pilot_size * 2^iter_times reconstructed samples
                                                       savepathextend=savepathextend,
                                                       # save path of the extended dataset
                                                       rawdata=rawdata,  # pilot data
                                                       rawlabels=rawlabels,  # pilot labels
                                                       random_seed=123,
                                                       modelname="AE",  # choose from AE, VAE
                                                       num_epochs=1000,
                                                       learning_rate=0.0005,
                                                       kl_weight=1,
                                                       loss_fn="MSE",
                                                       replace=True,
                                                       # whether to replace the failure features in each reconstruction
                                                       saveextend=True,
                                                       plot=True)

                rawdata = feed_data
                rawlabels = feed_labels
                print("Reconstruction finish.")
            # Training
            if ("GAN" in modelname):
                log_dict = training_GANs(savepathnew=savepathnew,  # path to save newly generated samples
                                         rawdata=rawdata,  # raw data matrix with samples in row, features in column
                                         rawlabels=rawlabels,
                                         # labels for each sample, n_samples * 1, will not be used in AE or VAE
                                         batch_size=round(rawdata.shape[0] * batch_frac),
                                         # batch size, note rawdata.shape[0] = n_pilot if no AE_head
                                         random_seed=random_seed,
                                         modelname=modelname,  # choose from "GAN","WGAN","WGANGP"
                                         num_epochs=num_epochs,
                                         learning_rate=learning_rate,
                                         new_size=new_size,  # how many new samples you want to generate
                                         save_new=True,  # whether to save the newly generated samples
                                         plot=False)

                print("GAN model training for one pilot size finishs.")

                log_pd = pd.DataFrame({'discriminator': log_dict['train_discriminator_loss_per_batch'],
                                       'generator': log_dict['train_generator_loss_per_batch']})
                log_pd.to_csv(Path(losspath), index=False)

            elif ("AE" in modelname):
                log_dict = training_AEs(savepath=savepath,
                                        savepathnew=savepathnew,
                                        rawdata=rawdata,
                                        rawlabels=rawlabels,
                                        batch_size=round(rawdata.shape[0] * batch_frac),
                                        random_seed=random_seed,
                                        modelname=modelname,
                                        num_epochs=num_epochs,
                                        learning_rate=learning_rate,
                                        kl_weight=kl_weight,
                                        loss_fn="MSE",
                                        save_recons=True,
                                        new_size=new_size,
                                        save_new=True,
                                        plot=False)

                print("VAEs model training for one pilot size finishs.")
                log_pd = pd.DataFrame({'kl': log_dict['train_kl_loss_per_batch'],
                                       'recons': log_dict['train_reconstruction_loss_per_batch']})
                log_pd.to_csv(Path(losspath), index=False)
            elif ("maf" in modelname):
                training_flows(savepathnew=savepathnew,
                               rawdata=rawdata,
                               batch_size=round(n_pilot * batch_frac),
                               random_seed=random_seed,
                               modelname=modelname,
                               num_blocks=5,
                               num_epoches=num_epochs,
                               learning_rate=learning_rate,
                               new_size=repli * n_samples,
                               num_hidden=226,
                               plot=False)
            elif ("realnvp" in modelname):
                training_flows(savepathnew=savepathnew,
                               rawdata=rawdata,
                               batch_size=round(n_pilot * batch_frac),
                               random_seed=random_seed,
                               modelname=modelname,
                               num_blocks=5,
                               num_epoches=num_epochs,
                               learning_rate=learning_rate,
                               new_size=repli * n_samples,
                               num_hidden=226,
                               plot=False)
            elif ("glow" in modelname):
                training_flows(savepathnew=savepathnew,
                               rawdata=rawdata,
                               batch_size=round(n_pilot * batch_frac),
                               random_seed=random_seed,
                               modelname=modelname,
                               num_blocks=5,
                               num_epoches=num_epochs,
                               learning_rate=learning_rate,
                               new_size=repli * n_samples,
                               num_hidden=226,
                               plot=False)
            else:
                print("wait for other models")


def ExperimentOnSet(dataname_Ind=True, batch_frac_Ind=False, learning_rate_Ind=False, AE_head_Ind=False):
    # pilot_size_set = [20, 40, 60, 80, 100, 150]
    pilot_size_set = [60,80,100,150]
    # model_set = ["VAE1-1", "VAE1-5", "VAE1-10", "GAN", "WGAN", "WGANGP"]
    # dataname_set = ["SKCM", "SKCMPositive_4"]
    # dataname_set = ["SKCMLAML", "SKCMLAMLPositive_3"]
    # dataname_set = ["SKCMLAMLPositive_3", "SKCMLAMLPositive_3_DESeq", "SKCMLAMLPositive_3_RUVr",
    #                 "SKCMLAMLPositive_3_PoissonSeq", "SKCMLAMLPositive_3_TMM"]
    # model_set = ["CVAE1-1", "CVAE1-5", "CVAE1-10"]
    model_set = ["realnvp"]
    dataname_set = ["SKCMPositive_4"]
    batch_frac_set = [0.1, 0.2, 0.3]
    learning_rate_set = [0.05, 0.005, 0.0005]
    if (dataname_Ind):
        for dataname in dataname_set:
            for model in model_set:
                batch_frac = 0.2
                learning_rate = 0.0005
                PilotExperiment(dataname=dataname, pilot_size=pilot_size_set, model=model, batch_frac=batch_frac,
                                learning_rate=learning_rate, AE_head=False)
    elif (batch_frac_Ind):
        for batch_frac in batch_frac_set:
            for model in model_set:
                learning_rate = 0.0005
                dataname = "SKCM"
                PilotExperiment(dataname=dataname, pilot_size=pilot_size_set, model=model, batch_frac=batch_frac,
                                learning_rate=learning_rate, AE_head=False)
    elif (learning_rate_Ind):
        for learning_rate in learning_rate_set:
            for model in model_set:
                dataname = "SKCM"
                batch_frac = 0.1
                PilotExperiment(dataname=dataname, pilot_size=pilot_size_set, model=model, batch_frac=batch_frac,
                                learning_rate=learning_rate, AE_head=False)
    elif (AE_head_Ind):
        for AE_head in [True, False]:
            for model in model_set:
                dataname = "SKCMPositive_4"
                batch_frac = 0.2
                learning_rate = 0.0005
                PilotExperiment(dataname=dataname, pilot_size=pilot_size_set, model=model, batch_frac=batch_frac,
                                learning_rate=learning_rate, AE_head=True)


if __name__ == '__main__':
    ExperimentOnSet()
