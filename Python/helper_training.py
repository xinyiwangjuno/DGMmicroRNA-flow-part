#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 15:14:37 2022

@author: yunhui, xinyi
"""

# %%
from helper_utils import set_all_seeds, plot_recons_samples, plot_new_samples, plot_training_loss, \
    plot_multiple_training_losses
from helper_train import train_AE, train_VAE, train_CVAE, train_GAN, train_WGAN, train_WGANGP
from helper_models import AE, VAE, CVAE, GAN
import flows as fnn
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import numpy as np
import copy
import sys
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from tqdm import tqdm
from tensorboardX import SummaryWriter


# %%
def training_AEs(savepath,  # path to save reconstructed samples
                 savepathnew,  # path to save newly generated samples
                 rawdata,  # raw data tensor with samples in row, features in column
                 rawlabels,  # labels for each sample, n_samples * 1, will not be used in AE or VAE
                 batch_size,  # batch size
                 random_seed,
                 modelname,  # choose from "AE","VAE","CVAE"
                 num_epochs,
                 learning_rate,
                 kl_weight=1,  # specify for VAE and CVAE
                 loss_fn="MSE",  # choose from MSE or WMSE, do not use WMSE if you do not know the weights
                 save_recons=False,  # wheter to save the reconstructed data
                 new_size=None,  # how many new samples you want to generate
                 save_new=False,  # whether to save the newly generated samples
                 plot=False):  # whether to plot the heatmaps of reconstructed and newly generated samples with the original ones

    set_all_seeds(random_seed)
    num_features = rawdata.shape[1]
    num_classes = rawlabels.shape[1]
    data = TensorDataset(rawdata, rawlabels)

    if modelname == "CVAE":
        model = CVAE(num_features, num_classes)
    elif modelname == "VAE":
        model = VAE(num_features)
    elif modelname == "AE":
        model = AE(num_features)
    else:
        raise ValueError("modelname is not supported by train_AEs funcion.")

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_loader = DataLoader(data, batch_size=batch_size, shuffle=True)

    if new_size is None:
        new_size = rawdata.shape[0]

    if modelname == "CVAE":
        log_dict = train_CVAE(num_epochs=num_epochs,
                              model=model,
                              loss_fn=loss_fn,
                              optimizer=optimizer,
                              train_loader=train_loader,
                              skip_epoch_stats=True,
                              reconstruction_term_weight=1,
                              kl_weight=kl_weight,
                              logging_interval=50,
                              save_model=None)
        plot_training_loss(log_dict['train_reconstruction_loss_per_batch'], num_epochs,
                           custom_label=" (reconstruction)")
        plt.show()
        plot_training_loss(log_dict['train_kl_loss_per_batch'], num_epochs, custom_label=" (KL)")
        plt.show()
        plot_training_loss(log_dict['train_combined_loss_per_batch'], num_epochs, custom_label=" (combined)")
        plt.show()

        if save_recons:
            # Plot generated data
            plot_recons_samples(savepath=savepath, data_loader=train_loader, model=model, n_features=num_features,
                                modelname="CVAE", plot=plot)
            plt.show()
        if save_new:
            # plot and save new generated data
            plot_new_samples(model=model, savepathnew=savepathnew, latent_size=34, modelname="CVAE",
                             num_images=new_size, plot=plot)
            plt.show()
    elif modelname == "VAE":
        log_dict = train_VAE(num_epochs=num_epochs,
                             model=model,
                             loss_fn=loss_fn,
                             optimizer=optimizer,
                             train_loader=train_loader,
                             skip_epoch_stats=True,
                             reconstruction_term_weight=1,
                             kl_weight=kl_weight,
                             logging_interval=50,
                             save_model=None)

        plot_training_loss(log_dict['train_reconstruction_loss_per_batch'], num_epochs,
                           custom_label=" (reconstruction)")
        plt.show()
        plot_training_loss(log_dict['train_kl_loss_per_batch'], num_epochs, custom_label=" (KL)")
        plt.show()
        plot_training_loss(log_dict['train_combined_loss_per_batch'], num_epochs, custom_label=" (combined)")
        plt.show()

        if save_recons:
            # Plot generated data
            plot_recons_samples(savepath=savepath, data_loader=train_loader, model=model, n_features=num_features,
                                modelname="VAE", plot=plot)
            plt.show()
        else:
            plot_recons_samples(savepath=None, data_loader=train_loader, model=model, n_features=num_features,
                                modelname="VAE", plot=plot)
            plt.show()
        if save_new:
            # plot and save new generated data
            plot_new_samples(model=model, savepathnew=savepathnew, latent_size=32, modelname="VAE", num_images=new_size,
                             plot=plot)
            plt.show()
        else:
            plot_new_samples(model=model, savepathnew=None, latent_size=32, modelname="VAE", num_images=new_size,
                             plot=plot)
            plt.show()


    else:
        log_dict = train_AE(num_epochs=num_epochs,
                            model=model,
                            loss_fn=loss_fn,
                            optimizer=optimizer,
                            train_loader=train_loader,
                            skip_epoch_stats=True,
                            logging_interval=50,
                            save_model=None)
        plot_training_loss(log_dict['train_loss_per_batch'], num_epochs, custom_label=" loss")
        plt.show()
        if save_recons:
            # Plot generated data
            plot_recons_samples(savepath=savepath, data_loader=train_loader, model=model, n_features=num_features,
                                modelname="AE", plot=plot)
            plt.show()
    return log_dict


def training_GANs(savepathnew,  # path to save newly generated samples
                  rawdata,  # raw data matrix with samples in row, features in column
                  rawlabels,  # labels for each sample, n_samples * 1, will not be used in AE or VAE
                  batch_size,  # batch size
                  random_seed,
                  modelname,  # choose from "GAN","WGAN","WGANGP"
                  num_epochs,
                  learning_rate,
                  new_size,  # how many new samples you want to generate
                  save_new=False,  # whether to save the newly generated samples
                  plot=False):  # whether to plot the heatmaps of reconstructed and newly generated samples with the original ones

    set_all_seeds(random_seed)
    num_features = rawdata.shape[1]
    data = TensorDataset(rawdata, rawlabels)
    train_loader = DataLoader(data, batch_size=batch_size, shuffle=True)
    latent_dim = 32

    model = GAN(num_features=num_features, latent_dim=latent_dim)

    optim_gen = torch.optim.Adam(model.generator.parameters(),
                                 betas=(0.5, 0.999),
                                 lr=learning_rate)
    optim_discr = torch.optim.Adam(model.discriminator.parameters(),
                                   betas=(0.5, 0.999),
                                   lr=learning_rate)

    if modelname == "GAN":
        log_dict = train_GAN(num_epochs=num_epochs,
                             model=model,
                             optimizer_gen=optim_gen,
                             optimizer_discr=optim_discr,
                             latent_dim=latent_dim,
                             train_loader=train_loader,
                             logging_interval=100,
                             save_model=None)
    elif modelname == "WGAN":
        log_dict = train_WGAN(num_epochs=num_epochs,
                              model=model,
                              optimizer_gen=optim_gen,
                              optimizer_discr=optim_discr,
                              latent_dim=latent_dim,
                              train_loader=train_loader,
                              logging_interval=100,
                              save_model=None)
    elif modelname == "WGANGP":
        log_dict = train_WGANGP(num_epochs=num_epochs,
                                model=model,
                                optimizer_gen=optim_gen,
                                optimizer_discr=optim_discr,
                                latent_dim=latent_dim,
                                train_loader=train_loader,
                                discr_iter_per_generator_iter=5,
                                logging_interval=100,
                                gradient_penalty=True,
                                gradient_penalty_weight=10,
                                save_model=None)

    plot_multiple_training_losses(
        losses_list=(log_dict['train_discriminator_loss_per_batch'],
                     log_dict['train_generator_loss_per_batch']),
        num_epochs=num_epochs,
        custom_labels_list=(' -- Discriminator', ' -- Generator')
    )
    plt.show()

    if save_new:
        # plot and save new generated data
        plot_new_samples(model=model, savepathnew=savepathnew, latent_size=latent_dim, modelname="GANs",
                         num_images=new_size, plot=plot)
        plt.show()
    else:
        plot_new_samples(model=model, savepathnew=None, latent_size=latent_dim, modelname="GANs", num_images=new_size,
                         plot=plot)
        plt.show()
    return log_dict


def training_iter(iter_times,  # how many times to iterative, will get pilot_size * 2^iter_times reconstructed samples
                  savepathextend,  # save final dataset
                  rawdata,  # pilot data
                  rawlabels,  # pilot labels
                  random_seed,
                  modelname,  # choose from AE, VAE
                  num_epochs,
                  learning_rate,
                  kl_weight=1,
                  loss_fn="MSE",
                  replace=False,
                  saveextend=True,
                  plot=False):
    set_all_seeds(random_seed)
    num_features = rawdata.shape[1]
    data = TensorDataset(rawdata, rawlabels)

    if modelname == "AE":
        model = AE(num_features)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        feed_data = rawdata
        feed_set = data
        for i in range(iter_times):
            batch_size = int(feed_data.shape[0] / 6)
            feed_loader = DataLoader(feed_set, batch_size=batch_size, shuffle=True)
            log_dict = train_AE(num_epochs=num_epochs,
                                model=model,
                                loss_fn=loss_fn,
                                optimizer=optimizer,
                                train_loader=feed_loader,
                                skip_epoch_stats=True,
                                logging_interval=50,
                                save_model=None)
            # Loss 
            plot_training_loss(log_dict['train_loss_per_batch'], num_epochs, custom_label=" (combined)")
            plt.show()
            feed_data_gen, feed_labels = plot_recons_samples(savepath=None, batch_size=batch_size,
                                                             data_loader=feed_loader, model=model,
                                                             n_features=num_features, modelname="AE", plot=plot)
            print(feed_data_gen.shape)
            if replace:
                new_sample_range = range(int(feed_data_gen.shape[0] / 2), feed_data_gen.shape[0])
                num_failures = 0
                for i_feature in range(feed_data_gen.shape[1]):
                    if (torch.std(feed_data_gen[new_sample_range, i_feature]) == 0) & (
                            torch.mean(feed_data_gen[new_sample_range, i_feature]) == 0):
                        feed_data_gen[new_sample_range, i_feature] = feed_data[:, i_feature]
                        num_failures += 1
                print("replace " + str(num_failures) + " zero features")
            feed_data = feed_data_gen
            feed_set = TensorDataset(feed_data, feed_labels)


    elif modelname == "VAE":
        model = VAE(num_features)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        feed_data = rawdata
        feed_set = data
        for i in range(iter_times):
            batch_size = int(feed_data.shape[0] / 6)
            feed_loader = DataLoader(feed_set, batch_size=batch_size, shuffle=True)
            log_dict = train_VAE(num_epochs=num_epochs,
                                 model=model,
                                 loss_fn=loss_fn,
                                 optimizer=optimizer,
                                 train_loader=feed_loader,
                                 skip_epoch_stats=True,
                                 reconstruction_term_weight=1,
                                 kl_weight=kl_weight,
                                 logging_interval=50,
                                 save_model=None)

            # Loss 
            plot_training_loss(log_dict['train_reconstruction_loss_per_batch'], num_epochs,
                               custom_label=" (reconstruction)")
            plot_training_loss(log_dict['train_kl_loss_per_batch'], num_epochs, custom_label=" (KL)")
            plot_training_loss(log_dict['train_combined_loss_per_batch'], num_epochs, custom_label=" (combined)")
            plt.show()
            feed_data_gen, feed_labels = plot_recons_samples(savepath=None, batch_size=batch_size,
                                                             data_loader=feed_loader, model=model,
                                                             n_features=num_features, modelname="VAE", plot=plot)
            print(feed_data_gen.shape)
            if replace:
                new_sample_range = range(int(feed_data_gen.shape[0] / 2), feed_data_gen.shape[0])
                num_failures = 0
                for i_feature in range(feed_data_gen.shape[1]):
                    if (torch.std(feed_data_gen[new_sample_range, i_feature]) == 0) & (
                            torch.mean(feed_data_gen[new_sample_range, i_feature]) == 0):
                        feed_data_gen[new_sample_range, i_feature] = feed_data[:, i_feature]
                        num_failures += 1
                print("replace " + str(num_failures) + " zero features")
            feed_data = feed_data_gen
            feed_set = TensorDataset(feed_data, feed_labels)
    if saveextend:
        np.savetxt(savepathextend, torch.cat((feed_data, feed_labels), dim=1).detach().numpy(), delimiter=",")

    return feed_data, feed_labels


# %%

def training_flows(savepathnew, rawdata, batch_size, random_seed,
                   modelname,
                   num_blocks,
                   num_epoches,
                   learning_rate,
                   new_size,
                   num_hidden,
                   plot=False):
    set_all_seeds(random_seed)
    device = torch.device("cpu")
    num_inputs = rawdata.shape[1]
    data = TensorDataset(rawdata)
    train_loader = DataLoader(data, batch_size=batch_size, shuffle=True)

    act = 'tanh'

    modules = []
    if modelname == 'glow':
        mask = torch.arange(0, num_inputs) % 2
        mask = mask.to(device).float()

        for _ in range(num_blocks):
            modules += [
                fnn.BatchNormFlow(num_inputs),
                fnn.LUInvertibleMM(num_inputs),
                fnn.CouplingLayer(
                    num_inputs, num_hidden, mask, num_cond_inputs=None,
                    s_act='tanh', t_act='relu')
            ]
            mask = 1 - mask
    elif modelname == 'realnvp':
        mask = torch.arange(0, num_inputs) % 2
        mask = mask.to(device).float()

        for _ in range(num_blocks):
            modules += [
                fnn.CouplingLayer(
                    num_inputs, num_hidden, mask, num_cond_inputs=None,
                    s_act='tanh', t_act='relu'),
                fnn.BatchNormFlow(num_inputs)
            ]
            mask = 1 - mask
    elif modelname == 'maf':
        for _ in range(num_blocks):
            modules += [
                fnn.MADE(num_inputs, num_hidden, num_cond_inputs=None, act=act),
                fnn.BatchNormFlow(num_inputs),
                fnn.Reverse(num_inputs)
            ]
    elif modelname == 'maf-split':
        for _ in range(num_blocks):
            modules += [
                fnn.MADESplit(num_inputs, num_hidden, num_cond_inputs=None,
                              s_act='tanh', t_act='relu'),
                fnn.BatchNormFlow(num_inputs),
                fnn.Reverse(num_inputs)
            ]
    elif modelname == 'maf-split-glow':
        for _ in range(num_blocks):
            modules += [
                fnn.MADESplit(num_inputs, num_hidden, num_cond_inputs=None,
                              s_act='tanh', t_act='relu'),
                fnn.BatchNormFlow(num_inputs),
                fnn.InvertibleMM(num_inputs)
            ]

    model = fnn.FlowSequential(*modules)

    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight)
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data.fill_(0)

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-6)
    writer = SummaryWriter(comment=modelname)
    global_step = 0

    def train(epoch, global_step, writer):
        # global global_step, writer
        model.train()
        train_loss = 0
        # samples = np.empty(shape=298)

        pbar = tqdm(total=len(train_loader.dataset))
        for batch_idx, data in enumerate(train_loader):
            if isinstance(data, list):
                if len(data) > 1:
                    cond_data = data[1].float()
                    cond_data = cond_data.to(device)
                else:
                    cond_data = None

                data = data[0]
            # import pdb
            # pdb.set_trace()
            data = data.to(device)
            optimizer.zero_grad()
            loss = -model.log_probs(data, cond_data).mean()
            train_loss += loss.item()
            # samples = np.vstack((samples, (model.sample(42).detach().numpy().reshape(42, -1))))
            # print(samples)
            loss.backward()
            optimizer.step()

            pbar.update(data.size(0))
            pbar.set_description('Train, Log likelihood in nats: {:.6f}'.format(
                -train_loss / (batch_idx + 1)))

            writer.add_scalar('training/loss', loss.item(), global_step)
            global_step += 1


        # mysamples = pow(2, np.array(samples)) - 1
        # save_file_path = 'outputs/SKCM/MAF/epoch_%d.txt' % (global_step)
        # print((mysamples[1:453, ]).shape)
        # np.savetxt(save_file_path, mysamples[1:453, ])

        pbar.close()

        for module in model.modules():
            if isinstance(module, fnn.BatchNormFlow):
                module.momentum = 0

            with torch.no_grad():
                model(train_loader.dataset.tensors[0].to(data.device))

        for module in model.modules():
            if isinstance(module, fnn.BatchNormFlow):
                module.momentum = 1

        return global_step

    for epoch in range(num_epoches):
        print('\nEpoch: {}'.format(epoch))

        global_step = train(epoch, global_step, writer)

    # plot and save new generated data
    plot_new_samples(model=model, savepathnew=savepathnew, latent_size=num_hidden, modelname=modelname,
                     num_images=new_size, plot=plot)
    plt.show()
