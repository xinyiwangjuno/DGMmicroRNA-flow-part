#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 15:22:40 2022

@author: yunhui
"""
import torch 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import random



def preprocessinglog2(dataset):
    return torch.log2(dataset+1)


def set_all_seeds(seed):
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_labels(n_samples, groups=None):
    set_all_seeds(10)
    if (groups is None):
        labels = torch.zeros([n_samples, 1])
        blurlabels = labels
    else:
        base = groups[0]
        labels = torch.zeros([n_samples, 1]).to(torch.float32)
        labels[groups != base, 0] = 1
        blurlabels = torch.zeros([n_samples, 1]).to(torch.float32)
        blurlabels[groups != base, 0] = (10 - 9) * torch.rand(sum(groups != base)) + 9
        blurlabels[groups == base, 0] = (1 - 0) * torch.rand(sum(groups == base)) + 0
    return labels, blurlabels


def draw_pilot(dataset, labels, blurlabels, n_pilot, seednum):
    set_all_seeds(seednum)
    n_samples = dataset.shape[0]
    if torch.unique(labels).shape[0] == 1:
        shuffled_indices = torch.randperm(n_samples)
        pilot_indices = shuffled_indices[-n_pilot:]
        rawdata = dataset[pilot_indices, :]
        rawlabels = labels[pilot_indices, :]
        rawblurlabels = blurlabels[pilot_indices, :]
    else:
        base = labels[0, :]
        n_pilot_1 = round(n_pilot / 2)
        n_pilot_2 = n_pilot - n_pilot_1
        n_samples_1 = sum(labels[:, 0] == base)
        n_samples_2 = sum(labels[:, 0] != base)
        dataset_1 = dataset[labels[:, 0] == base, :]
        dataset_2 = dataset[labels[:, 0] != base, :]
        labels_1 = labels[labels[:, 0] == base, :]
        labels_2 = labels[labels[:, 0] != base, :]
        blurlabels_1 = blurlabels[labels[:, 0] == base, :]
        blurlabels_2 = blurlabels[labels[:, 0] != base, :]
        shuffled_indices_1 = torch.randperm(n_samples_1)
        pilot_indices_1 = shuffled_indices_1[-n_pilot_1:]
        rawdata_1 = dataset_1[pilot_indices_1, :]
        rawlabels_1 = labels_1[pilot_indices_1, :]
        rawblurlabels_1 = blurlabels_1[pilot_indices_1, :]
        shuffled_indices_2 = torch.randperm(n_samples_2)
        pilot_indices_2 = shuffled_indices_2[-n_pilot_2:]
        rawdata_2 = dataset_2[pilot_indices_2, :]
        rawlabels_2 = labels_2[pilot_indices_2, :]
        rawblurlabels_2 = blurlabels_2[pilot_indices_2, :]
        rawdata = torch.cat((rawdata_1, rawdata_2), dim=0)
        rawlabels = torch.cat((rawlabels_1, rawlabels_2), dim=0)
        rawblurlabels = torch.cat((rawblurlabels_1, rawblurlabels_2), dim=0)
    return rawdata,rawlabels,rawblurlabels


def plot_training_loss(minibatch_losses, num_epochs, averaging_iterations = 100, custom_label = ''):

    iter_per_epoch = len(minibatch_losses) // num_epochs

    plt.figure()
    ax1 = plt.subplot(1, 1, 1)
    ax1.plot(range(len(minibatch_losses)),
             (minibatch_losses), label=f'Minibatch Loss{custom_label}')
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Loss')

    if len(minibatch_losses) < 1000:
        num_losses = len(minibatch_losses) // 2
    else:
        num_losses = 1000

    ax1.set_ylim([
        0, np.max(minibatch_losses[num_losses:])*1.5
        ])

    ax1.plot(np.convolve(minibatch_losses,
                         np.ones(averaging_iterations,)/averaging_iterations,
                         mode='valid'),
             label=f'Running Average{custom_label}')
    ax1.legend()

    ###################
    # Set scond x-axis
    ax2 = ax1.twiny()
    newlabel = list(range(num_epochs+1))

    newpos = [e*iter_per_epoch for e in newlabel]

    ax2.set_xticks(newpos[::10])
    ax2.set_xticklabels(newlabel[::10])

    ax2.xaxis.set_ticks_position('bottom')
    ax2.xaxis.set_label_position('bottom')
    ax2.spines['bottom'].set_position(('outward', 45))
    ax2.set_xlabel('Epochs')
    ax2.set_xlim(ax1.get_xlim())
    ###################

    plt.tight_layout()



def plot_recons_samples(savepath, model, modelname, data_loader, batch_size, n_features, plot = False):

    orig_all = torch.zeros([1,n_features])
    decoded_all = torch.zeros([1,n_features])
    labels = torch.zeros(1)+3
    
    for batch_idx, (features,lab) in enumerate(data_loader):
        if modelname == "CVAE":
            labels=torch.cat((labels,lab[:,0]), dim = 0)
            with torch.no_grad():
                encoded, z_mean, z_log_var, decoded_images = model(features,lab)[:batch_size]
        elif modelname == "VAE":
            labels=torch.cat((labels,lab[:,0]), dim = 0)        
            with torch.no_grad():
                encoded, z_mean, z_log_var, decoded_images = model(features)[:batch_size]
        else:
            labels=torch.cat((labels,lab[:,0]), dim = 0) 
            with torch.no_grad():
                encoded, decoded_images = model(features)[:batch_size]
     
        orig_all = torch.cat((orig_all,features[:batch_size]), dim = 0)
        decoded_all = torch.cat((decoded_all,decoded_images), dim = 0)
    
            
    orig_all = orig_all[1:]
    decoded_all = decoded_all[1:]
    labels = labels[1:]
        
    if modelname == "CVAE":
        orig_all = torch.cat((orig_all,torch.unsqueeze(labels,1)), dim = 1)
        decoded_all = torch.cat((decoded_all,torch.unsqueeze(labels,1)), dim = 1)
    if plot:
        sns.heatmap(torch.cat((orig_all,decoded_all), dim = 0).detach().numpy(), cmap = "YlGnBu")
        plt.show()
    
    if savepath is not None:
        np.savetxt(savepath, torch.cat((orig_all,decoded_all), dim = 0).detach().numpy(), delimiter = ",")
    else:
        return torch.cat((orig_all,decoded_all),dim = 0).detach(), torch.cat((torch.unsqueeze(labels,1), torch.unsqueeze(labels,1)), dim = 0)

                
                
# def plot_latent_space_with_labels(num_classes, data_loader, encoding_fn):
#     d = {i:[] for i in range(num_classes)}

#     with torch.no_grad():
#         for i, (features,targets) in enumerate(data_loader):
#             embedding = encoding_fn(features)
#             for i in range(num_classes):
#                 if i in targets:
#                     mask = targets == i
#                     d[i].append(embedding[mask].numpy())

#     colors = list(mcolors.TABLEAU_COLORS.items())
#     for i in range(num_classes):
#         d[i] = np.concatenate(d[i])
#         plt.scatter(
#             d[i][:, 0], d[i][:, 1],
#             color=colors[i][1],
#             label=f'{i}',
#             alpha=0.5)

#     plt.legend()
    
    
def plot_new_samples(model, modelname, savepathnew, latent_size, num_images, plot = False):

    with torch.no_grad():

        ##########################
        ### RANDOM SAMPLE
        ##########################    

        rand_features = torch.randn(num_images, latent_size)
        if modelname == "CVAE":
            labels = torch.zeros(num_images,2)
            labels[:int(num_images/2),0] = 1
            labels[int(num_images/2):,1] = 1
            rand_features = torch.cat((rand_features,labels), dim = 1)
            new_images = model.decoder(rand_features)
            new_labels = torch.zeros(num_images)
            new_labels[:int(num_images/2)] = 20
            new_images = torch.cat((new_images, torch.unsqueeze(new_labels,1)), dim = 1)
        elif modelname == "AE":
            new_images = model.decoder(rand_features)
        elif modelname == "VAE":
            new_images = model.decoder(rand_features)
        elif modelname == "GANs":
            new_images = model.generator(rand_features)
        elif modelname == 'glow':
            new_images = model.sample(num_images)
        elif modelname == 'realnvp':
            new_images = model.sample(num_images)
        elif modelname == 'maf':
            new_images = model.sample(num_images)

        ##########################
        ### VISUALIZATION
        ##########################
        # last column of saved data is the labels: 0 for MXF, 20 for PMFH
        # either generated for VAE or setted for CVAE
        if plot:
            sns.heatmap(new_images.detach().numpy(), cmap = "YlGnBu")
            plt.show()
        
        if savepathnew is not None:

            np.savetxt(savepathnew, new_images.detach().numpy(), delimiter = ",")
            
            
            
def plot_multiple_training_losses(losses_list,
                                  num_epochs, 
                                  averaging_iterations=100, 
                                  custom_labels_list=None):

    for i,_ in enumerate(losses_list):
        if not len(losses_list[i]) == len(losses_list[0]):
            raise ValueError('All loss tensors need to have the same number of elements.')
    
    if custom_labels_list is None:
        custom_labels_list = [str(i) for i,_ in enumerate(custom_labels_list)]
    
    iter_per_epoch = len(losses_list[0]) // num_epochs

    plt.figure()
    ax1 = plt.subplot(1, 1, 1)
    
    for i, minibatch_loss_tensor in enumerate(losses_list):
        ax1.plot(range(len(minibatch_loss_tensor)),
                 (minibatch_loss_tensor),
                  label=f'Minibatch Loss{custom_labels_list[i]}')
        ax1.set_xlabel('Iterations')
        ax1.set_ylabel('Loss')

        ax1.plot(np.convolve(minibatch_loss_tensor,
                             np.ones(averaging_iterations,)/averaging_iterations,
                             mode='valid'),
                 color='black')
    
    if len(losses_list[0]) < 1000:
        num_losses = len(losses_list[0]) // 2
    else:
        num_losses = 1000
    maxes = [np.max(losses_list[i][num_losses:]) for i,_ in enumerate(losses_list)]
    ax1.set_ylim([0, np.max(maxes)*1.5])
    ax1.legend()

    ###################
    # Set scond x-axis
    ax2 = ax1.twiny()
    newlabel = list(range(num_epochs+1))

    newpos = [e*iter_per_epoch for e in newlabel]

    ax2.set_xticks(newpos[::10])
    ax2.set_xticklabels(newlabel[::10])

    ax2.xaxis.set_ticks_position('bottom')
    ax2.xaxis.set_label_position('bottom')
    ax2.spines['bottom'].set_position(('outward', 45))
    ax2.set_xlabel('Epochs')
    ax2.set_xlim(ax1.get_xlim())
    ###################

    plt.tight_layout()
    