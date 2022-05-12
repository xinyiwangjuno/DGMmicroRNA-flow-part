#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 15:04:48 2022

@author: yunhui
"""

import torch
import time
import torch.nn.functional as F



#%%

def compute_epoch_loss_autoencoder(model, data_loader, loss_fn):
    model.eval()
    curr_loss, num_examples = 0., 0
    with torch.no_grad():
        for features, _ in data_loader:
            logits = model(features)
            loss = loss_fn(logits, features, reduction='sum')
            num_examples += features.size(0)
            curr_loss += loss

        curr_loss = curr_loss / num_examples
    return curr_loss
    
def WMSE(output, target, reduction = "none"):
    weights = 1/(torch.mean(target,0)+1)
    loss = weights*(output - target)**2
    return loss

#%%
def train_AE(num_epochs, 
             model,
             optimizer, 
             train_loader,
             loss_fn = "MSE",
             logging_interval = 100, 
             skip_epoch_stats = False,
             save_model = None):
    
    log_dict = {'train_loss_per_batch': [],
                'train_combined_loss_per_epoch': []}

    if loss_fn == "MSE":
        loss_fn = F.mse_loss
    elif loss_fn == "WMSE":
        loss_fn = WMSE
        

    start_time = time.time()
    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (features,_) in enumerate(train_loader):
            # FORWARD AND BACK PROP
            encoded,  decoded = model(features)

            batchsize = features.shape[0]
            pixelwise = loss_fn(decoded, features, reduction='none')
            pixelwise = pixelwise.view(batchsize, -1).sum(axis=1) # sum over pixels
            pixelwise = pixelwise.mean() # average over batch dimension
                
            loss = pixelwise 
                
            optimizer.zero_grad()
    
            loss.backward()
    
            # UPDATE MODEL PARAMETERS
            optimizer.step()
    
            # LOGGING
            log_dict['train_loss_per_batch'].append(pixelwise.item())                
                
            print('Epoch: %03d/%03d | Batch %04d/%04d | Loss: %.4f'
                      % (epoch+1, num_epochs, batch_idx,
                         len(train_loader), loss))
    
        if not skip_epoch_stats:
            model.eval()
                
            with torch.set_grad_enabled(False):  # save memory during inference
                    
                train_loss = compute_epoch_loss_autoencoder(
                        model, train_loader, loss_fn)
                print('***Epoch: %03d/%03d | Loss: %.3f' % (
                          epoch+1, num_epochs, train_loss))
                log_dict['train_combined_loss_per_epoch'].append(train_loss.item())
    
        print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))
    
    print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))
    if save_model is not None:
        torch.save(model.state_dict(), save_model)
        
    return log_dict
#%%

def train_VAE(num_epochs, 
              model, 
              optimizer, 
              train_loader, 
              loss_fn = "MSE",
              logging_interval = 100, 
              skip_epoch_stats = False,
              reconstruction_term_weight = 1,
              kl_weight = 2,
              save_model = None):
    
    log_dict = {'train_combined_loss_per_batch': [],
                'train_combined_loss_per_epoch': [],
                'train_reconstruction_loss_per_batch': [],
                'train_kl_loss_per_batch': []}

    if loss_fn == "MSE":
        loss_fn = F.mse_loss
    elif loss_fn == "WMSE":
        loss_fn = WMSE
        

    start_time = time.time()
    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (features,_) in enumerate(train_loader):
            # FORWARD AND BACK PROP
            encoded, z_mean, z_log_var, decoded = model(features)
                
            # total loss = reconstruction loss + KL divergence
            #kl_divergence = (0.5 * (z_mean**2 + 
            #                        torch.exp(z_log_var) - z_log_var - 1)).sum()
            kl_div = -0.5 * torch.sum(1 + z_log_var 
                                  - z_mean**2 
                                  - torch.exp(z_log_var), 
                                  axis=1) # sum over latent dimension
    
            batchsize = kl_div.size(0)
            kl_div = kl_div.mean() # average over batch dimension
        
            pixelwise = loss_fn(decoded, features, reduction='none')
            pixelwise = pixelwise.view(batchsize, -1).sum(axis=1) # sum over pixels
            pixelwise = pixelwise.mean() # average over batch dimension
                
            loss = reconstruction_term_weight*pixelwise + kl_weight*kl_div
                
            optimizer.zero_grad()
    
            loss.backward()
    
            # UPDATE MODEL PARAMETERS
            optimizer.step()
    
            # LOGGING
            log_dict['train_combined_loss_per_batch'].append(loss.item())
            log_dict['train_reconstruction_loss_per_batch'].append(pixelwise.item())
            log_dict['train_kl_loss_per_batch'].append(kl_div.item())
                
                
            print('Epoch: %03d/%03d | Batch %04d/%04d | Loss: %.4f'
                      % (epoch+1, num_epochs, batch_idx,
                         len(train_loader), loss))
    
        if not skip_epoch_stats:
            model.eval()
                
            with torch.set_grad_enabled(False):  # save memory during inference
                    
                train_loss = compute_epoch_loss_autoencoder(
                        model, train_loader, loss_fn)
                print('***Epoch: %03d/%03d | Loss: %.3f' % (
                          epoch+1, num_epochs, train_loss))
                log_dict['train_combined_loss_per_epoch'].append(train_loss.item())
    
        print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))
    
    print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))
    if save_model is not None:
        torch.save(model.state_dict(), save_model)
    
    return log_dict


#%%


def train_CVAE(num_epochs,
               model, 
               optimizer, 
               train_loader, 
               loss_fn = "MSE",
               logging_interval = 100, 
               skip_epoch_stats = False,
               reconstruction_term_weight = 1,
               kl_weight = 1,
               save_model = None):
    
    log_dict = {'train_combined_loss_per_batch': [],
                'train_combined_loss_per_epoch': [],
                'train_reconstruction_loss_per_batch': [],
                'train_kl_loss_per_batch': []}

    if loss_fn == "MSE":
        loss_fn = F.mse_loss
    elif loss_fn == "WMSE":
        loss_fn = WMSE
        

    start_time = time.time()
    for epoch in range(num_epochs):

        model.train()
        for batch_idx, (features, lab) in enumerate(train_loader):


                # FORWARD AND BACK PROP
            encoded, z_mean, z_log_var, decoded = model(features,lab)
            
                # total loss = reconstruction loss + KL divergence
                #kl_divergence = (0.5 * (z_mean**2 + 
                #                        torch.exp(z_log_var) - z_log_var - 1)).sum()
            kl_div = -0.5 * torch.sum(1 + z_log_var 
                                          - z_mean**2 
                                          - torch.exp(z_log_var), 
                                          axis=1) # sum over latent dimension

            batchsize = kl_div.size(0)
            kl_div = kl_div.mean() # average over batch dimension
    
            pixelwise = loss_fn(decoded, features, reduction='none')
            pixelwise = pixelwise.view(batchsize, -1).sum(axis=1) # sum over pixels
            pixelwise = pixelwise.mean() # average over batch dimension
            
            loss = reconstruction_term_weight*pixelwise + kl_weight*kl_div
            
            optimizer.zero_grad()

            loss.backward()

            # UPDATE MODEL PARAMETERS
            optimizer.step()

            # LOGGING
            log_dict['train_combined_loss_per_batch'].append(loss.item())
            log_dict['train_reconstruction_loss_per_batch'].append(pixelwise.item())
            log_dict['train_kl_loss_per_batch'].append(kl_div.item())
            
            
            print('Epoch: %03d/%03d | Batch %04d/%04d | Loss: %.4f'
                      % (epoch+1, num_epochs, batch_idx,
                         len(train_loader), loss))

        if not skip_epoch_stats:
            model.eval()
            
            with torch.set_grad_enabled(False):  # save memory during inference
                
                 train_loss = compute_epoch_loss_autoencoder(
                        model, train_loader, loss_fn)
                 print('***Epoch: %03d/%03d | Loss: %.3f' % (
                        epoch+1, num_epochs, train_loss))
                 log_dict['train_combined_loss_per_epoch'].append(train_loss.item())

        print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))
    
    print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))

    if save_model is not None:
        torch.save(model.state_dict(), save_model)
    
    return log_dict

#%% 
def train_GAN(num_epochs,
              model, 
              optimizer_gen,
              optimizer_discr, 
              latent_dim, 
              train_loader, 
              logging_interval = 100, 
              save_model = None):
    
    log_dict = {'train_generator_loss_per_batch': [],
                'train_discriminator_loss_per_batch': [],
                'train_discriminator_real_acc_per_batch': [],
                'train_discriminator_fake_acc_per_batch': []}

    loss_fn = F.binary_cross_entropy_with_logits


    start_time = time.time()
    for epoch in range(num_epochs):

        model.train()
        for batch_idx, (features, _) in enumerate(train_loader):

            batch_size = features.size(0)

            # real images
            real_images = features
            real_labels = torch.ones(batch_size) # real label = 1

            # generated (fake) images
            noise = torch.randn(batch_size, latent_dim)  
            fake_images = model.generator_forward(noise)
            fake_labels = torch.zeros(batch_size) # fake label = 0
            flipped_fake_labels = real_labels # here, fake label = 1

            # --------------------------
            # Train Discriminator
            # --------------------------

            optimizer_discr.zero_grad()

            # get discriminator loss on real images
            discr_pred_real = model.discriminator_forward(real_images).view(-1) # Nx1 -> N
            real_loss = loss_fn(discr_pred_real, real_labels)
            # real_loss.backward()

            # get discriminator loss on fake images
            discr_pred_fake = model.discriminator_forward(fake_images.detach()).view(-1)
            fake_loss = loss_fn(discr_pred_fake, fake_labels)
            # fake_loss.backward()

            # combined loss
            discr_loss = 0.5*(real_loss + fake_loss)
            discr_loss.backward()

            optimizer_discr.step()

            # --------------------------
            # Train Generator
            # --------------------------

            optimizer_gen.zero_grad()

            # get discriminator loss on fake images with flipped labels
            discr_pred_fake = model.discriminator_forward(fake_images).view(-1)
            gener_loss = loss_fn(discr_pred_fake, flipped_fake_labels)
            gener_loss.backward()

            optimizer_gen.step()

            # --------------------------
            # Logging
            # --------------------------   
            log_dict['train_generator_loss_per_batch'].append(gener_loss.item())
            log_dict['train_discriminator_loss_per_batch'].append(discr_loss.item())
            
            predicted_labels_real = torch.where(discr_pred_real.detach() > 0., 1., 0.)
            predicted_labels_fake = torch.where(discr_pred_fake.detach() > 0., 1., 0.)
            acc_real = (predicted_labels_real == real_labels).float().mean()*100.
            acc_fake = (predicted_labels_fake == fake_labels).float().mean()*100.
            log_dict['train_discriminator_real_acc_per_batch'].append(acc_real.item())
            log_dict['train_discriminator_fake_acc_per_batch'].append(acc_fake.item())         
            
            if not batch_idx % logging_interval:
                print('Epoch: %03d/%03d | Batch %03d/%03d | Gen/Dis Loss: %.4f/%.4f' 
                       % (epoch+1, num_epochs, batch_idx, 
                          len(train_loader), gener_loss.item(), discr_loss.item()))


        print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))
    
    print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))
    
    if save_model is not None:
        torch.save(model.state_dict(), save_model)
    
    return log_dict

#%%
def train_WGAN(num_epochs, 
               model, 
               optimizer_gen,
               optimizer_discr, 
               latent_dim, 
               train_loader, 
               logging_interval = 100, 
               save_model = None):
    
    log_dict = {'train_generator_loss_per_batch': [],
                'train_discriminator_loss_per_batch': [],
                'train_discriminator_real_acc_per_batch': [],
                'train_discriminator_fake_acc_per_batch': []}

    # if loss == 'regular':
    #     loss_fn = F.binary_cross_entropy_with_logits
    # elif loss == 'wasserstein':
    #     def loss_fn(y_pred, y_true):
    #         return -torch.mean(y_pred * y_true)
    # else:
    #     raise ValueError('This loss is not supported.')
    def loss_fn(y_pred, y_true):
        return -torch.mean(y_pred * y_true)

    start_time = time.time()
    for epoch in range(num_epochs):

        model.train()
        for batch_idx, (features, _) in enumerate(train_loader):

            batch_size = features.size(0)

            # real images
            real_images = features
            real_labels = torch.ones(batch_size) # real label = 1

            # generated (fake) images
            noise = torch.randn(batch_size, latent_dim)  
            fake_images = model.generator_forward(noise)
            
            # if loss == 'regular':
            #     fake_labels = torch.zeros(batch_size) # fake label = 0
            # elif loss == 'wasserstein':
            #     fake_labels = -real_labels # fake label = -1    
            fake_labels = -real_labels # fake label = -1    
            flipped_fake_labels = real_labels # here, fake label = 1

            # --------------------------
            # Train Discriminator
            # --------------------------

            optimizer_discr.zero_grad()

            # get discriminator loss on real images
            discr_pred_real = model.discriminator_forward(real_images).view(-1) # Nx1 -> N
            real_loss = loss_fn(discr_pred_real, real_labels)
            # real_loss.backward()

            # get discriminator loss on fake images
            discr_pred_fake = model.discriminator_forward(fake_images.detach()).view(-1)
            fake_loss = loss_fn(discr_pred_fake, fake_labels)
            # fake_loss.backward()

            # combined loss
            discr_loss = 0.5*(real_loss + fake_loss)
            discr_loss.backward()

            optimizer_discr.step()
            
            # if loss == 'wasserstein':
            #     for p in model.discriminator.parameters():
            #         p.data.clamp_(-0.01, 0.01)
            for p in model.discriminator.parameters():
                p.data.clamp_(-0.01, 0.01)
            # --------------------------
            # Train Generator
            # --------------------------

            optimizer_gen.zero_grad()

            # get discriminator loss on fake images with flipped labels
            discr_pred_fake = model.discriminator_forward(fake_images).view(-1)
            gener_loss = loss_fn(discr_pred_fake, flipped_fake_labels)
            gener_loss.backward()

            optimizer_gen.step()

            # --------------------------
            # Logging
            # --------------------------   
            log_dict['train_generator_loss_per_batch'].append(gener_loss.item())
            log_dict['train_discriminator_loss_per_batch'].append(discr_loss.item())
            
            predicted_labels_real = torch.where(discr_pred_real.detach() > 0., 1., 0.)
            predicted_labels_fake = torch.where(discr_pred_fake.detach() > 0., 1., 0.)
            acc_real = (predicted_labels_real == real_labels).float().mean()*100.
            acc_fake = (predicted_labels_fake == fake_labels).float().mean()*100.
            log_dict['train_discriminator_real_acc_per_batch'].append(acc_real.item())
            log_dict['train_discriminator_fake_acc_per_batch'].append(acc_fake.item())         
            
            if not batch_idx % logging_interval:
                print('Epoch: %03d/%03d | Batch %03d/%03d | Gen/Dis Loss: %.4f/%.4f' 
                       % (epoch+1, num_epochs, batch_idx, 
                          len(train_loader), gener_loss.item(), discr_loss.item()))


        print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))
    
    print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))
    
    if save_model is not None:
        torch.save(model.state_dict(), save_model)
    
    return log_dict


def train_WGANGP(num_epochs, 
                 model, 
                 optimizer_gen, 
                 optimizer_discr, 
                 latent_dim, 
                 train_loader,
                 discr_iter_per_generator_iter = 5,
                 logging_interval = 100, 
                 gradient_penalty = True,
                 gradient_penalty_weight = 10,
                 save_model = None):
    
    log_dict = {'train_generator_loss_per_batch': [],
                'train_discriminator_loss_per_batch': [],
                'train_discriminator_real_acc_per_batch': [],
                'train_discriminator_fake_acc_per_batch': []}

    if gradient_penalty:
        log_dict['train_gradient_penalty_loss_per_batch'] = []

    def loss_fn(y_pred, y_true):
        return -torch.mean(y_pred * y_true)

    start_time = time.time()
    
    
    skip_generator = 1
    for epoch in range(num_epochs):

        model.train()
        
        for batch_idx, (features, _) in enumerate(train_loader):

            batch_size = features.size(0)

            # real images
            real_images = features
            real_labels = torch.ones(batch_size) # real label = 1

            # generated (fake) images
            noise = torch.randn(batch_size, latent_dim)  # format NCHW
            fake_images = model.generator_forward(noise)
            
            fake_labels = -real_labels # fake label = -1    
            flipped_fake_labels = real_labels # here, fake label = 1

    
            # --------------------------
            # Train Discriminator
            # --------------------------

            optimizer_discr.zero_grad()

            # get discriminator loss on real images
            discr_pred_real = model.discriminator_forward(real_images).view(-1) # Nx1 -> N
            real_loss = loss_fn(discr_pred_real, real_labels)
            # real_loss.backward()

            # get discriminator loss on fake images
            discr_pred_fake = model.discriminator_forward(fake_images.detach()).view(-1)
            fake_loss = loss_fn(discr_pred_fake, fake_labels)
            # fake_loss.backward()
            
            # combined loss
            discr_loss = 0.5*(real_loss + fake_loss)

            ###################################################
            # gradient penalty
            if gradient_penalty:

                alpha = torch.rand(batch_size, 1, 1, 1)

                interpolated = alpha * real_images + (1 - alpha) * fake_images.detach()
                interpolated.requires_grad = True

                discr_out = model.discriminator_forward(interpolated)

                grad_values = torch.ones(discr_out.size())
                gradients = torch.autograd.grad(
                    outputs=discr_out,
                    inputs=interpolated,
                    grad_outputs=grad_values,
                    create_graph=True,
                    retain_graph=True)[0]

                gradients = gradients.view(batch_size, -1)

                # calc. norm of gradients, adding epsilon to prevent 0 values
                epsilon = 1e-13
                gradients_norm = torch.sqrt(
                    torch.sum(gradients ** 2, dim=1) + epsilon)

                gp_penalty_term = ((gradients_norm - 1) ** 2).mean() * gradient_penalty_weight
                discr_loss += gp_penalty_term
                
                log_dict['train_gradient_penalty_loss_per_batch'].append(gp_penalty_term.item())
            #######################################################
            
            discr_loss.backward()

            optimizer_discr.step()
            
            # Use weight clipping (standard Wasserstein GAN)
            if not gradient_penalty:
                for p in model.discriminator.parameters():
                    p.data.clamp_(-0.01, 0.01)

            
            if skip_generator <= discr_iter_per_generator_iter:
                
                # --------------------------
                # Train Generator
                # --------------------------

                optimizer_gen.zero_grad()

                # get discriminator loss on fake images with flipped labels
                discr_pred_fake = model.discriminator_forward(fake_images).view(-1)
                gener_loss = loss_fn(discr_pred_fake, flipped_fake_labels)
                gener_loss.backward()

                optimizer_gen.step()
                
                skip_generator += 1
                
            else:
                skip_generator = 1
                gener_loss = torch.tensor(0.)

            # --------------------------
            # Logging
            # --------------------------   
            log_dict['train_generator_loss_per_batch'].append(gener_loss.item())
            log_dict['train_discriminator_loss_per_batch'].append(discr_loss.item())
            
            predicted_labels_real = torch.where(discr_pred_real.detach() > 0., 1., 0.)
            predicted_labels_fake = torch.where(discr_pred_fake.detach() > 0., 1., 0.)
            acc_real = (predicted_labels_real == real_labels).float().mean()*100.
            acc_fake = (predicted_labels_fake == fake_labels).float().mean()*100.
            log_dict['train_discriminator_real_acc_per_batch'].append(acc_real.item())
            log_dict['train_discriminator_fake_acc_per_batch'].append(acc_fake.item())         
            
            if not batch_idx % logging_interval:
                print('Epoch: %03d/%03d | Batch %03d/%03d | Gen/Dis Loss: %.4f/%.4f' 
                       % (epoch+1, num_epochs, batch_idx, 
                          len(train_loader), gener_loss.item(), discr_loss.item()))


        print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))
    
    print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))
    
    if save_model is not None:
        torch.save(model.state_dict(), save_model)
    
    return log_dict



