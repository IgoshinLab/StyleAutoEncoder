# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Loss functions."""

import numpy as np
import torch
import torch.nn as nn

from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import upfirdn2d


# ----------------------------------------------------------------------------
def compute_l2_norm(model):
    l2_norm = 0.0
    for param in model.parameters():
        l2_norm += torch.norm(param.data, p=2) ** 2
    return torch.sqrt(l2_norm).item()


def compute_layerwise_l2_norm(model, part):
    for name, param in model.named_parameters():
        norm = torch.norm(param.data, p=2).item()
        training_stats.report(f'Parameters/{part}/{name}', norm)


# ----------------------------------------------------------------------------

class Loss:
    def accumulate_gradients(self, phase, real_img1, real_img2, real_c, gain, cur_nimg):  # to be overridden by subclass
        raise NotImplementedError()


# ----------------------------------------------------------------------------

class StyleGAN2Loss(Loss):
    def __init__(self, device, G, D, E, augment_pipe=None, r1_gamma=10, style_mixing_prob=0, pl_weight=0,
                 pl_batch_shrink=2, pl_decay=0.01, pl_no_weight_grad=False, blur_init_sigma=0, blur_fade_kimg=0,
                 kld_weight=0.02, total_steps=10000, increment_step=1000, noise_mode='random'):
        super().__init__()
        self.device = device
        self.G = G
        self.D = D
        self.E = E
        self.augment_pipe = augment_pipe
        self.r1_gamma = r1_gamma
        self.style_mixing_prob = style_mixing_prob
        self.pl_weight = pl_weight
        self.pl_batch_shrink = pl_batch_shrink
        self.pl_decay = pl_decay
        self.pl_no_weight_grad = pl_no_weight_grad
        self.pl_mean = torch.zeros([], device=device)
        self.blur_init_sigma = blur_init_sigma
        self.blur_fade_kimg = blur_fade_kimg
        self.triplet_loss = nn.TripletMarginLoss(margin=0.1, p=2)
        self.kld_weight = kld_weight
        self.noise_mode = noise_mode

    def run_G(self, z, c, update_emas=False):
        ws = self.G.mapping(z, c, update_emas=update_emas)
        if self.style_mixing_prob > 0:
            with torch.autograd.profiler.record_function('style_mixing'):
                cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff,
                                     torch.full_like(cutoff, ws.shape[1]))
                ws[:, cutoff:] = self.G.mapping(torch.randn_like(z), c, update_emas=False)[:, cutoff:]
        img = self.G.synthesis(ws, update_emas=update_emas, noise_mode=self.noise_mode)
        return img, ws

    def run_E(self, img, c, blur_sigma=0, update_emas=False):
        blur_size = np.floor(blur_sigma * 3)
        if blur_size > 0:
            with torch.autograd.profiler.record_function('blur'):
                f = torch.arange(-blur_size, blur_size + 1, device=img.device).div(blur_sigma).square().neg().exp2()
                img = upfirdn2d.filter2d(img, f / f.sum())
        if self.augment_pipe is not None:
            img = self.augment_pipe(img)
        encode = self.E(img, c, update_emas=update_emas)
        return encode

    def run_D(self, img, c, blur_sigma=0, update_emas=False):
        blur_size = np.floor(blur_sigma * 3)
        img1, img2 = img
        if blur_size > 0:
            with torch.autograd.profiler.record_function('blur'):
                f = torch.arange(-blur_size, blur_size + 1, device=img1.device).div(blur_sigma).square().neg().exp2()
                img1 = upfirdn2d.filter2d(img1, f / f.sum())
                img2 = upfirdn2d.filter2d(img2, f / f.sum())
        if self.augment_pipe is not None:
            img1 = self.augment_pipe(img1)
            img2 = self.augment_pipe(img2)
        logits = self.D([img1, img2], c, update_emas=update_emas)
        return logits

    def mu_var(self, img, c, blur_sigma=0, update_emas=False):
        blur_size = np.floor(blur_sigma * 3)
        if blur_size > 0:
            with torch.autograd.profiler.record_function('blur'):
                f = torch.arange(-blur_size, blur_size + 1, device=img.device).div(blur_sigma).square().neg().exp2()
                img = upfirdn2d.filter2d(img, f / f.sum())
        if self.augment_pipe is not None:
            img = self.augment_pipe(img)
        mu, logvar = self.E.mu_var(img, c)
        return mu, logvar

    def calc_gradient_penalty(self, real_data, fake_data, compare_data):
        """
            Copied from https://github.com/caogang/wgan-gp
        """
        # LAMBDA = 10
        # BATCH_SIZE = real_data.size()[0]

        # Creating interpolates for gradient penalty
        alpha = torch.rand(real_data.shape[0], 1, 1, 1, device=real_data.device)
        interpolates = alpha * real_data + ((1 - alpha) * fake_data)
        interpolates.requires_grad_(True)

        disc_interpolates = self.D([compare_data, interpolates], [None, None])

        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                        grad_outputs=torch.ones_like(disc_interpolates),
                                        create_graph=True, retain_graph=True, only_inputs=True)[0]

        # # Derivatives of the gradient close to 0 can cause problems because of
        # # the square root, so manually calculate norm and add epsilon
        # grad_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
        # gradient_penalty = ((grad_norm - 1) ** 2).mean() * LAMBDA
        # Calculate gradient penalty
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

        return gradient_penalty

    def accumulate_gradients(self, phase, real_img1, real_img2, real_c, gain, cur_nimg, state="training"):
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        if self.pl_weight == 0:
            phase = {'Greg': 'none', 'Gboth': 'Gmain'}.get(phase, phase)
        if self.r1_gamma == 0:
            phase = {'Dreg': 'none', 'Dboth': 'Dmain'}.get(phase, phase)
        blur_sigma = max(1 - cur_nimg / (self.blur_fade_kimg * 1e3),
                         0) * self.blur_init_sigma if self.blur_fade_kimg > 0 else 0

        switch_flag = np.random.random() > 0.5

        # Gmain: Maximize logits for generated images.
        if phase in ['Gmain', 'Gboth']:
            with torch.autograd.profiler.record_function('Gmain_forward'):
                gen_z = self.run_E(real_img1, real_c)

                ## KL-Divergence loss for VAE
                mu, logvar = self.mu_var(real_img1, real_c)
                training_stats.report(f'{state}_Value/E/mu', mu)
                training_stats.report(f'{state}_Value/E/logvar', logvar)

                '''The / 50 here is important for the separation'''
                KLD_loss = torch.mean(
                    0.5 * torch.sum(mu ** 2 + torch.exp(logvar) - logvar - 1, 1)) * self.kld_weight
                training_stats.report(f'{state}_Loss/E/KLD_loss', KLD_loss)
                # # The magnitude loss is important
                # mag_loss = torch.mean(torch.abs(torch.sqrt(torch.sum(torch.square(gen_z), 1)) - 1))
                # training_stats.report('Loss/E/mag_loss', mag_loss)

                gen_img, _gen_ws = self.run_G(gen_z, real_c)
                gen_logits = self.run_D([real_img1, gen_img], [real_c, real_c],
                                        blur_sigma=blur_sigma) if switch_flag else self.run_D([gen_img, real_img1],
                                                                                              [real_c, real_c],
                                                                                              blur_sigma=blur_sigma)
                training_stats.report(f'{state}_Loss/scores/fake', gen_logits)

                '''Gradient penalty is not compatible with multiple GPUs at this time.'''
                # (gradient_penalty, grad) = self.calc_gradient_penalty(real_img1, gen_img, real_img2)
                # training_stats.report('Loss/E/gradient_penalty', gradient_penalty)
                loss_Gmain = torch.nn.functional.softplus(-gen_logits)  # -log(sigmoid(gen_logits))
                loss_Gconstraints = KLD_loss  # + gradient_penalty
                training_stats.report(f'{state}_Loss/G/loss', loss_Gmain)

                # compute_layerwise_l2_norm(self.E, 'E')
                # compute_layerwise_l2_norm(self.G, 'G')
            if state == "training":
                with torch.autograd.profiler.record_function('Gmain_backward'):
                    loss_Gmain.mean().mul(gain).backward(retain_graph=True)
                    loss_Gconstraints.mul(gain).backward()

        # Gpl: Apply path length regularization.
        if phase in ['Greg', 'Gboth']:
            with torch.autograd.profiler.record_function('Gpl_forward'):
                gen_z = self.run_E(real_img1, real_c)
                batch_size = gen_z.shape[0] // self.pl_batch_shrink
                gen_img, gen_ws = self.run_G(gen_z[:batch_size], real_c[:batch_size])
                pl_noise = torch.randn_like(gen_img) / np.sqrt(gen_img.shape[2] * gen_img.shape[3])
                with torch.autograd.profiler.record_function('pl_grads'), conv2d_gradfix.no_weight_gradients(
                        self.pl_no_weight_grad):
                    pl_grads = \
                        torch.autograd.grad(outputs=[(gen_img * pl_noise).sum()], inputs=[gen_ws], create_graph=True,
                                            only_inputs=True)[0]
                pl_lengths = pl_grads.square().sum(2).mean(1).sqrt()
                pl_mean = self.pl_mean.lerp(pl_lengths.mean(), self.pl_decay)
                self.pl_mean.copy_(pl_mean.detach())
                pl_penalty = (pl_lengths - pl_mean).square()
                training_stats.report(f'{state}_Loss/pl_penalty', pl_penalty)
                loss_Gpl = pl_penalty * self.pl_weight
                training_stats.report(f'{state}_Loss/G/reg', loss_Gpl)

                # compute_layerwise_l2_norm(self.E, 'E')
                # compute_layerwise_l2_norm(self.G, 'G')
            if state == "training":
                with torch.autograd.profiler.record_function('Gpl_backward'):
                    loss_Gpl.mean().mul(gain).backward()

        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        if phase in ['Dmain', 'Dboth']:
            with torch.autograd.profiler.record_function('Dgen_forward'):
                gen_z = self.run_E(real_img1, real_c)
                gen_img, _gen_ws = self.run_G(gen_z, real_c, update_emas=True)
                gen_logits = self.run_D([gen_img, real_img1], [real_c, real_c], blur_sigma=blur_sigma,
                                        update_emas=True) if switch_flag else self.run_D([real_img1, gen_img],
                                                                                         [real_c, real_c],
                                                                                         blur_sigma=blur_sigma,
                                                                                         update_emas=True)

                loss_Dgen = torch.nn.functional.softplus(gen_logits)  # -log(1 - sigmoid(gen_logits))
                training_stats.report(f'{state}_Loss/D/gen_loss', loss_Dgen)

                # compute_layerwise_l2_norm(self.D, 'D')
            if state == "training":
                with torch.autograd.profiler.record_function('Dgen_backward'):
                    loss_Dgen.mean().mul(gain).backward()

        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if phase in ['Dmain', 'Dreg', 'Dboth']:
            name = 'Dreal' if phase == 'Dmain' else 'Dr1' if phase == 'Dreg' else 'Dreal_Dr1'
            with torch.autograd.profiler.record_function(name + '_forward'):
                real_img1_tmp = real_img1.detach().requires_grad_(phase in ['Dreg', 'Dboth'])
                real_img2_tmp = real_img2.detach().requires_grad_(phase in ['Dreg', 'Dboth'])
                real_logits = self.run_D([real_img2_tmp, real_img1_tmp], [real_c, real_c],
                                         blur_sigma=blur_sigma) if switch_flag else self.run_D(
                    [real_img1_tmp, real_img2_tmp], [real_c, real_c], blur_sigma=blur_sigma)
                training_stats.report(f'{state}_Loss/scores/real', real_logits)

                loss_Dreal = 0
                if phase in ['Dmain', 'Dboth']:
                    loss_Dreal = torch.nn.functional.softplus(-real_logits)  # -log(sigmoid(real_logits))
                    training_stats.report(f'{state}_Loss/D/real_loss', loss_Dreal)

                loss_Dr1 = 0
                if phase in ['Dreg', 'Dboth']:
                    with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                        r1_grads = \
                            torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img1_tmp, real_img2_tmp],
                                                create_graph=True, only_inputs=True)[0]
                    r1_penalty = r1_grads.square().sum([1, 2, 3])
                    loss_Dr1 = r1_penalty * (self.r1_gamma / 2)
                    training_stats.report(f'{state}_Loss/r1_penalty', r1_penalty)
                    training_stats.report(f'{state}_Loss/D/reg', loss_Dr1)

                # Add gradient penalty here
                if phase in ['Dmain', 'Dboth']:
                    gradient_penalty = self.calc_gradient_penalty(real_img1, gen_img, real_img2)
                    training_stats.report(f'{state}_Loss/D/gradient_penalty', gradient_penalty)
                    if state == "training":
                        gradient_penalty.backward()  # Include this in the backward pass

                # compute_layerwise_l2_norm(self.D, 'D')
            if state == "training":
                with torch.autograd.profiler.record_function(name + '_backward'):
                    (loss_Dreal + loss_Dr1).mean().mul(gain).backward()

# ----------------------------------------------------------------------------
