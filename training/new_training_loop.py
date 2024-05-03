import copy
import itertools
import json
import os
import pickle
import time

import PIL.Image
import numpy as np
import psutil
import torch
from tensorboardX import SummaryWriter

import dnnlib
import legacy
from torch_utils import misc
from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import grid_sample_gradfix
from torch_utils.vis import scatterBWImages

from metrics import metric_main


# ----------------------------------------------------------------------------

def setup_snapshot_image_grid(training_set, random_seed=0):
    rnd = np.random.RandomState(random_seed)
    gw = np.clip(7680 // training_set.image_shape[2], 7, 32)
    gh = np.clip(4320 // training_set.image_shape[1], 4, 32)

    # No labels => show random subset of training samples.
    if not training_set.has_labels:
        all_indices = list(range(len(training_set)))
        rnd.shuffle(all_indices)
        grid_indices = [all_indices[i % len(all_indices)] for i in range(gw * gh)]

    else:
        # Group training samples by label.
        label_groups = dict()  # label => [idx, ...]
        for idx in range(len(training_set)):
            label = tuple(training_set.__getlabel__(idx).flat[::-1])
            if label not in label_groups:
                label_groups[label] = []
            label_groups[label].append(idx)

        # Reorder.
        label_order = sorted(label_groups.keys())
        for label in label_order:
            rnd.shuffle(label_groups[label])

        # Organize into grid.
        grid_indices = []
        for y in range(gh):
            label = label_order[y % len(label_order)]
            indices = label_groups[label]
            grid_indices += [indices[x % len(indices)] for x in range(gw)]
            label_groups[label] = [indices[(i + gw) % len(indices)] for i in range(len(indices))]

    # Load data.
    images1, images2, labels = zip(*[training_set[i] for i in grid_indices])
    return (gw, gh), np.stack(images1), np.stack(images2), np.stack(labels)


# ----------------------------------------------------------------------------

def save_image_grid(img, fname, drange, grid_size):
    lo, hi = drange
    img = np.asarray(img, dtype=np.float32)
    img = (img - lo) * (255 / (hi - lo))
    img = np.rint(img).clip(0, 255).astype(np.uint8)

    gw, gh = grid_size
    _N, C, H, W = img.shape
    img = img.reshape([gh, gw, C, H, W])
    img = img.transpose(0, 3, 1, 4, 2)
    img = img.reshape([gh * H, gw * W, C])

    assert C in [1, 3]
    if C == 1:
        PIL.Image.fromarray(img[:, :, 0], 'L').save(fname)
    if C == 3:
        PIL.Image.fromarray(img, 'RGB').save(fname)


def calc_gradient_penalty(real_data, fake_data, conditional_data, D):
    """
        Copied from https://github.com/caogang/wgan-gp
    """
    LAMBDA = 10
    BATCH_SIZE = real_data.size()[0]

    alpha = torch.rand(BATCH_SIZE, 1)
    alpha = alpha.expand(BATCH_SIZE, int(real_data.nelement() / BATCH_SIZE)).contiguous().view_as(real_data)
    alpha = alpha.cuda()
    alpha = torch.autograd.Variable(alpha, requires_grad=True)

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    disc_interpolates = D(conditional_data, interpolates)

    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                    grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]

    # Derivatives of the gradient close to 0 can cause problems because of
    # the square root, so manually calculate norm and add epsilon
    grad_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

    gradient_penalty = ((grad_norm - 1) ** 2).mean() * LAMBDA

    return gradient_penalty, grad_norm


def training_loop(
        run_dir='.',  # Output directory.
        citers=3,  # Train the D for citers in each epoch.
        training_set_kwargs={},  # Options for training set.
        validation_set_kwargs={},  # Options for validation set.
        data_loader_kwargs={},  # Options for torch.utils.data.DataLoader.
        validation_data_loader_kwargs={},  # Options for torch.utils.data.DataLoader.
        E_kwargs={},  # Options for encoder network.
        G_kwargs={},  # Options for generator network.
        D_kwargs={},  # Options for discriminator network.
        G_opt_kwargs={},  # Options for generator optimizer.
        D_opt_kwargs={},  # Options for discriminator optimizer.
        augment_kwargs=None,  # Options for augmentation pipeline. None = disable.
        loss_kwargs={},  # Options for loss function.
        metrics=[],  # Metrics to evaluate during training.
        random_seed=0,  # Global random seed.
        num_gpus=1,  # Number of GPUs participating in the training.
        rank=0,  # Rank of the current process in [0, num_gpus].
        batch_size=4,  # Total batch size for one training iteration. Can be larger than batch_gpu * num_gpus.
        batch_gpu=4,  # Number of samples processed at a time by one GPU.
        ema_kimg=10,  # Half-life of the exponential moving average (EMA) of generator weights.
        ema_rampup=0.05,  # EMA ramp-up coefficient. None = no rampup.
        G_reg_interval=None,  # How often to perform regularization for G? None = disable lazy regularization.
        D_reg_interval=16,  # How often to perform regularization for D? None = disable lazy regularization.
        augment_p=0,  # Initial value of augmentation probability.
        ada_target=None,  # ADA target value. None = fixed p.
        ada_interval=4,  # How often to perform ADA adjustment?
        ada_kimg=500,
        # ADA adjustment speed, measured in how many kimg it takes for p to increase/decrease by one unit.
        total_kimg=25000,  # Total length of the training, measured in thousands of real images.
        kimg_per_tick=4,  # Progress snapshot interval.
        image_snapshot_ticks=50,  # How often to save image snapshots? None = disable.
        network_snapshot_ticks=50,  # How often to save network snapshots? None = disable.
        resume_pkl=None,  # Network pickle to resume training from.
        resume_kimg=0,  # First kimg to report when resuming training.
        cudnn_benchmark=True,  # Enable torch.backends.cudnn.benchmark?
        abort_fn=None,
        # Callback function for determining whether to abort training. Must return consistent results across ranks.
        progress_fn=None,  # Callback function for updating training progress. Called for all ranks.
        record_adam=False,  # The flag for recording adam mean and variation
):
    # Initialize.
    start_time = time.time()
    device = torch.device('cuda', rank)
    np.random.seed(random_seed * num_gpus + rank)
    torch.manual_seed(random_seed * num_gpus + rank)
    torch.backends.cudnn.benchmark = cudnn_benchmark  # Improves training speed.
    torch.backends.cuda.matmul.allow_tf32 = False  # Improves numerical accuracy.
    torch.backends.cudnn.allow_tf32 = False  # Improves numerical accuracy.
    conv2d_gradfix.enabled = True  # Improves training speed.
    grid_sample_gradfix.enabled = True  # Avoids errors with the augmentation pipe.

    # Load training set.
    if rank == 0:
        print('Loading training set...')
    training_set = dnnlib.util.construct_class_by_name(**training_set_kwargs)  # subclass of training.dataset.Dataset
    training_set_sampler = misc.InfiniteSampler(dataset=training_set, rank=rank, num_replicas=num_gpus,
                                                seed=random_seed)
    training_set_iterator = iter(torch.utils.data.DataLoader(dataset=training_set, sampler=training_set_sampler,
                                                             batch_size=batch_size // num_gpus, **data_loader_kwargs))

    # Load validation set
    if validation_set_kwargs:
        if rank == 0:
            print("Loading validation set...")
        validation_set = dnnlib.util.construct_class_by_name(
            **validation_set_kwargs)  # subclass of training.dataset.Dataset
        validation_set_loader = torch.utils.data.DataLoader(dataset=validation_set, batch_size=batch_size // num_gpus,
                                                            **validation_data_loader_kwargs)
    else:
        validation_set_loader = None

    if rank == 0:
        print()
        print('Num images: ', len(training_set))
        print('Image shape:', training_set.image_shape)
        print('Label shape:', training_set.label_shape)
        print()

    # Construct networks.
    if rank == 0:
        print('Constructing networks...')
    common_kwargs = dict(c_dim=training_set.label_dim, img_resolution=training_set.resolution,
                         img_channels=training_set.num_channels)
    E = dnnlib.util.construct_class_by_name(**E_kwargs, **common_kwargs).train().requires_grad_(False).to(
        device)  # subclass of torch.nn.Module
    G = dnnlib.util.construct_class_by_name(**G_kwargs, **common_kwargs).train().requires_grad_(False).to(
        device)  # subclass of torch.nn.Module
    D = dnnlib.util.construct_class_by_name(**D_kwargs, **common_kwargs).train().requires_grad_(False).to(
        device)  # subclass of torch.nn.Module
    E_ema = copy.deepcopy(E).eval()
    G_ema = copy.deepcopy(G).eval()

    # Resume from existing pickle.
    if (resume_pkl is not None) and (rank == 0):
        print(f'Resuming from "{resume_pkl}"')
        with dnnlib.util.open_url(resume_pkl) as f:
            resume_data = legacy.load_network_pkl(f)
        for name, module in [('E', E), ('G', G), ('D', D), ('G_ema', G_ema), ('E_ema', E_ema)]:
            misc.copy_params_and_buffers(resume_data[name], module, require_all=False)

    # Print network summary tables.
    if rank == 0:
        z = torch.empty([batch_gpu, G.z_dim], device=device)
        c = torch.empty([batch_gpu, G.c_dim], device=device)
        img = misc.print_module_summary(G, [z, c])
        misc.print_module_summary(D, [[img, img], [c, c]])
        misc.print_module_summary(E, [img, c])
    log = SummaryWriter(run_dir, '')

    # Setup augmentation.
    if rank == 0:
        print('Setting up augmentation...')
    augment_pipe = None
    ada_stats = None
    if (augment_kwargs is not None) and (augment_p > 0 or ada_target is not None):
        augment_pipe = dnnlib.util.construct_class_by_name(**augment_kwargs).train().requires_grad_(False).to(
            device)  # subclass of torch.nn.Module
        augment_pipe.p.copy_(torch.as_tensor(augment_p))
        if ada_target is not None:
            ada_stats = training_stats.Collector(regex='Loss/signs/real')

    # Distribute across GPUs.
    if rank == 0:
        print(f'Distributing across {num_gpus} GPUs...')
    for module in [E, G, D, G_ema, E_ema, augment_pipe]:
        if module is not None and num_gpus > 1:
            for param in misc.params_and_buffers(module):
                torch.distributed.broadcast(param, src=0)

    # Setup training phases.
    if rank == 0:
        print('Setting up training phases...')
    loss = dnnlib.util.construct_class_by_name(device=device, G=G, D=D, E=E, augment_pipe=augment_pipe,
                                               **loss_kwargs)  # subclass of training.loss.Loss
    phases = []
    for name, module, opt_kwargs, reg_interval in [('D', [D], D_opt_kwargs, D_reg_interval),
                                                   ('G', [G, E], G_opt_kwargs, G_reg_interval)]:
        if len(module) == 1:
            params = module[0].parameters()
        else:
            params = itertools.chain(module[0].parameters(), module[1].parameters())
        if reg_interval is None:
            opt = dnnlib.util.construct_class_by_name(params=params,
                                                      **opt_kwargs)  # subclass of torch.optim.Optimizer
            if name == "D":
                for citer in range(citers):
                    phases += [dnnlib.EasyDict(name=name + 'both', module=module, opt=opt, interval=1)]
            else:
                phases += [dnnlib.EasyDict(name=name + 'both', module=module, opt=opt, interval=1)]
        else:  # Lazy regularization.
            mb_ratio = reg_interval / (reg_interval + 1)
            opt_kwargs = dnnlib.EasyDict(opt_kwargs)
            opt_kwargs.lr = opt_kwargs.lr * mb_ratio
            opt_kwargs.betas = [beta ** mb_ratio for beta in opt_kwargs.betas]
            opt = dnnlib.util.construct_class_by_name(params,
                                                      **opt_kwargs)  # subclass of torch.optim.Optimizer
            if name == "D":
                for citer in range(citers):
                    phases += [dnnlib.EasyDict(name=name + 'main', module=module, opt=opt, interval=1)]
                    phases += [dnnlib.EasyDict(name=name + 'reg', module=module, opt=opt, interval=reg_interval)]
            else:
                phases += [dnnlib.EasyDict(name=name + 'main', module=module, opt=opt, interval=1)]
                phases += [dnnlib.EasyDict(name=name + 'reg', module=module, opt=opt, interval=reg_interval)]
    for phase in phases:
        phase.start_event = None
        phase.end_event = None
        if rank == 0:
            phase.start_event = torch.cuda.Event(enable_timing=True)
            phase.end_event = torch.cuda.Event(enable_timing=True)

    # Export sample images.
    grid_size = None
    grid_z = None
    grid_c = None
    if rank == 0:
        print('Exporting sample images...')
        grid_size, images1, images2, labels = setup_snapshot_image_grid(training_set=training_set)
        save_image_grid(images1, os.path.join(run_dir, 'reals.png'), drange=[-1, 1], grid_size=grid_size)
        if G.z_dim == 2:
            x = np.linspace(-4, 4, grid_size[0])
            y = np.linspace(-4, 4, grid_size[1])
            xv, yv = np.meshgrid(x, y)
            grid_z = np.array([xv.flatten(), yv.flatten()]).transpose((1, 0))
            grid_z = torch.from_numpy(grid_z).to(device).split(batch_gpu)
        else:
            grid_z = torch.randn([labels.shape[0], G.z_dim], device=device).split(batch_gpu)
        grid_c = torch.from_numpy(labels).to(device).split(batch_gpu)
        images = torch.cat([G_ema(z=z, c=c, noise_mode='const').cpu() for z, c in zip(grid_z, grid_c)]).numpy()
        save_image_grid(images, os.path.join(run_dir, 'fakes_init.png'), drange=[-1, 1], grid_size=grid_size)

    # Initialize logs.
    if rank == 0:
        print('Initializing logs...')
    stats_collector = training_stats.Collector(regex='.*')
    stats_metrics = dict()
    stats_jsonl = None
    stats_tfevents = None
    if rank == 0:
        stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'wt')
        try:
            import torch.utils.tensorboard as tensorboard
            stats_tfevents = tensorboard.SummaryWriter(run_dir)
        except ImportError as err:
            print('Skipping tfevents export:', err)

    # Train.
    if rank == 0:
        print(f'Training for {total_kimg} kimg...')
        print()
    cur_nimg = resume_kimg * 1000
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    batch_idx = 0
    if progress_fn is not None:
        progress_fn(0, total_kimg)

    break_flag = False
    while True:

        # Fetch training data.
        with torch.autograd.profiler.record_function('data_fetch'):
            all_gen_z = torch.randn([len(phases) * batch_size, G.z_dim], device=device)
            all_gen_z = [phase_gen_z.split(batch_gpu) for phase_gen_z in all_gen_z.split(batch_size)]
            all_gen_c = [training_set.get_label(np.random.randint(len(training_set))) for _ in
                         range(len(phases) * batch_size)]
            all_gen_c = torch.from_numpy(np.stack(all_gen_c)).pin_memory().to(device)
            all_gen_c = [phase_gen_c.split(batch_gpu) for phase_gen_c in all_gen_c.split(batch_size)]

        # Execute training phases.
        for phase, phase_gen_z, phase_gen_c in zip(phases, all_gen_z, all_gen_c):
            if batch_idx % phase.interval != 0:
                continue
            if phase.start_event is not None:
                phase.start_event.record(torch.cuda.current_stream(device))

            with torch.autograd.profiler.record_function('data_fetch'):
                phase_real_img1, phase_real_img2, phase_real_c = next(training_set_iterator)
                phase_real_img1 = phase_real_img1.to(device).to(torch.float32).split(
                    batch_gpu)
                phase_real_img2 = phase_real_img2.to(device).to(torch.float32).split(batch_gpu)
                phase_real_c = phase_real_c.to(device).split(batch_gpu)

            # Accumulate gradients.
            phase.opt.zero_grad(set_to_none=True)
            for module_i in phase.module:
                module_i.requires_grad_(True)
            for real_img1, real_img2, real_c in zip(phase_real_img1, phase_real_img2, phase_real_c):
                loss.accumulate_gradients(phase=phase.name, real_img1=real_img1, real_img2=real_img2,
                                          real_c=real_c,
                                          gain=phase.interval, cur_nimg=cur_nimg)
            for module_i in phase.module:
                module_i.requires_grad_(False)

            # Update weights.
            with torch.autograd.profiler.record_function(phase.name + '_opt'):
                params = []
                for module_i in phase.module:
                    for parameter in module_i.parameters():
                        if parameter.grad is not None:
                            params.append(parameter)
                if len(params) > 0:
                    flat = torch.cat([param.grad.flatten() for param in params])
                    if num_gpus > 1:
                        torch.distributed.all_reduce(flat)
                        flat /= num_gpus
                    misc.nan_to_num(flat, nan=0, posinf=1e5, neginf=-1e5, out=flat)
                    grads = flat.split([param.numel() for param in params])
                    for param, grad in zip(params, grads):
                        param.grad = grad.reshape(param.shape)
                phase.opt.step()

            # Record the terms in Adam optimizer for debug
            if record_adam:
                for module_i in phase.module:
                    for name, param in module_i.named_parameters():
                        if param in phase.opt.state:
                            m = phase.opt.state[param]['exp_avg']
                            v = phase.opt.state[param]['exp_avg_sq']
                            if torch.sum(torch.isnan(m)):
                                print(f"{phase.name}/{name}/m goes to NaN!")
                                break_flag |= True
                            if torch.sum(torch.isinf(m)):
                                print(f"{phase.name}/{name}/m goes to INF!")
                                break_flag |= True
                            if torch.sum(torch.isnan(v)):
                                print(f"{phase.name}/{name}/v goes to NaN!")
                                break_flag |= True
                            if torch.sum(torch.isinf(v)):
                                print(f"{phase.name}/{name}/v goes to INF!")
                                break_flag |= True
                            # if torch.sum(v < 1e-8) > 0:
                            #     print(f"{phase.name}/{name}/v is smaller than EPS (1e-8)!")
                            #     break_flag |= True
                            training_stats.report(f"Adam_m_mean/{phase.name}/{name}", m.mean().item())
                            training_stats.report(f"Adam_v_mean/{phase.name}/{name}", v.mean().item())
                            training_stats.report(f"Adam_m_l2_norm/{phase.name}/{name}", torch.norm(m, p=2).item())
                            training_stats.report(f"Adam_v_l2_norm/{phase.name}/{name}", torch.norm(v, p=2).item())
                if break_flag:
                    '''You can also add the feature maps.'''
                    adam_stat = {
                        'input': [phase_real_img1[batch_idx].cpu().numpy() for batch_idx in
                                  range(len(phase_real_img1))],
                        'compare': [phase_real_img2[batch_idx].cpu().numpy() for batch_idx in
                                    range(len(phase_real_img2))]
                    }
                    for module_i in phase.module:
                        for name, param in module_i.named_parameters():
                            adam_stat[name] = {
                                'param': param
                            }
                            if param in phase.opt.state:
                                adam_stat[name]['m'] = phase.opt.state[param]['exp_avg']
                                adam_stat[name]['v'] = phase.opt.state[param]['exp_avg_sq']
                    pickle.dump(adam_stat, open(f'{run_dir}/{cur_nimg}_adam_optimizer_state.pkl', 'wb'))
                    break

            # Phase done.
            if phase.end_event is not None:
                phase.end_event.record(torch.cuda.current_stream(device))

        # Update G_ema.
        with torch.autograd.profiler.record_function('G_ema'):
            ema_nimg = ema_kimg * 1000
            if ema_rampup is not None:
                ema_nimg = min(ema_nimg, cur_nimg * ema_rampup)
            ema_beta = 0.5 ** (batch_size / max(ema_nimg, 1e-8))
            for p_ema, p in zip(G_ema.parameters(), G.parameters()):
                p_ema.copy_(p.lerp(p_ema, ema_beta))
            for b_ema, b in zip(G_ema.buffers(), G.buffers()):
                b_ema.copy_(b)

        # Update E_ema.
        with torch.autograd.profiler.record_function('E_ema'):
            ema_nimg = ema_kimg * 1000
            if ema_rampup is not None:
                ema_nimg = min(ema_nimg, cur_nimg * ema_rampup)
            ema_beta = 0.5 ** (batch_size / max(ema_nimg, 1e-8))
            for p_ema, p in zip(E_ema.parameters(), E.parameters()):
                p_ema.copy_(p.lerp(p_ema, ema_beta))
            for b_ema, b in zip(E_ema.buffers(), E.buffers()):
                b_ema.copy_(b)

        # Update state.
        cur_nimg += batch_size
        batch_idx += 1

        # Execute ADA heuristic.
        if (ada_stats is not None) and (batch_idx % ada_interval == 0):
            ada_stats.update()
            adjust = np.sign(ada_stats['Loss/signs/real'] - ada_target) * (batch_size * ada_interval) / (
                    ada_kimg * 1000)
            augment_pipe.p.copy_((augment_pipe.p + adjust).max(misc.constant(0, device=device)))

        # Perform maintenance tasks once per tick.
        done = (cur_nimg >= total_kimg * 1000)
        if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
            continue

        # Print status line, accumulating the same information in training_stats.
        tick_end_time = time.time()
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<8.1f}"]
        fields += [
            f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
        fields += [
            f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
        fields += [
            f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"]
        fields += [f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"]
        fields += [
            f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2 ** 30):<6.2f}"]
        fields += [
            f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2 ** 30):<6.2f}"]
        fields += [
            f"reserved {training_stats.report0('Resources/peak_gpu_mem_reserved_gb', torch.cuda.max_memory_reserved(device) / 2 ** 30):<6.2f}"]
        torch.cuda.reset_peak_memory_stats()
        fields += [
            f"augment {training_stats.report0('Progress/augment', float(augment_pipe.p.cpu()) if augment_pipe is not None else 0):.3f}"]
        training_stats.report0('Timing/total_hours', (tick_end_time - start_time) / (60 * 60))
        training_stats.report0('Timing/total_days', (tick_end_time - start_time) / (24 * 60 * 60))

        if rank == 0:
            print(' '.join(fields))

        # Check for abort.
        if (not done) and (abort_fn is not None) and abort_fn():
            done = True
            if rank == 0:
                print()
                print('Aborting...')

        # Save image snapshot.
        if (rank == 0) and (image_snapshot_ticks is not None) and (done or cur_tick % image_snapshot_ticks == 0):
            images = torch.cat([G_ema(z=z, c=c, noise_mode='random').cpu() for z, c in zip(grid_z, grid_c)]).numpy()
            save_image_grid(images, os.path.join(run_dir, f'fakes{cur_nimg // 1000:06d}.png'), drange=[-1, 1],
                            grid_size=grid_size)
            # Save image projections
            z_mu = []
            z_var = []
            features = []
            x_fakes = []
            x_fakes2 = []
            for real_img1, real_img2, real_c in zip(phase_real_img1, phase_real_img2, phase_real_c):
                p_label, _ = E.mu_var(real_img1, real_c)
                p_var = E(real_img1, real_c)
                z_mu.append(p_label)
                z_var.append(p_var)
                feat = G.mapping(p_label, real_c)
                x_fake = G.synthesis(feat)
                features.append(feat[:, 0, :])
                x_fakes.append(x_fake)
                x_fake2 = G(p_var, real_c)
                x_fakes2.append(x_fake2)
            z_mu = torch.cat(z_mu, dim=0)
            z_var = torch.cat(z_var, dim=0)
            features = torch.cat(features, dim=0)
            x_fakes = torch.cat(x_fakes, dim=0)
            x_fakes2 = torch.cat(x_fakes2, dim=0)
            if G.z_dim == 2:
                phase_real_img1_np = [pri.cpu().numpy() for pri in phase_real_img1]
                phase_real_img1_np = np.concatenate(phase_real_img1_np, axis=0)
                projection = scatterBWImages(z_mu.cpu().numpy(), phase_real_img1_np)
                projection2 = scatterBWImages(z_var.cpu().numpy(), x_fakes2.cpu().numpy())
                projections = np.concatenate([projection, projection2], axis=1)
                im = PIL.Image.fromarray(projections)
                im.save(os.path.join(run_dir, f'projection{cur_nimg // 1000:06d}.png'))
            # Save image in tensorboard
            img_pair = torch.cat([phase_real_img1[0], x_fakes], dim=2)
            img_pair = torch.cat([img_pair, img_pair], dim=3)
            log.add_embedding(z_mu,
                              metadata=phase_real_c[0],
                              label_img=(phase_real_img1[0] + 1) / 2,
                              global_step=cur_nimg // 1000,
                              tag=f'z_mu/{cur_nimg // 1000:06d}')
            log.add_embedding(z_var,
                              metadata=phase_real_c[0],
                              label_img=(phase_real_img1[0] + 1) / 2,
                              global_step=cur_nimg // 1000,
                              tag=f'z_var/{cur_nimg // 1000:06d}')
            log.add_embedding(features,
                              metadata=phase_real_c[0],
                              label_img=(img_pair + 1) / 2,
                              global_step=cur_nimg // 1000,
                              tag=f'w/{cur_nimg // 1000:06d}')

        # Save network snapshot.
        snapshot_pkl = None
        snapshot_data = None
        if ((network_snapshot_ticks is not None) and (done or cur_tick % network_snapshot_ticks == 0)) or break_flag:
            snapshot_data = dict(G=G, D=D, E=E, G_ema=G_ema, E_ema=E_ema, augment_pipe=augment_pipe,
                                 training_set_kwargs=dict(training_set_kwargs))
            for key, value in snapshot_data.items():
                if isinstance(value, torch.nn.Module):
                    value = copy.deepcopy(value).eval().requires_grad_(False)
                    if num_gpus > 1:
                        misc.check_ddp_consistency(value, ignore_regex=r'.*\.[^.]+_(avg|ema)')
                        for param in misc.params_and_buffers(value):
                            torch.distributed.broadcast(param, src=0)
                    snapshot_data[key] = value.cpu()
                del value  # conserve memory
            snapshot_pkl = os.path.join(run_dir, f'network-snapshot-{cur_nimg // 1000:06d}.pkl')
            if rank == 0:
                with open(snapshot_pkl, 'wb') as f:
                    pickle.dump(snapshot_data, f)

            # Evaluate on the evaluation set
            if validation_set_kwargs:
                for phase_real_img1, phase_real_img2, phase_real_c in validation_set_loader:
                    phase_real_img1 = phase_real_img1.to(device).to(torch.float32).split(batch_gpu)
                    phase_real_img2 = phase_real_img2.to(device).to(torch.float32).split(batch_gpu)
                    phase_real_c = phase_real_c.to(device).split(batch_gpu)
                    for real_img1, real_img2, real_c in zip(phase_real_img1, phase_real_img2, phase_real_c):
                        mu, logvar = E.mu_var(real_img1, None)
                        recon = G(mu, None).detach()
                        # Calculate loss
                        # L_tot = L_G + L_D + L_VAE + L_reg
                        # L_G = loss_Gmain + loss_Gconstraints + loss_Gpl
                        gen_logits = loss.run_D([real_img1, recon], [None, None], blur_sigma=0)
                        loss_Gmain = torch.nn.functional.softplus(-gen_logits)
                        # L_D = loss_Dreal + loss_Dgen + gradient_penalty
                        real_logits = loss.run_D([real_img2, real_img1], [None, None], blur_sigma=0)
                        loss_Dreal = torch.nn.functional.softplus(-real_logits)
                        loss_Dgen = torch.nn.functional.softplus(gen_logits)
                        # L_VAE = KLD_loss
                        KLD_loss = torch.mean(
                            0.5 * torch.sum(mu ** 2 + torch.exp(logvar) - logvar - 1, 1)) * 0.1
                        # Lreg = loss_Dr1
                        loss_tot = loss_Gmain + loss_Dreal + loss_Dgen

                        training_stats.report(f'validation_Loss/G/loss', loss_Gmain)
                        training_stats.report(f'validation_Loss/D/real_loss', loss_Dreal)
                        training_stats.report(f'validation_Loss/D/gen_loss', loss_Dgen)
                        training_stats.report(f'validation_Loss/E/KLD_loss', KLD_loss)
                        training_stats.report(f'validation_Loss/tot_loss', loss_tot)
                        training_stats.report(f'validation_Loss/scores/real', real_logits)
                        training_stats.report(f'validation_Loss/scores/fake', gen_logits)
        if break_flag:
            break

        # Evaluate metrics.
        if (snapshot_data is not None) and (len(metrics) > 0):
            if rank == 0:
                print('Evaluating metrics...')
            for metric in metrics:
                result_dict = metric_main.calc_metric(metric=metric, G=snapshot_data['G_ema'],
                                                      dataset_kwargs=training_set_kwargs, num_gpus=num_gpus,
                                                      rank=rank, device=device)
                if rank == 0:
                    metric_main.report_metric(result_dict, run_dir=run_dir, snapshot_pkl=snapshot_pkl)
                stats_metrics.update(result_dict.results)
        del snapshot_data  # conserve memory

        # Collect statistics.
        for phase in phases:
            value = []
            if (phase.start_event is not None) and (phase.end_event is not None):
                phase.end_event.synchronize()
                value = phase.start_event.elapsed_time(phase.end_event)
            training_stats.report0('Timing/' + phase.name, value)
        stats_collector.update()
        stats_dict = stats_collector.as_dict()

        # Update logs.
        timestamp = time.time()
        if stats_jsonl is not None:
            fields = dict(stats_dict, timestamp=timestamp)
            stats_jsonl.write(json.dumps(fields) + '\n')
            stats_jsonl.flush()
        if stats_tfevents is not None:
            global_step = int(cur_nimg / 1e3)
            walltime = timestamp - start_time
            for name, value in stats_dict.items():
                stats_tfevents.add_scalar(name, value.mean, global_step=global_step, walltime=walltime)
            for name, value in stats_metrics.items():
                stats_tfevents.add_scalar(f'Metrics/{name}', value, global_step=global_step, walltime=walltime)
            stats_tfevents.flush()
        if progress_fn is not None:
            progress_fn(cur_nimg // 1000, total_kimg)

        # Update state.
        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        if done:
            break

    log.close()
    # Done.
    if rank == 0:
        print()
        print('Exiting...')

# ----------------------------------------------------------------------------
