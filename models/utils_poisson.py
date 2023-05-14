import torch
import numpy as np


def forward_pz(sde, config, samples_batch, m, labels):
    """Perturbing the augmented training data. See Algorithm 2 in PFGM paper.

    Args:
      sde: An `methods.SDE` object that represents the forward SDE.
      config: configurations
      samples_batch: A mini-batch of un-augmented training data
      m: A 1D torch tensor. The exponents of (1+\tau).

    Returns:
      Perturbed samples
    """
    label_to_sigma = {0: 0.1, 1: 0.2, 2: 0.3, 3: 0.4, 4: 0.5, 5: 0.6, 6: 0.7, 7: 0.8, 8: 0.9, 9: 1.0}
    tau = config.training.tau
    z = torch.randn((len(samples_batch), 1, 1, 1)).to(samples_batch.device) * config.model.sigma_end
    z = z.abs()
    # Confine the norms of perturbed data.
    # see Appendix B.1.1 of https://arxiv.org/abs/2209.11178
    if config.training.restrict_M:
        idx = (z < 0.005).squeeze()
        num = int(idx.int().sum())
        restrict_m = int(sde.M * 0.7)
        m[idx] = torch.rand((num,), device=samples_batch.device) * restrict_m

    data_dim = config.data.channels * config.data.image_size * config.data.image_size
    multiplier = (1+tau) ** m

    # Assume `labels` is a tensor containing the label of each sample
    if labels is not None:
        assert labels is instanceof(torch.Tensor)
        sigmas = torch.tensor([label_to_sigma[label.item()] for label in labels])
        noise = torch.randn_like(samples_batch) * sigmas.view(-1, 1, 1, 1)
    else:
        noise = torch.randn_like(samples_batch).reshape(len(samples_batch), -1) * config.model.sigma_end
    norm_m = torch.norm(noise, p=2, dim=1) * multiplier
    # Perturb z
    perturbed_z = z.squeeze() * multiplier
    # Sample uniform angle
    gaussian = torch.randn(len(samples_batch), data_dim).to(samples_batch.device)
    unit_gaussian = gaussian / torch.norm(gaussian, p=2, dim=1, keepdim=True)
    # Construct the perturbation for x
    perturbation_x = unit_gaussian * norm_m[:, None]
    perturbation_x = perturbation_x.view_as(samples_batch)
    # Perturb x
    perturbed_x = samples_batch + perturbation_x
    # Augment the data with extra dimension z
    perturbed_samples_vec = torch.cat((perturbed_x.reshape(len(samples_batch), -1),
                                       perturbed_z[:, None]), dim=1)
    return perturbed_samples_vec
