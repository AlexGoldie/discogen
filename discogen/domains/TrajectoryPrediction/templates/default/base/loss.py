import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Laplace, MultivariateNormal
from typing import Dict, Any


def compute_loss(
    predictions: Dict[str, torch.Tensor],
    batch: Dict[str, Any],
    config: Dict[str, Any]
) -> torch.Tensor:
    # Extract ground truth
    if 'input_dict' in batch:
        inputs = batch['input_dict']
    else:
        inputs = batch

    ground_truth = torch.cat([
        inputs['center_gt_trajs'][..., :2],
        inputs['center_gt_trajs_mask'].unsqueeze(-1)
    ], dim=-1)

    center_gt_final_valid_idx = inputs.get('center_gt_final_valid_idx', None)

    # Create criterion and compute loss
    criterion = Criterion(config)
    return criterion(predictions, ground_truth, center_gt_final_valid_idx)


class Criterion(nn.Module):

    def __init__(self, config: Dict[str, Any]):
        super(Criterion, self).__init__()
        self.config = config

    def forward(self, out, gt, center_gt_final_valid_idx):
        return self.nll_loss_multimodes(out, gt, center_gt_final_valid_idx)

    def get_BVG_distributions(self, pred):
        B = pred.size(0)
        T = pred.size(1)
        mu_x = pred[:, :, 0].unsqueeze(2)
        mu_y = pred[:, :, 1].unsqueeze(2)
        sigma_x = pred[:, :, 2]
        sigma_y = pred[:, :, 3]
        rho = pred[:, :, 4]

        # Create the base covariance matrix for a single element
        cov = torch.stack([
            torch.stack([sigma_x ** 2, rho * sigma_x * sigma_y], dim=-1),
            torch.stack([rho * sigma_x * sigma_y, sigma_y ** 2], dim=-1)
        ], dim=-2)

        # Expand this base matrix to match the desired shape
        biv_gauss_dist = MultivariateNormal(
            loc=torch.cat((mu_x, mu_y), dim=-1),
            covariance_matrix=cov,
            validate_args=False
        )
        return biv_gauss_dist

    def get_Laplace_dist(self, pred):
        return Laplace(pred[:, :, :2], pred[:, :, 2:4], validate_args=False)

    def nll_pytorch_dist(self, pred, data, mask, rtn_loss=True):
        biv_gauss_dist = self.get_Laplace_dist(pred)
        num_active_per_timestep = mask.sum()
        data_reshaped = data[:, :, :2]
        if rtn_loss:
            return ((-biv_gauss_dist.log_prob(data_reshaped)).sum(-1) * mask).sum(1)  # Laplace
        else:
            return ((-biv_gauss_dist.log_prob(data_reshaped)).sum(dim=2) * mask).sum(1)  # Laplace

    def nll_loss_multimodes(self, output, data, center_gt_final_valid_idx):
        modes_pred = output['predicted_probability']
        pred = output['predicted_trajectory'].permute(1, 2, 0, 3)  # [B,c,T,5] -> [c,T,B,5]
        mask = data[..., -1]

        entropy_weight = 40.0
        kl_weight = 20.0
        use_FDEADE_aux_loss = True

        modes = len(pred)
        nSteps, batch_sz, dim = pred[0].shape

        log_lik_list = []
        with torch.no_grad():
            for kk in range(modes):
                nll = self.nll_pytorch_dist(pred[kk].transpose(0, 1), data, mask, rtn_loss=False)
                log_lik_list.append(-nll.unsqueeze(1))  # Add a new dimension to concatenate later

            # Concatenate the list to form the log_lik tensor
            log_lik = torch.cat(log_lik_list, dim=1)

            priors = modes_pred
            log_priors = torch.log(priors)
            log_posterior_unnorm = log_lik + log_priors

            # Compute logsumexp for normalization, ensuring no in-place operations
            logsumexp = torch.logsumexp(log_posterior_unnorm, dim=-1, keepdim=True)
            log_posterior = log_posterior_unnorm - logsumexp

            # Compute the posterior probabilities without in-place operations
            post_pr = torch.exp(log_posterior)
            # Ensure post_pr is a tensor on the correct device
            post_pr = post_pr.to(data.device)

        # Compute loss.
        loss = 0.0
        for kk in range(modes):
            nll_k = self.nll_pytorch_dist(pred[kk].transpose(0, 1), data, mask, rtn_loss=True) * post_pr[:, kk]
            loss += nll_k.mean()

        # Adding entropy loss term to ensure that individual predictions do not try to cover multiple modes.
        entropy_vals = []
        for kk in range(modes):
            entropy_vals.append(self.get_BVG_distributions(pred[kk]).entropy())
        entropy_vals = torch.stack(entropy_vals).permute(2, 0, 1)
        entropy_loss = torch.mean((entropy_vals).sum(2).max(1)[0])
        loss += entropy_weight * entropy_loss

        # KL divergence between the prior and the posterior distributions.
        kl_loss_fn = torch.nn.KLDivLoss(reduction='batchmean')
        kl_loss = kl_weight * kl_loss_fn(torch.log(modes_pred), post_pr)

        # compute ADE/FDE loss - L2 norms with between best predictions and GT.
        if use_FDEADE_aux_loss:
            adefde_loss = self.l2_loss_fde(pred, data, mask)
        else:
            adefde_loss = torch.tensor(0.0).to(data.device)

        # post_entropy
        final_loss = loss + kl_loss + adefde_loss

        return final_loss

    def l2_loss_fde(self, pred, data, mask):
        fde_loss = (torch.norm(
            (pred[:, -1, :, :2].transpose(0, 1) - data[:, -1, :2].unsqueeze(1)), 2, dim=-1
        ) * mask[:, -1:])
        ade_loss = (torch.norm(
            (pred[:, :, :, :2].transpose(1, 2) - data[:, :, :2].unsqueeze(0)), 2, dim=-1
        ) * mask.unsqueeze(0)).mean(dim=2).transpose(0, 1)
        loss, min_inds = (fde_loss + ade_loss).min(dim=1)
        return 100.0 * loss.mean()
