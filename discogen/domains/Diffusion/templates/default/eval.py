import os

import torch
from torch import distributed as dist
from einops import rearrange, repeat
from pytorch_fid.inception import InceptionV3
from torch.nn.functional import adaptive_avg_pool2d


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def sqrtm(matrix):
    """
    Compute matrix square root using SVD.
    """
    U, S, Vh = torch.linalg.svd(matrix, driver="gesvd")

    sqrtS = torch.sqrt(S)
    sqrtm = U @ torch.diag(sqrtS) @ Vh

    return sqrtm


class FIDEvaluator:
    def __init__(
        self,
        real_dataset_dl,
        channels=3,
        num_fake_samples=50000,
        batch_size=128,
        stats_dir="./results",
        rank=0,
        world_size=1,
        inception_block_idx=2048,
    ):
        self.batch_size = batch_size
        self.n_samples = num_fake_samples // world_size
        self.channels = channels
        self.dl = real_dataset_dl
        self.stats_dir = stats_dir

        assert inception_block_idx in InceptionV3.BLOCK_INDEX_BY_DIM
        self.block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[inception_block_idx]
        inception_v3 = InceptionV3([self.block_idx]).to(rank)
        inception_v3.eval()
        self.inception_v3 = torch.compile(inception_v3)
        self.inception_v3 = inception_v3.half()

        self.rank = rank
        self.world_size = world_size

    @torch.inference_mode()
    def evaluate(self, sampler, model):
        fake_features = []
        batches = num_to_groups(self.n_samples, self.batch_size)

        model.eval()

        for batch in batches:
            samples = sampler.sample(model, batch_size=batch)

            features = self._calculate_inception_features(samples)
            fake_features.append(features)

        fake_features = torch.cat(fake_features, dim=0)
        fake_features_all = self._gather_features(fake_features)

        fake_mu = fake_features_all.mean(dim=0)
        fake_sigma = torch.cov(fake_features_all.T)

        real_mu, real_sigma = self._load_or_precal_real_stats()

        fid = self._calculate_fid(fake_mu, fake_sigma, real_mu, real_sigma)

        return fid.item()

    def _calculate_fid(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
        """Adapted from Dougal J. Sutherland's implementation in numpy."""
        mu1, sigma1, mu2, sigma2 = mu1.float(), sigma1.float(), mu2.float(), sigma2.float()

        diff = mu1 - mu2
        try:
            covmean = sqrtm(sigma1 @ sigma2)
        except torch._C._LinAlgError:
            offset = torch.eye(sigma1.shape[0], device=self.rank) * eps
            covmean = sqrtm((sigma1 + offset) @ (sigma2 + offset))

        if torch.is_complex(covmean):
            if not torch.allclose(
                torch.diagonal(covmean).imag, torch.zeros_like(torch.diagonal(covmean).imag), atol=1e-3
            ):
                m = torch.max(torch.abs(covmean.imag))

            covmean = covmean.real

        tr_covmean = torch.trace(covmean)
        fid = (diff @ diff) + torch.trace(sigma1) + torch.trace(sigma2) - 2 * tr_covmean

        return fid

    def _calculate_inception_features(self, samples):
        samples = samples.half()

        if self.channels == 1:
            samples = repeat(samples, "b 1 ... -> b c ...", c=3)

        self.inception_v3.eval()
        features = self.inception_v3(samples)[0]

        if features.size(2) != 1 or features.size(3) != 1:
            features = adaptive_avg_pool2d(features, output_size=(1, 1))
        features = rearrange(features, "... 1 1 -> ...")
        return features

    def _gather_features(self, local_features):
        if self.world_size == 1:
            return local_features

        gathered = [torch.zeros_like(local_features) for _ in range(self.world_size)]
        dist.all_gather(gathered, local_features)
        return torch.cat(gathered, dim=0)

    @torch.inference_mode()
    def _load_or_precal_real_stats(self):
        path = os.path.join(self.stats_dir)
        try:
            stats = torch.load(path)
            return stats["mu"].to(self.rank), stats["sigma"].to(self.rank)

        except OSError:
            pass

        features_local = []

        for batch in self.dl:
            batch = batch.to(self.rank)
            features = self._calculate_inception_features(batch)
            features_local.append(features)

        features_local = torch.cat(features_local, dim=0)
        features_all = self._gather_features(features_local)

        mu = features.mean(dim=0)
        sigma = torch.cov(features_all.T)

        if self.rank == 0:
            torch.save({"mu": mu, "sigma": sigma}, path)

        return mu, sigma
