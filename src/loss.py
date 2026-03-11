from torch import Tensor
from src import dist_utils
import torch.distributed as dist
import torch
import torch.nn.functional as F


class SimpleContrastiveLoss:
    def __init__(self, temperature: float = 0.02):
        self.temperature = temperature

    def __call__(self, x: Tensor, y: Tensor, target: Tensor = None, reduction: str = 'mean') -> Tensor:
        if target is None:
            target_per_qry = y.size(0) // x.size(0)
            target = torch.arange(
                0, x.size(0) * target_per_qry, target_per_qry, device=x.device, dtype=torch.long)
        logits = torch.matmul(x, y.transpose(0, 1))
        loss = F.cross_entropy(logits / self.temperature, target, reduction=reduction)
        return loss


class DistributedContrastiveLoss(SimpleContrastiveLoss):
    def __init__(self, n_target: int = 0, scale_loss: bool = True, temperature: float = 0.02):
        assert dist.is_initialized(), "Distributed training has not been properly initialized."
        super().__init__()
        self.word_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.scale_loss = scale_loss
        self.temperature = temperature

    def __call__(self, x: Tensor, y: Tensor, **kwargs):
        dist_x = self.gather_tensor(x)
        dist_y = self.gather_tensor(y)
        loss = super().__call__(dist_x, dist_y, **kwargs)
        if self.scale_loss:
            loss = loss * self.word_size
        return loss

    def gather_tensor(self, t):
        gathered = [torch.empty_like(t) for _ in range(self.word_size)]
        dist.all_gather(gathered, t)
        gathered[self.rank] = t
        return torch.cat(gathered, dim=0)

class InExampleContrastiveLoss:
    """
    Categorization loss: cross_entropy of 1 out of K classes (target labels)
    x.shape=[bsz, hdim], y.shape=[bsz, num_label, hdim]
    """

    def __init__(self, n_hard_negatives: int = 0, temperature: float = 1.0, ndim: int = None, *args, **kwargs):
        self.target_per_qry = n_hard_negatives + 1
        self.temperature = temperature
        self.ndim = ndim

    def __call__(self, x: Tensor, y: Tensor, reduction: str = 'mean'):
        # print("gather InExampleContrastiveLoss")
        if torch.distributed.is_initialized():
            x = dist_utils.dist_gather(x)
            y = dist_utils.dist_gather(y)
        bsz, ndim = x.size(0), x.size(1)
        target = torch.zeros(bsz, dtype=torch.long, device=x.device)
        if self.ndim:
            ndim = self.ndim
            x = x[:, :ndim]
            y = y[:, :ndim]
        logits = torch.einsum('bod,bsd->bs', x.view(bsz, 1, ndim), y.view(bsz, -1, ndim)) * self.temperature
        preds = torch.argmax(logits, dim=-1)
        loss = F.cross_entropy(logits, target, reduction=reduction)
        loss_detail = {"logits": logits, "labels": target, "preds": preds}
        return loss, loss_detail

class MultiCategorySupConLoss:
    """
    Multi-category Supervised Contrastive Learning with Weighted Mask.
    Uses Jaccard Similarity (IoU) between multi-hot labels as weights.

    For GradCache compatibility: takes (x, y) where x is qry_reps and y is tgt_reps,
    but uses only x (qry_reps) for SupConLoss computation.
    
    Combines MultiCategorySupConLoss with SimpleContrastiveLoss:
    loss = supcon_loss_weight * supcon_loss + (1 - supcon_loss_weight) * contrastive_loss
    """
    def __init__(self, temperature: float = 0.07, base_temperature: float = 0.07,
                 score_threshold: int = 3, supcon_loss_weight: float = 1.0, contrastive_temperature: float = 0.02):
        """
        Args:
            temperature: Temperature parameter for contrastive loss
            base_temperature: Base temperature for scaling
            score_threshold: Threshold for converting scores to binary (scores >= threshold -> 1)
            supcon_loss_weight: Weight for MultiCategorySupConLoss (default: 1.0, only SupConLoss)
            contrastive_temperature: Temperature for SimpleContrastiveLoss
        """
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.score_threshold = score_threshold
        self.supcon_loss_weight = supcon_loss_weight
        self.contrastive_loss = SimpleContrastiveLoss(temperature=contrastive_temperature)

    def _scores_to_multihot(self, category_scores: Tensor) -> Tensor:
        """
        Convert category scores to multi-hot encoding.

        Args:
            category_scores: [bsz, num_categories] - scores for each category (0-5)

        Returns:
            [bsz, num_categories] - binary multi-hot encoding (scores > threshold -> 1)
        """
        # Convert scores to binary: score >= threshold means category is present
        return (category_scores >= self.score_threshold).float()

    def __call__(self, x: Tensor, y: Tensor, labels: Tensor = None, **kwargs):
        """
        Args:
            x: [bsz, hidden_dim] - query representations (used for SupConLoss)
            y: [bsz, hidden_dim] - target representations (not used, kept for compatibility)
            labels: [bsz, num_categories] - category scores (0-5 integers) for each category
                   OR [bsz] if single category score (will be converted to multi-hot)
        """
        if labels is None:
            raise ValueError("MultiCategorySupConLoss requires 'labels' in loss_kwargs")

        features = x  # Use query representations
        device = features.device
        batch_size = features.shape[0]

        # Convert labels to multi-hot encoding if needed
        if labels.dim() == 1:
            labels = labels.view(-1, 1)

        # Ensure labels is [bsz, num_categories]
        if labels.dim() != 2:
            raise ValueError(f"labels should be [bsz, num_categories] or [bsz], got shape {labels.shape}")

        # Convert scores to multi-hot binary encoding
        multihot_labels = self._scores_to_multihot(labels)

        # 1. Weighted Mask (Jaccard Similarity)
        # Intersection: [bsz, bsz]
        intersection = torch.matmul(multihot_labels, multihot_labels.T)

        # Union: |A| + |B| - |A ∩ B|
        row_sums = multihot_labels.sum(dim=1).view(-1, 1)
        union = row_sums + row_sums.T - intersection

        # Jaccard Index (IoU)
        mask = intersection / (union + 1e-8)

        # 2. Logits
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature
        )

        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Self-contrast mask
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # 3. Log Prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)

        # 4. Weighted mean of log-likelihood over positives
        mask_pos_weights = mask.sum(1)
        mask_pos_weights = torch.where(mask_pos_weights < 1e-6, torch.ones_like(mask_pos_weights), mask_pos_weights)

        weighted_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_weights

        supcon_loss = -(self.temperature / self.base_temperature) * weighted_log_prob_pos
        supcon_loss = supcon_loss.mean()

        # Combine with SimpleContrastiveLoss if supcon_loss_weight < 1.0
        if self.supcon_loss_weight < 1.0:
            contrastive_loss = self.contrastive_loss(x, y, **kwargs)
            loss = self.supcon_loss_weight * supcon_loss + (1.0 - self.supcon_loss_weight) * contrastive_loss
        else:
            loss = supcon_loss

        return loss
class DistributedMultiCategorySupConLoss(MultiCategorySupConLoss):
    """Distributed version of MultiCategorySupConLoss"""
    def __init__(self, temperature: float = 0.07, base_temperature: float = 0.07,
                 scale_loss: bool = True, score_threshold: int = 3,
                 supcon_loss_weight: float = 1.0, contrastive_temperature: float = 0.02):
        super().__init__(temperature, base_temperature, score_threshold,
                        supcon_loss_weight, contrastive_temperature)
        if dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
        else:
            self.world_size = 1
            self.rank = 0
        self.scale_loss = scale_loss
        # Replace SimpleContrastiveLoss with DistributedContrastiveLoss for distributed training
        if dist.is_initialized():
            self.contrastive_loss = DistributedContrastiveLoss(
                n_target=0, scale_loss=scale_loss, temperature=contrastive_temperature
            )

    def __call__(self, x: Tensor, y: Tensor, labels: Tensor = None, **kwargs):
        if not dist.is_initialized():
            return super().__call__(x, y, labels, **kwargs)

        dist_x = self.gather_tensor(x)
        dist_y = self.gather_tensor(y)
        dist_labels = self.gather_tensor(labels) if labels is not None else None

        loss = super().__call__(dist_x, dist_y, dist_labels, **kwargs)

        if self.scale_loss:
            loss = loss * self.world_size
        return loss

    def gather_tensor(self, t):
        gathered = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(gathered, t)
        gathered[self.rank] = t
        return torch.cat(gathered, dim=0)