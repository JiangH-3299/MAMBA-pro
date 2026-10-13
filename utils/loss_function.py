import torch

class loss():
    def __init__(self, sigma):
        super(loss, self).__init__()
        
        self.sigma = sigma
    
    def softrank(self, scores):
        #scores: [n_samples]
        score = scores.unsqueeze(0) #[1,n_samples]
        pairwise_diff = score.unsqueeze(-1) - score.unsqueeze(-2)
        pairwise_prob = torch.sigmoid(pairwise_diff * self.sigma)
        soft_ranks = torch.sum(pairwise_prob, dim=-1) + 0.5
        
        return soft_ranks #[1,n_samples]
    
    def spearman_rank_correlation(self, soft_rank_pred, soft_rank_true):
        d = soft_rank_pred - soft_rank_true
        d_squared = torch.pow(d, 2)  # d_i^2
        n = soft_rank_pred.size(1)
        spearman_r = 1 - (6 * torch.sum(d_squared) / (n * (n**2 - 1)))
        
        return spearman_r
    
    def spearman_loss(self, y_true, y_pred):
        with torch.no_grad():
            soft_rank_true = self.softrank(y_true)
        soft_rank_pred = self.softrank(y_pred)
        
        spearman_r = self.spearman_rank_correlation(soft_rank_pred, soft_rank_true)
        spearman_loss = 1 - spearman_r
        
        return spearman_loss
    
    def grouped_spearman_loss(self, y_true, y_pred):
        y_true = y_true.to(y_pred.dtype)

        max_val = torch.max(y_true).ceil().item()
        intervals = torch.arange(
            start=0,
            end=max_val + 1,
            step=1,
            device=y_true.device,
            dtype=y_pred.dtype
        )
        
        total_loss = 0.0
        valid_groups = 0
        
        for i in range(len(intervals) - 1):
            lower = intervals[i]
            upper = intervals[i+1]
            
            mask = (y_true >= lower) & (y_true < upper)
            n_samples = torch.sum(mask)
            if n_samples < 2:
                continue

            y_true_group = y_true[mask]
            y_pred_group = y_pred[mask]
            group_loss = self.spearman_loss(y_true_group, y_pred_group)
            total_loss += group_loss
            valid_groups += 1
        
        if valid_groups == 0:
            return torch.tensor(0.0, device=y_true.device, requires_grad=True)
        
        return total_loss / valid_groups
    
    def rmse_loss(self, y_true, y_pred):
        mse = torch.mean((y_true - y_pred) ** 2)
        rmse = torch.sqrt(mse)
        
        return rmse
        
    