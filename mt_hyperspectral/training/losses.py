import torch
import torch.nn as nn

class CorrelatedWeightedMSELoss(nn.Module):
    """权重损失函数，考虑变量之间的相关性"""
    
    def __init__(self, weights=None, correlation_penalty=0.1, output_size=15):
        super(CorrelatedWeightedMSELoss, self).__init__()
        self.weights = weights if weights is not None else torch.ones(output_size)
        self.correlation_penalty = correlation_penalty
        self.mse = nn.MSELoss(reduction='none')
        
    def forward(self, pred, target):
        if self.weights.device != pred.device:
            self.weights = self.weights.to(pred.device)
        
        # 基本的MSE损失
        base_loss = self.mse(pred, target)
        weighted_loss = base_loss * self.weights.view(1, -1)
        
        # 计算预测输出之间的相关性损失
        batch_size = pred.size(0)
        if batch_size > 1:
            # 计算预测输出之间的协方差矩阵
            centered_pred = pred - pred.mean(dim=0, keepdim=True)
            covar = torch.matmul(centered_pred.t(), centered_pred) / (batch_size - 1)
            
            # 获取对角线以外的元素（相关系数）
            mask = torch.ones_like(covar) - torch.eye(covar.size(0), device=covar.device)
            correlation_loss = torch.sum(torch.abs(covar * mask)) / (covar.size(0) * (covar.size(0) - 1))
            
            # 结合基本损失和相关性损失
            total_loss = weighted_loss.mean() + self.correlation_penalty * correlation_loss
            return total_loss
        else:
            return weighted_loss.mean()
