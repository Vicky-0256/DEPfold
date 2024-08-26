
#code copy from  https://github.com/A4Bio/RFold/blob/master/rfold.py


import torch

def process_upper_triangle(y):
    # 确保y是三维的，第一维是batch
    assert y.dim() == 3, "Input tensor should be 3-dimensional (batch, seq_len, seq_len)"
    
    # 创建一个严格的上三角掩码（不包括对角线）
    mask = torch.triu(torch.ones_like(y[0]), diagonal=1).unsqueeze(0)
    
    # 应用掩码，将对角线和下三角部分置为0
    y_upper = y * mask
    
    # 使矩阵对称
    # y_symmetric = y_upper + y_upper.transpose(-1, -2)
    y_symmetric = y_upper
    
    return y_symmetric

def row_col_softmax(y):

    # 首先处理矩阵，只保留右上三角（不包括对角线）并使其对称
    # y_processed = process_upper_triangle(y)
    y_processed = y
    
    row_softmax = torch.softmax(y_processed, dim=-1)
    col_softmax = torch.softmax(y_processed, dim=-2)
    return 0.5 * (row_softmax + col_softmax)

def row_col_argmax(y):
    y_pred = row_col_softmax(y)
    y_pred = process_upper_triangle(y_pred)


    y_hat = y_pred + torch.randn_like(y) * 1e-12
    col_max = torch.argmax(y_hat, 1)
    col_one = torch.zeros_like(y_hat).scatter(1, col_max.unsqueeze(1), 1.0)
    row_max = torch.argmax(y_hat, 2)
    row_one = torch.zeros_like(y_hat).scatter(2, row_max.unsqueeze(2), 1.0)
    int_one = row_one * col_one 

    return int_one

