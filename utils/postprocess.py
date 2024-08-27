
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

def constraint_matrix(y):
    # 获取批次大小和序列长度
    batch_size, seq_length, _ = y.shape
    
    # 创建一个全 1 的矩阵
    matrix = torch.ones_like(y)
    
    # 创建一个掩码来标识不能形成依存关系的位置
    mask = torch.ones(seq_length, seq_length, device=y.device)
    for i in range(seq_length):
        st, en = max(i-3, 0), min(i+3, seq_length-1)
        mask[i, st:en+1] = 0
    
    # 将掩码应用到每个批次
    matrix = matrix * mask.unsqueeze(0)
    
    return matrix


def row_col_softmax(y):

    # 首先处理矩阵，只保留右上三角（不包括对角线）并使其对称
    # y_processed = process_upper_triangle(y)
    y_processed = y

    
    row_softmax = torch.softmax(y_processed, dim=-1)
    col_softmax = torch.softmax(y_processed, dim=-2)
    return 0.5 * (row_softmax + col_softmax)

def row_col_argmax(y):

    # y_pred = row_col_softmax(y)
    # y_pred = process_upper_triangle(y)
    # y_pred = row_col_softmax(y_pred)
    y_pred=y
    # torch.set_printoptions(edgeitems=y_pred.size(-1), 
    #         linewidth=1000)
    # print(y_pred)
    y_pred = y_pred * constraint_matrix(y_pred)

    # torch.set_printoptions(edgeitems=y_pred.size(-1), sci_mode=False, precision=1,
    #         linewidth=1000)
    # print(y_pred)

    y_hat = y_pred + torch.randn_like(y) * 1e-12
    col_max = torch.argmax(y_hat, 1)
    col_one = torch.zeros_like(y_hat).scatter(1, col_max.unsqueeze(1), 1.0)
    row_max = torch.argmax(y_hat, 2)
    row_one = torch.zeros_like(y_hat).scatter(2, row_max.unsqueeze(2), 1.0)
    int_one = row_one * col_one 

    return int_one

