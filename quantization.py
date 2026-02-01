import torch

def quantize_blockwise(tensor, bits=8, group_size=128):
    """
    [비대칭 양자화 구현]
    s = (Xmax - Xmin) / (2^n - 1)
    z = round(-Xmin / s)
    Q(X) = clamp(round(X/s) + z)
    """
    # 1. Reshape for grouping
    original_shape = tensor.shape
    numel = tensor.numel()
    
    pad_len = (group_size - (numel % group_size)) % group_size
    if pad_len > 0:
        tensor = torch.nn.functional.pad(tensor.flatten(), (0, pad_len))
    
    tensor_grouped = tensor.view(-1, group_size)
    
    # 2. Min/Max Calculation
    x_min = tensor_grouped.min(dim=1, keepdim=True)[0]
    x_max = tensor_grouped.max(dim=1, keepdim=True)[0]
    
    q_max = 2**bits - 1
    
    scale = (x_max - x_min) / q_max
    scale[scale == 0] = 1.0
    
    zero_point = torch.round(-x_min / scale)
    
    # 3. Quantize
    quantized_data = torch.round(tensor_grouped / scale) + zero_point
    
    # Clamp & Cast
    quantized_data = torch.clamp(quantized_data, 0, q_max).to(torch.int8)
    
    return quantized_data, scale, zero_point, original_shape, pad_len

def dequantize_blockwise(quantized_data, scales, zero_points, original_shape, pad_len):
    """
    압축 해제 (Dequantization)
    X_recon = (Q - Z) * S
    """
    # [FIX] INT8 데이터를 UINT8로 해석해야 함 (0~255 범위를 위해)
    # PyTorch int8은 signed(-128~127)이므로 view(uint8)로 해석 변경 후 float 변환
    q_float = quantized_data.view(torch.uint8).to(torch.float32)
    
    # 역양자화
    dequantized = (q_float - zero_points) * scales
    
    # Shape 복구
    dequantized = dequantized.flatten()
    if pad_len > 0:
        dequantized = dequantized[:-pad_len]
        
    return dequantized.view(original_shape)