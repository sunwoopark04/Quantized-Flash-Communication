import torch
import triton
import triton.language as tl

# ==========================================
# 1. Asymmetric INT8 Kernel (기존 유지)
# ==========================================
@triton.jit
def quantize_kernel_asym_int8(x_ptr, q_ptr, s_ptr, z_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    
    max_val = tl.max(x, axis=0)
    min_val = tl.min(x, axis=0)
    q_max = 255.0
    scale = (max_val - min_val) / q_max
    scale = tl.where(scale < 1e-9, 1.0, scale)
    
    z_point = -min_val / scale
    z_point = tl.extra.cuda.libdevice.round(z_point)
    z_point = tl.clamp(z_point, 0.0, 255.0)
    
    q = tl.extra.cuda.libdevice.round(x / scale) + z_point
    q = tl.clamp(q, 0.0, 255.0)
    
    tl.store(q_ptr + offsets, q.to(tl.int8), mask=mask)
    tl.store(s_ptr + pid, scale.to(tl.float16))
    tl.store(z_ptr + pid, z_point.to(tl.float16))

# ==========================================
# 2. Asymmetric INT4 Kernel (기존 유지)
# ==========================================
@triton.jit
def quantize_kernel_asym_int4(x_ptr, q_ptr, s_ptr, z_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    
    max_val = tl.max(x, axis=0)
    min_val = tl.min(x, axis=0)
    q_max = 15.0
    scale = (max_val - min_val) / q_max
    scale = tl.where(scale < 1e-9, 1.0, scale)
    
    z_point = -min_val / scale
    z_point = tl.extra.cuda.libdevice.round(z_point)
    z_point = tl.clamp(z_point, 0.0, 15.0)
    
    q = tl.extra.cuda.libdevice.round(x / scale) + z_point
    q = tl.clamp(q, 0.0, 15.0)
    
    tl.store(q_ptr + offsets, q.to(tl.int8), mask=mask)
    tl.store(s_ptr + pid, scale.to(tl.float16))
    tl.store(z_ptr + pid, z_point.to(tl.float16))

# ==========================================
# [New] 3. Symmetric INT4 Kernel (추가)
# ==========================================
@triton.jit
def quantize_kernel_sym_int4(x_ptr, q_ptr, s_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    
    # Symmetric Logic: 절대값 최대치 기준
    abs_max = tl.max(tl.abs(x), axis=0)
    scale = abs_max / 7.5 # -8 ~ 7 범위를 위한 스케일
    scale = tl.where(scale < 1e-9, 1.0, scale)
    
    # 영점(Zero-point)을 8.0으로 고정하여 대칭성 유지
    q = tl.extra.cuda.libdevice.round(x / scale) + 8.0
    q = tl.clamp(q, 0.0, 15.0)
    
    tl.store(q_ptr + offsets, q.to(tl.int8), mask=mask)
    tl.store(s_ptr + pid, scale.to(tl.float16))

# ==========================================
# 4. Wrapper (Mode 분기 추가)
# ==========================================
def triton_quantize(x, bits=8, block_size=128, mode="asym"):
    if not x.is_contiguous(): x = x.contiguous()
    n_elements = x.numel()
    n_blocks = triton.cdiv(n_elements, block_size)
    q_x = torch.empty(n_elements, dtype=torch.int8, device=x.device)
    s_x = torch.empty(n_blocks, dtype=torch.float16, device=x.device)
    z_x = torch.zeros(n_blocks, dtype=torch.float16, device=x.device)
    
    grid = (n_blocks,)
    if mode == "asym":
        if bits == 8:
            quantize_kernel_asym_int8[grid](x, q_x, s_x, z_x, n_elements, BLOCK_SIZE=block_size)
        elif bits <= 4:
            quantize_kernel_asym_int4[grid](x, q_x, s_x, z_x, n_elements, BLOCK_SIZE=block_size)
    else: # Symmetric Mode
        if bits <= 4:
            quantize_kernel_sym_int4[grid](x, q_x, s_x, n_elements, BLOCK_SIZE=block_size)
            z_x.fill_(8.0) # 역양자화 시 8.0을 상수로 사용하도록 설정
    return q_x, s_x, z_x

# ==========================================
# 5. Asymmetric Dequantize (기존 유지)
# ==========================================
@triton.jit
def dequantize_kernel_asym(q_ptr, s_ptr, z_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    q = tl.load(q_ptr + offsets, mask=mask, other=0.0).to(tl.uint8).to(tl.float32)
    s = tl.load(s_ptr + pid).to(tl.float32)
    z = tl.load(z_ptr + pid).to(tl.float32)
    
    # (Q - Z) * S -> Asym/Sym 공통 수식 (Sym은 Z가 8.0 상수가 됨)
    out = (q - z) * s
    tl.store(out_ptr + offsets, out.to(tl.float16), mask=mask)

def triton_dequantize(q_x, s_x, z_x, original_shape, block_size=128):
    if not q_x.is_contiguous(): q_x = q_x.contiguous()
    n_elements = q_x.numel()
    out = torch.empty(n_elements, dtype=torch.float16, device=q_x.device)
    grid = (triton.cdiv(n_elements, block_size),)
    dequantize_kernel_asym[grid](q_x, s_x, z_x, out, n_elements, BLOCK_SIZE=block_size)
    return out.view(original_shape)