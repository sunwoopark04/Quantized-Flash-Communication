import torch
import torch.distributed as dist
import time
from kernels import triton_quantize, triton_dequantize
from kernels_jit import pack_int4, unpack_int4 

def flash_all_reduce(tensor, group_size=128, bits=8, quant_type="asym"):
    torch.cuda.synchronize()
    start_time = time.perf_counter()

    if bits == 16:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()
        return tensor, (time.perf_counter() - start_time) * 1000

    world_size = dist.get_world_size()
    original_shape = tensor.shape
    tensor_flat = tensor.flatten()
    
    alignment = group_size * world_size
    pad_len = (alignment - (tensor_flat.numel() % alignment)) % alignment
    if pad_len > 0:
        tensor_flat = torch.nn.functional.pad(tensor_flat, (0, pad_len))
    
    chunks = list(torch.chunk(tensor_flat, world_size, dim=0))
    
    # ========================================================================
    # [Phase 1] Reduce-Scatter
    # ========================================================================
    phase1_bits = 4 if bits == 6 else bits
    send_q_list, send_s_list, send_z_list = [], [], []
    
    for chunk in chunks:
        # mode=quant_type 파라미터 전달
        q, s, z = triton_quantize(chunk, bits=phase1_bits, block_size=group_size, mode=quant_type)
        
        if phase1_bits == 4:
            q = pack_int4(q) # Minki님의 lop3/sub 적용 패킹 커널 (레이아웃만 담당하므로 공용)
            
        send_q_list.append(q.contiguous())
        send_s_list.append(s.contiguous())
        send_z_list.append(z.contiguous())
        
    recv_q_list = [torch.empty_like(q) for q in send_q_list]
    recv_s_list = [torch.empty_like(s) for s in send_s_list]
    recv_z_list = [torch.empty_like(z) for z in send_z_list]
    
    dist.all_to_all(recv_q_list, send_q_list)
    dist.all_to_all(recv_s_list, send_s_list)
    dist.all_to_all(recv_z_list, send_z_list)
    
    local_sum = torch.zeros_like(chunks[0], dtype=torch.float16)
    for i in range(world_size):
        q_recv = recv_q_list[i]
        if phase1_bits == 4:
            q_recv = unpack_int4(q_recv, chunks[0].shape)
        
        # Sym 모드일 경우 recv_z_list[i]에는 이미 8.0 상수가 들어있음
        local_sum += triton_dequantize(q_recv, recv_s_list[i], recv_z_list[i], chunks[0].shape, group_size)
        
    # ========================================================================
    # [Phase 2] All-Gather
    # ========================================================================
    if bits == 6:
        phase2_bits = 8
    else:
        phase2_bits = bits
    
    # mode=quant_type 파라미터 전달
    reduced_q, reduced_s, reduced_z = triton_quantize(local_sum, bits=phase2_bits, block_size=group_size, mode=quant_type)
    
    if phase2_bits == 4:
        reduced_q = pack_int4(reduced_q)
    
    gather_q_list = [torch.empty_like(reduced_q) for _ in range(world_size)]
    gather_s_list = [torch.empty_like(reduced_s) for _ in range(world_size)]
    gather_z_list = [torch.empty_like(reduced_z) for _ in range(world_size)]
    
    dist.all_gather(gather_q_list, reduced_q.contiguous())
    dist.all_gather(gather_s_list, reduced_s.contiguous())
    dist.all_gather(gather_z_list, reduced_z.contiguous())
    
    final_parts = []
    for i in range(world_size):
        q_gather = gather_q_list[i]
        if phase2_bits == 4:
            q_gather = unpack_int4(q_gather, chunks[0].shape)
            
        final_parts.append(triton_dequantize(q_gather, gather_s_list[i], gather_z_list[i], chunks[0].shape, group_size))
        
    output = torch.cat(final_parts, dim=0)
    if pad_len > 0: 
        output = output[:-pad_len]
    
    torch.cuda.synchronize()
    comm_time = (time.perf_counter() - start_time) * 1000
    return output.view(original_shape), comm_time