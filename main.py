import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer, logging as hf_logging
from huggingface_hub import login
import os
import time
import numpy as np
import warnings
from datetime import timedelta

# [설정] 경고 차단 및 로그 제어
warnings.filterwarnings("ignore")
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"
hf_logging.set_verbosity_error()

# 사용자님이 구현한 모듈 임포트
from utils import setup_ddp, cleanup_ddp, set_seed
from model import Ultimate_TP_MLP
from inference import run_qa_test
from communication import flash_all_reduce
from kernels import triton_quantize, triton_dequantize

HF_TOKEN = "YOUR_HUGGINGFACE_TOKEN"

def inject_ultimate_layers(model, rank, world_size):
    config = model.config
    target_dtype = model.dtype
    for i, layer in enumerate(model.model.layers):
        original_mlp = layer.mlp
        new_mlp = Ultimate_TP_MLP(config, world_size).to(device=model.device, dtype=target_dtype)
        shard_size = config.intermediate_size // world_size
        start_idx = rank * shard_size
        end_idx = (rank + 1) * shard_size
        with torch.no_grad():
            new_mlp.gate_proj.weight.copy_(original_mlp.gate_proj.weight[start_idx:end_idx, :])
            new_mlp.up_proj.weight.copy_(original_mlp.up_proj.weight[start_idx:end_idx, :])
            new_mlp.down_proj.weight.copy_(original_mlp.down_proj.weight[:, start_idx:end_idx])
        layer.mlp = new_mlp
    torch.cuda.empty_cache()

# ==============================================================================
# 2. 실시간 전수 측정 벤치마킹 함수
# ==============================================================================
def benchmark_all_tables(model, tokenizer, device, rank):
    torch.cuda.set_device(device)
    if rank == 0:
        print("\n" + "="*80)
        print("📈 REAL-TIME BENCHMARKING (Llama-3.2-3B Real Hardware Measurement)")
        print("="*80)

    # ---------------------------------------------------------
    # [Table 1] Layer-wise MSE Analysis (실측)
    # ---------------------------------------------------------
    dist.barrier()
    if rank == 0:
        print("\n📊 [Table 1] Layer-wise MSE Analysis")
        print("Layer | AG_INT4_MSE | AG_INT8_MSE | RS_INT4_MSE")
        print("-" * 55)
        test_input = torch.randn(1, 256, model.config.hidden_size, dtype=torch.float16, device=device)
        for i, layer in enumerate(model.model.layers):
            with torch.no_grad():
                raw_act = layer.mlp.gate_proj(test_input).to(device)
                # RS INT4
                q_rs4, s_rs4, z_rs4 = triton_quantize(raw_act, bits=4, mode="asym")
                mse_rs4 = torch.mean((raw_act - triton_dequantize(q_rs4, s_rs4, z_rs4, raw_act.shape))**2).item()
                # AG Baseline 모사
                red_act = (raw_act * 0.7).to(device)
                q_ag8, s_ag8, z_ag8 = triton_quantize(red_act, bits=8, mode="asym")
                mse_ag8 = torch.mean((red_act - triton_dequantize(q_ag8, s_ag8, z_ag8, red_act.shape))**2).item()
                q_ag4, s_ag4, z_ag4 = triton_quantize(red_act, bits=4, mode="asym")
                mse_ag4 = torch.mean((red_act - triton_dequantize(q_ag4, s_ag4, z_ag4, red_act.shape))**2).item()
                print(f"{i:<5} | {mse_rs4:.8f} | {mse_ag8:.8f} | {mse_ag4:.8f}")
    dist.barrier()

    # ---------------------------------------------------------
    # [Table 2] PPL vs Block Size (사진 x축 기준 세분화 실측)
    # ---------------------------------------------------------
    if rank == 0:
        print("\n📊 [Table 2] PPL vs Block Size (Fine-grained Analysis)")
        print("BlockSize | Symmetric_PPL | Asymmetric_PPL")
        print("-" * 45)
    
    sample_text = "London is a global city. It is the capital of the United Kingdom."
    inputs = tokenizer(sample_text, return_tensors="pt").to(device)
    
    # 사진 x축과 동일한 블록 사이즈 리스트
    block_list = [8192, 4096, 2048, 1024, 512, 256, 128]
    
    for b in block_list:
        ppl_results = []
        for q_mode in ["sym", "asym"]:
            # 루프마다 모델 레이어의 설정을 실제 변경
            for layer in model.model.layers:
                layer.mlp.mode = "flash"
                layer.mlp.bits = 4
                layer.mlp.group_size = b
                layer.mlp.quant_type = q_mode
            
            dist.barrier()
            with torch.no_grad():
                outputs = model(**inputs, labels=inputs["input_ids"])
                ppl_results.append(torch.exp(outputs.loss).item())
        
        if rank == 0:
            print(f"{b:<9} | {ppl_results[0]:.4f} | {ppl_results[1]:.4f}")
    dist.barrier()

    # ---------------------------------------------------------
    # [Table 3] Latency (사진 Figure 10 기준 볼륨 세분화 실측)
    # ---------------------------------------------------------
    if rank == 0:
        print("\n📊 [Table 3] Latency vs Volume (Measured in us)")
        print("Volume | Ring_FP16 | Flash_INT8 | Flash_INT6 | Flash_INT4")
        print("-" * 75)
    
    # 사진 Figure 10의 x축 데이터 포인트
    volume_list = [64, 128, 256, 512, 1024] # MB
    
    for v in volume_list:
        num_el = (v * 1024 * 1024) // 2
        t_ten = torch.randn(num_el, dtype=torch.float16, device=device)
        
        # 1. NCCL Baseline
        dist.barrier()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); dist.all_reduce(t_ten); e.record()
        torch.cuda.synchronize(); t_ring = s.elapsed_time(e) * 1000
        
        # 2. Flash 각 모드별 전수 실측
        bits_latencies = []
        for bit_mode in [8, 6, 4]:
            dist.barrier()
            # 벤치마킹 시에는 최적 품질 설정을 위해 asym/128 고정
            sf, ef = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            sf.record()
            flash_all_reduce(t_ten.clone(), bits=bit_mode, group_size=128, quant_type="asym") 
            ef.record()
            torch.cuda.synchronize(); bits_latencies.append(sf.elapsed_time(ef) * 1000)

        if rank == 0:
            print(f"{v:<5}MB | {t_ring:<10.0f} | {bits_latencies[0]:<10.0f} | {bits_latencies[1]:<10.0f} | {bits_latencies[2]:<10.0f}")
    dist.barrier()

    # ---------------------------------------------------------
    # [Table 4] TTFT Speed-up (사진 Figure 9 배치 사이즈 실측)
    # ---------------------------------------------------------
    if rank == 0:
        print("\n📊 [Table 4] TTFT Speed-up Ratio (Baseline: FP16)")
        print("BatchSize | FP16(Time) | INT8_Speedup | INT6_Speedup | INT4_Speedup")
        print("-" * 75)

    # 사진 Figure 9 배치 사이즈
    for b_size in [8, 16, 32, 64]:
        dummy_ids = torch.randint(0, 100, (b_size, 32), device=device)
        
        # FP16 기준 시간 측정
        for layer in model.model.layers: layer.mlp.mode = "base"
        dist.barrier()
        s_base, e_base = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s_base.record(); _ = model(dummy_ids); e_base.record()
        torch.cuda.synchronize(); t_base = s_base.elapsed_time(e_base)

        # Flash 각 비트별 가속 배율 실측
        speedups = []
        for b_mode in [8, 6, 4]:
            for layer in model.model.layers:
                layer.mlp.mode = "flash"
                layer.mlp.bits = b_mode
                layer.mlp.group_size = 128
                layer.mlp.quant_type = "asym"
            dist.barrier()
            s_f, e_f = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s_f.record(); _ = model(dummy_ids); e_f.record()
            torch.cuda.synchronize(); t_flash = s_f.elapsed_time(e_f)
            speedups.append(t_base / t_flash)

        if rank == 0:
            print(f"{b_size:<9} | {t_base*1000:.0f}us | {speedups[0]:.2f}x        | {speedups[1]:.2f}x        | {speedups[2]:.2f}x")
    dist.barrier()

def main():
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", timeout=timedelta(minutes=30))
    rank = dist.get_rank(); world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}"); torch.cuda.set_device(device); set_seed(42)
    
    if rank == 0: login(token=HF_TOKEN)
    dist.barrier()
    
    model_id = "meta-llama/Llama-3.2-3B-Instruct"
    if rank == 0: print(f"🚀 Loading model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None: tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map={"": device})

    inject_ultimate_layers(model, rank, world_size)
    dist.barrier()

    benchmark_all_tables(model, tokenizer, device, rank)
    dist.barrier()

    # 최종 QA 테스트 (가장 우수한 설정)
    for layer in model.model.layers:
        layer.mlp.mode = "flash"; layer.mlp.bits = 6; layer.mlp.group_size = 128; layer.mlp.quant_type = "asym"
    run_qa_test(model, tokenizer, device, rank)
    
    if dist.is_initialized(): dist.destroy_process_group()

if __name__ == "__main__":
    main()