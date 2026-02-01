import torch
from torch.utils.cpp_extension import load_inline

# ==============================================================================
# 1. C++ / CUDA Source
# ==============================================================================
cpp_source = """
torch::Tensor pack_int4_cuda(torch::Tensor input);
torch::Tensor unpack_int4_cuda(torch::Tensor input, int original_numel);
"""

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h> // [Add] FP16 연산을 위해 필수

// [Keep] Packing 커널은 기존 그대로 유지
__global__ void pack_int4_paper_kernel(const int8_t* __restrict__ in, int32_t* __restrict__ out, int n_out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_out) {
        uint32_t u0 = 0, u1 = 0;
        #pragma unroll
        for (int i=0; i<4; ++i) u1 |= ((uint32_t)(in[idx*8 + i] & 0x0F) << (i*8));
        #pragma unroll
        for (int i=0; i<4; ++i) u0 |= ((uint32_t)(in[idx*8 + 4 + i] & 0x0F) << (i*8));
        u0 |= (u0 >> 12);
        u1 |= (u1 >> 12);
        out[idx] = (int32_t)__byte_perm(u1, u0, 0x5140);
    }
}

__device__ __forceinline__ half2 magic_convert(uint32_t val) {
    uint32_t result_bits;
    // Magic Numbers: 0x64006400 represents 1024.0 in FP16 (two times)
    const uint32_t magic_num = 0x64006400; 
    const uint32_t mask = 0x000F000F;

    asm volatile(
        "lop3.b32 %0, %1, %2, %3, 0xEA;" 
        : "=r"(result_bits) 
        : "r"(val), "r"(mask), "r"(magic_num)
    );

    // 2. sub.f16x2 (Subtract Magic Number)
    // Result = (1024.0 + w) - 1024.0
    half2 magic_h2 = *reinterpret_cast<const half2*>(&magic_num);
    half2 packed_h2 = *reinterpret_cast<half2*>(&result_bits);
    
    // __hsub2 compiles to sub.f16x2 instruction
    return __hsub2(packed_h2, magic_h2);
}


__global__ void unpack_int4_paper_kernel(const int32_t* __restrict__ in, half* __restrict__ out, int n_in) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n_in) {
        uint32_t packed = (uint32_t)in[idx];
        
        half2 res0 = magic_convert(packed >> 0);  // w1, w0
        half2 res1 = magic_convert(packed >> 4);  // w3, w2
        half2 res2 = magic_convert(packed >> 8);  // w5, w4
        half2 res3 = magic_convert(packed >> 12); // w7, w6

        half2* out_ptr = reinterpret_cast<half2*>(out + idx * 8);
        out_ptr[0] = res0;
        out_ptr[1] = res1;
        out_ptr[2] = res2;
        out_ptr[3] = res3;
    }
}

torch::Tensor pack_int4_cuda(torch::Tensor input) {
    auto n_out = input.numel() / 8;
    auto out = torch::empty({n_out}, torch::dtype(torch::kInt32).device(input.device()));
    int threads = 256;
    int blocks = (n_out + threads - 1) / threads;
    pack_int4_paper_kernel<<<blocks, threads>>>((int8_t*)input.data_ptr(), out.data_ptr<int32_t>(), n_out);
    return out;
}

torch::Tensor unpack_int4_cuda(torch::Tensor input, int original_numel) {
    // [Modify] 입력은 packed tensor(int32), 출력은 FP16 tensor(half)
    auto n_packed = input.numel(); 
    // 출력은 원본 개수만큼의 FP16 (kHalf)
    auto out = torch::empty({original_numel}, torch::dtype(torch::kHalf).device(input.device()));
    
    int threads = 256;
    int blocks = (n_packed + threads - 1) / threads;
    
    // n_packed 개수만큼 스레드 실행 (1스레드가 8개 data 처리)
    unpack_int4_paper_kernel<<<blocks, threads>>>(
        input.data_ptr<int32_t>(), 
        (half*)out.data_ptr(), // cast to half* (pybind11 handle detail)
        n_packed
    );
    return out;
}
"""

# ==============================================================================
# 2. Compile & Wrapper
# ==============================================================================
print("⏳ Loading JIT Kernels (with lop3 & sub.f16x2)...", end="", flush=True)
flash_ops = load_inline(
    name='flash_ops_paper_real_v1',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['pack_int4_cuda', 'unpack_int4_cuda'],
    verbose=False 
)
print(" Done.")

def pack_int4(q_tensor):
    numel = q_tensor.numel()
    pad_len = (8 - (numel % 8)) % 8
    if pad_len > 0:
        q_tensor = torch.nn.functional.pad(q_tensor.flatten(), (0, pad_len))
    return flash_ops.pack_int4_cuda(q_tensor.contiguous())

def unpack_int4(packed_tensor, original_shape):
    original_numel = original_shape.numel()
    # Unpack 결과가 이제 FP16으로 나옵니다.
    unpacked = flash_ops.unpack_int4_cuda(packed_tensor.contiguous(), packed_tensor.numel() * 8)
    return unpacked[:original_numel].view(original_shape)

# ==============================================================================
# 3. Unit Test
# ==============================================================================
if __name__ == "__main__":
    def test_implementation():
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == 'cpu': 
            print("❌ CUDA not available, skipping test.")
            return

        # 1. Input Data (INT8 Range 0~15)
        input_data = torch.randint(0, 16, (128, 64), dtype=torch.int8, device=device)
        
        # 2. Pack (INT8 -> INT32 Packed)
        packed = pack_int4(input_data)
        
        # 3. Unpack (INT32 Packed -> FP16 Dequantized)
        # [중요] 논문의 Fast Dequantization 결과는 FP16(실수)입니다.
        unpacked_fp16 = unpack_int4(packed, input_data.shape)
        
        # 4. Verify
        # 비교를 위해 FP16 결과를 다시 INT8로 변환합니다.
        # (5.0 -> 5)
        unpacked_int8 = unpacked_fp16.to(torch.int8)

        if torch.equal(input_data, unpacked_int8):
            print(f"✅ Fast Dequantization Test Passed!")
            print(f"   Input shape: {input_data.shape} (INT8)")
            print(f"   Output shape: {unpacked_fp16.shape} (FP16)")
            print(f"   Example Input: {input_data[0, :8].tolist()}")
            print(f"   Example Output: {unpacked_fp16[0, :8].tolist()}")
        else:
            print("❌ Kernel Test Failed")
            print("Input:", input_data[0, :8])
            print("Output:", unpacked_int8[0, :8])

    test_implementation()