import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
import time
from communication import flash_all_reduce

class Ultimate_TP_MLP(nn.Module):
    """ 실제 Llama 모델의 MLP 레이어를 대체하여 통신 가속을 주입하는 클래스 """
    def __init__(self, config, world_size):
        super().__init__()
        self.world_size = world_size
        
        # 기본값 설정: 일반적인 환경에서는 항상 Asymmetric(비대칭)과 128 Group Size를 사용
        self.mode = "base" 
        self.bits = 16 
        self.group_size = 128   # [추가] 벤치마크 세분화를 위한 변수
        self.quant_type = "asym" # [추가] 기본값을 Asymmetric으로 고정하여 안전성 확보
        
        self.hidden_size = config.hidden_size
        self.shard_size = config.intermediate_size // world_size
        
        # 실제 모델의 가중치를 주입받을 레이어 정의
        self.gate_proj = nn.Linear(self.hidden_size, self.shard_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.shard_size, bias=False)
        self.down_proj = nn.Linear(self.shard_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()
        
        self.last_comm_latency = 0.0

    def forward(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        intermediate = self.act_fn(gate) * up
        partial_output = self.down_proj(intermediate)
        
        # 모드에 따른 분기 처리
        if self.mode == "flash":
            # [수정] communication.py의 실제 로직에 맞춰 파라미터를 명시적으로 전달
            # 벤치마킹 루프에서 이 속성들을 변경하면 실시간으로 반영됨
            output, comm_time = flash_all_reduce(
                partial_output, 
                group_size=self.group_size, 
                bits=self.bits, 
                quant_type=self.quant_type
            )
            self.last_comm_latency = comm_time

        else:
            # Baseline: 표준 NCCL All-Reduce (FP16)
            torch.cuda.synchronize()
            start = time.perf_counter()
            dist.all_reduce(partial_output, op=dist.ReduceOp.SUM)
            torch.cuda.synchronize()
            self.last_comm_latency = (time.perf_counter() - start) * 1000
            output = partial_output
            
        return output