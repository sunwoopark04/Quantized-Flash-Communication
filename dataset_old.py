    import torch
    from torch.utils.data import Dataset, DataLoader
    from torch.utils.data.distributed import DistributedSampler

    class SyntheticTextDataset(Dataset):
        """
        Llama-3 입력을 흉내 내는 가상의 데이터셋
        (Batch Size, Sequence Length, Hidden Dimension)
        """
        def __init__(self, num_samples=1000, seq_len=128, hidden_size=4096):
            self.num_samples = num_samples
            self.seq_len = seq_len
            self.hidden_size = hidden_size

        def __len__(self):
            return self.num_samples

        def __getitem__(self, idx):
            # 랜덤한 벡터 생성 (실제 텍스트 대신 노이즈 사용)
            return torch.randn(self.seq_len, self.hidden_size)

    def get_dataloader(batch_size=32, seq_len=128, hidden_size=4096):
        """
        멀티 GPU용 DataLoader 생성
        """
        dataset = SyntheticTextDataset(num_samples=1000, seq_len=seq_len, hidden_size=hidden_size)
        
        # [핵심] DistributedSampler
        # 전체 데이터를 4등분해서 각 GPU에 서로 다른 데이터를 나눠주는 역할
        sampler = DistributedSampler(dataset, shuffle=False)
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,      # Sampler가 섞어주므로 shuffle=False
            num_workers=4,        # 데이터 로딩 속도 가속
            pin_memory=True,      # CPU -> GPU 전송 속도 향상
            drop_last=True        # 배치 크기가 안 맞으면 마지막 버림
        )
        
        return dataloader