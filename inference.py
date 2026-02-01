import torch
import time

def run_qa_test(model, tokenizer, device, rank):
    """
    [QA Test] 실제 대화 테스트 및 속도 측정
    """
    # 논문 재현 설정 (Flash INT6)
    for layer in model.model.layers:
        if hasattr(layer.mlp, "mode"):
            layer.mlp.mode = "flash"
            layer.mlp.bits = 6 
        
    prompt = "Explain london."
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    attention_mask = torch.ones_like(input_ids)
    
    if rank == 0:
        print(f"\n📝 [QA Test] Prompt: {prompt}")
        
    torch.cuda.synchronize()
    
    # [추가] 매번 다른 답변이 나오도록, QA 테스트 직전에 시드를 현재 시간으로 변경
    torch.manual_seed(int(time.time())) 
    
    start_t = time.perf_counter()
    
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids, 
            attention_mask=attention_mask, 
            max_new_tokens=200,  # [유지] 답변이 끊기지 않게 넉넉히 설정
            do_sample=True, 
            temperature=0.7, 
            pad_token_id=tokenizer.pad_token_id
        )
    
    torch.cuda.synchronize()
    total_time = (time.perf_counter() - start_t) * 1000
        
    if rank == 0:
        output_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        final_answer = output_text[len(prompt):].strip()
        
        print(f"💡 Answer:\n{final_answer}")
        print(f"🚀 Total Generation Time: {total_time:.2f}ms")