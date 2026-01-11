from transformers import AutoTokenizer, AutoModelForCausalLM
from specSampling import specSampling, specSampling_new, specSampling_new_multi
from regrSampling import regrSampling
from PPL import calculate_ppl_regr, calculate_ppl_spec
from datasets import load_dataset
from dataclasses import dataclass
import torch
import time
import gc

@dataclass
class SpecConfig:
    speed: bool = 0
    ppl: bool = 0
    speedNew: bool = 1
    speedMulti: bool = 0
    temperature: float = 0.85
    top_p: float = 0.9
    maxLen: int = 200
    sentenceLimit: int = 100
    sentences: int = 20
    gamma: int = 4
    maxLenNew: int = 1000

def loadData(name, split, sentences):
    print(f"正在加载数据集...")
    if name == "wikitext":
        dataSet = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
        text = [s for s in dataSet["text"] if len(s) >= SpecConfig.sentenceLimit]
        print("加载完成")
    else:
        raise(NotImplementedError)
    return text[:sentences]

def ppl_test(p_model, q_model, texts):
    print("开始计算PPL...")
    ppls = []
    for i in range(len(texts)):
        smallPPL = calculate_ppl_regr(q_model, texts[i])
        bigPPL = calculate_ppl_regr(p_model, texts[i])
        specPPL = calculate_ppl_spec(p_model, q_model, texts[i])
        ppls.append((smallPPL, bigPPL, specPPL))
    print("计算完成")
    return ppls

def speed_test_regr(model, prompt, maxLen, temperature: float = 1., 
                    top_k: int = 1, top_p: float = 1.):
    gc.disable()
    start_time = time.time()
    result = regrSampling(
        x=prompt,
        model=model,
        maxLen=1,
        temperature=temperature,
        top_p=top_p
    )
    torch.cuda.synchronize()
    ttft = time.time() - start_time
    start_time = time.time()
    result = regrSampling(
        x=prompt,
        model=model,
        maxLen=maxLen,
        temperature=temperature,
        top_p=top_p
    )
    torch.cuda.synchronize()
    total = time.time() - start_time
    tpot = (total / maxLen * 1000)
    throughput = maxLen / total
    gc.enable()
    return ttft, tpot, throughput

def speed_test_spec(p_model, q_model, prompt, maxLen, temperature: float = 1., 
                    top_k: int = 1, top_p: float = 1.):
    gc.disable()
    start_time = time.time()
    result = specSampling(
        prefix=prompt,
        q_model=q_model,
        p_model=p_model,
        maxLen=4,
        gamma=4,
        temperature=temperature,
        top_p=top_p
    )
    torch.cuda.synchronize()
    ttft = time.time() - start_time
    start_time = time.time()
    result = specSampling(
        prefix=prompt,
        q_model=q_model,
        p_model=p_model,
        maxLen=maxLen,
        gamma=4,
        temperature=temperature,
        top_p=top_p
    )
    torch.cuda.synchronize()
    total = time.time() - start_time
    tpot = (total / maxLen * 1000)
    throughput = maxLen / total
    gc.enable()
    return ttft, tpot, throughput

def speed_test(p_model, q_model, prompt, maxLen, temperature: float = 1., 
               top_k: int = 1, top_p: float = 1.):
    print("开始速度测试...")
    small_ttft, small_tpot, small_throughput = speed_test_regr(q_model, prompt, maxLen, temperature, top_k, top_p)
    big_ttft, big_tpot, big_throughput = speed_test_regr(p_model, prompt, maxLen, temperature, top_k, top_p)
    spec_ttft, spec_tpot, spec_throughput = speed_test_spec(p_model, q_model, prompt, maxLen, temperature, top_k, top_p)
    return {
        "small": (small_ttft, small_tpot, small_throughput),
        "big": (big_ttft, big_tpot, big_throughput),
        "spec": (spec_ttft, spec_tpot, spec_throughput)
    }

def speed_benchmark(p_model_path, q_model_path):
    print(f"正在加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(p_model_path)
    p_model16 = AutoModelForCausalLM.from_pretrained(
        p_model_path, device_map="auto", dtype=torch.float16
    )
    q_model16 = AutoModelForCausalLM.from_pretrained(
        q_model_path, device_map="auto", dtype=torch.float16
    )
    device = p_model16.device
    print("模型加载完成")
    print(f"模型当前运行在: {device}")
    inputs = tokenizer("Computer Science is", return_tensors="pt")
    prompt = inputs["input_ids"].to(device)
    speeds = speed_test(p_model16, q_model16, prompt, SpecConfig.maxLen, temperature=SpecConfig.temperature, top_p=SpecConfig.top_p)
    return speeds

def ppl_benchmark(p_model_path, q_model_path):
    print("正在加载模型...")
    p_model32 = AutoModelForCausalLM.from_pretrained(
        p_model_path, device_map="auto", dtype=torch.float32
    )
    q_model32 = AutoModelForCausalLM.from_pretrained(
        q_model_path, device_map="auto", dtype=torch.float32
    )
    texts = loadData("wikitext", "test", SpecConfig.sentences)
    ppls = ppl_test(p_model32, q_model32, texts)
    return ppls

def speed_test_spec_new(p_model, q_model, prompt, maxLen, temperature: float = 1., 
                        top_k: int = 1, top_p: float = 1.):
    gc.disable()
    start_time = time.time()
    result = specSampling_new(
        prefix=prompt,
        q_model=q_model,
        p_model=p_model,
        maxLen=4,
        gamma=4,
        temperature=temperature,
        top_p=top_p
    )
    torch.cuda.synchronize()
    ttft = time.time() - start_time
    start_time = time.time()
    result = specSampling_new(
        prefix=prompt,
        q_model=q_model,
        p_model=p_model,
        maxLen=maxLen,
        gamma=4,
        temperature=temperature,
        top_p=top_p
    )
    torch.cuda.synchronize()
    total = time.time() - start_time
    tpot = (total / maxLen * 1000)
    throughput = maxLen / total
    gc.enable()
    return ttft, tpot, throughput

def speed_test_new(p_model, q_model, prompt, maxLen, temperature: float = 1., 
                   top_k: int = 1, top_p: float = 1.):
    print("开始新速度测试...")
    spec_ttft_new, spec_tpot_new, spec_throughput_new = speed_test_spec_new(p_model, q_model, prompt, maxLen, temperature, top_k, top_p)
    spec_ttft, spec_tpot, spec_throughput = speed_test_spec(p_model, q_model, prompt, maxLen, temperature, top_k, top_p)
    return {
        "spec": (spec_ttft, spec_tpot, spec_throughput),
        "new": (spec_ttft_new, spec_tpot_new, spec_throughput_new)
    }

def speed_benchmark_new(p_model_path, q_model_path):
    print(f"正在加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(p_model_path)
    p_model16 = AutoModelForCausalLM.from_pretrained(
        p_model_path, device_map="auto", dtype=torch.float16
    )
    q_model16 = AutoModelForCausalLM.from_pretrained(
        q_model_path, device_map="auto", dtype=torch.float16
    )
    device = p_model16.device
    print("模型加载完成")
    print(f"模型当前运行在: {device}")
    inputs = tokenizer("He redid the main theme", return_tensors="pt")
    # inputs = tokenizer("He portrayed an emergency physician", return_tensors="pt")
    prompt = inputs["input_ids"].to(device)
    speeds = speed_test_new(p_model16, q_model16, prompt, SpecConfig.maxLenNew, temperature=SpecConfig.temperature, top_p=SpecConfig.top_p)
    return speeds

def speed_benchmark_multi(p_model_path, q_model_path, r_model_path):
    print(f"正在加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(p_model_path)
    p_model16 = AutoModelForCausalLM.from_pretrained(
        p_model_path, device_map="auto", dtype=torch.float16
    )
    q_model16 = AutoModelForCausalLM.from_pretrained(
        q_model_path, device_map="auto", dtype=torch.float16
    )
    r_model16 = AutoModelForCausalLM.from_pretrained(
        r_model_path, device_map="auto", dtype=torch.float16
    )
    device = p_model16.device
    print("模型加载完成")
    print(f"模型当前运行在: {device}")
    inputs = tokenizer("He redid the main theme", return_tensors="pt")
    # inputs = tokenizer("He portrayed an emergency physician", return_tensors="pt")
    prompt = inputs["input_ids"].to(device)
    print("开始多级速度测试...")
    gc.disable()
    maxLen = 500
    temperature = SpecConfig.temperature
    top_p = SpecConfig.top_p
    start_time = time.time()
    result = specSampling_new_multi(
        prefix=prompt,
        q_model=q_model16,
        p_model=p_model16,
        r_model=r_model16,
        maxLen=maxLen,
        gamma1=4,
        gamma2=4,
        temperature=temperature,
        top_p=top_p
    )
    torch.cuda.synchronize()
    total = time.time() - start_time
    tpot = (total / maxLen * 1000)
    throughput = maxLen / total
    gc.enable()
    return tpot, throughput

if __name__ == "__main__":
    p_model_path = "./models/pythia-2.8b"
    q_model_path = "./models/pythia-70m"
    q_model_path2 = "./models/pythia-410m"
    if SpecConfig.speed:
        print(speed_benchmark(p_model_path, q_model_path))
    if SpecConfig.ppl:
        print(ppl_benchmark(p_model_path, q_model_path))
    if SpecConfig.speedNew:
        print(speed_benchmark_new(p_model_path, q_model_path2))
    if SpecConfig.speedMulti:
        print(speed_benchmark_multi(p_model_path, q_model_path, q_model_path2))