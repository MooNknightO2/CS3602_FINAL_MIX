import torch
from tqdm import tqdm
from utils import norm_logits, sample

@torch.no_grad()
def regrSampling(x: torch.Tensor, model: torch.nn.Module, maxLen: int, 
                 temperature: float = 1, top_k: int = 0, top_p: float = 0):
    n = x.shape[1]
    T = n + maxLen
    with tqdm(total=T, desc="regrSampling") as pbar:
        while n < T:
            outputs = model(x)
            last_p = norm_logits(outputs.logits[:, -1, :], temperature, top_k, top_p)
            idx_next = sample(last_p)
            x = torch.cat((x, idx_next), dim=1)
            n += 1
            pbar.update(n - pbar.n)
    return x