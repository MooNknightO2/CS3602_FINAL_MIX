import torch
from tqdm import tqdm
from utils import norm_logits, sample, max_fn

@torch.no_grad()
def specSampling(prefix: torch.Tensor, q_model: torch.nn.Module,
                 p_model: torch.nn.Module, maxLen: int, gamma: int,
                 temperature: float = 1, top_k: int = 0, top_p: float = 0.9) -> torch.Tensor:
    """
    Args:
        prefix: input sequence, (batch = 1, prefix_seqLen)
        q_model: approx model, the small one
        p_model: target model, the large one
        max_len: the max overall generated tokens number.
        gamma: the token number small model guesses.
        temperature (optional): Defaults to 1.
        top_k (optional): Defaults to 0.
        top_p (optional): Defaults to 0.
    Returns:
        torch.Tensor: generated tokens (batch, target_seqLen)
    """

    seqLen = prefix.shape[1]
    T = seqLen + maxLen
    assert prefix.shape[0] == 1
    with tqdm(total=T, desc="specSampling") as pbar:
        while prefix.shape[1] < T:
            x = prefix
            preLen = prefix.shape[1]
            for _ in range(gamma):
                q = q_model(x).logits
                next_tok = sample(norm_logits(q[:, -1, :], temperature, top_k, top_p))
                x = torch.cat((x, next_tok), dim=1)
            for i in range(q.shape[1]):
                q[:, i, :] = norm_logits(q[:, i, :], temperature, top_k, top_p)
            p = p_model(x).logits
            for i in range(p.shape[1]):
                p[:, i, :] = norm_logits(p[:, i, :], temperature, top_k, top_p)
            flag = True
            n = preLen - 1
            for i in range(gamma):
                r = torch.rand(1, device=p.device)
                j = x[:, preLen + i]
                if r < torch.min(torch.tensor([1], device=q.device), 
                                 p[:, preLen + i - 1, j] / q[:, preLen + i - 1, j]):
                    n += 1
                else:
                    t = sample(max_fn(p[:, n, :] - q[:, n, :]))
                    flag = False
                    break
            prefix = x[:, :(n + 1)]
            if flag:
                t = sample(p[:, -1, :])
            prefix = torch.cat((prefix, t), dim = 1)
            pbar.update(n - pbar.n)
    return prefix

@torch.no_grad()
def specSampling_new(prefix: torch.Tensor, q_model: torch.nn.Module,
                     p_model: torch.nn.Module, maxLen: int, gamma: int,
                     temperature: float = 1, top_k: int = 0, top_p: float = 0.9) -> torch.Tensor:
    """
    Args:
        prefix: input sequence, (batch = 1, prefix_seqLen)
        q_model: approx model, the small one
        p_model: target model, the large one
        max_len: the max overall generated tokens number.
        gamma: the token number small model guesses.
        temperature (optional): Defaults to 1.
        top_k (optional): Defaults to 0.
        top_p (optional): Defaults to 0.
    Returns:
        torch.Tensor: generated tokens (batch, target_seqLen)
    """

    seqLen = prefix.shape[1]
    T = seqLen + maxLen
    assert prefix.shape[0] == 1
    with tqdm(total=T, desc="specSampling") as pbar:
        while prefix.shape[1] < T:
            x = prefix
            preLen = prefix.shape[1]
            for _ in range(gamma):
                q = q_model(x).logits
                next_tok = sample(norm_logits(q[:, -1, :], temperature, top_k, top_p))
                x = torch.cat((x, next_tok), dim=1)
            for i in range(q.shape[1]):
                q[:, i, :] = norm_logits(q[:, i, :], temperature, top_k, top_p)
            p = p_model(x).logits
            for i in range(p.shape[1]):
                p[:, i, :] = norm_logits(p[:, i, :], temperature, top_k, top_p)
            flag = True
            n = preLen - 1
            for i in range(gamma):
                r = torch.rand(1, device=p.device)
                j = x[:, preLen + i]
                if r < torch.min(torch.tensor([1], device=q.device), 
                                 p[:, preLen + i - 1, j] / q[:, preLen + i - 1, j]):
                    n += 1
                else:
                    t = sample(max_fn(p[:, n, :] - q[:, n, :]))
                    flag = False
                    break
            prefix = x[:, :(n + 1)]
            if flag:
                gamma += 1
                t = sample(p[:, -1, :])
            else:
                gamma = max(4, gamma // 2)
            prefix = torch.cat((prefix, t), dim = 1)
            pbar.update(n - pbar.n)
    return prefix

@torch.no_grad()
def specSampling_new_multi(prefix: torch.Tensor, q_model: torch.nn.Module,
                           r_model: torch.nn.Module, p_model: torch.nn.Module, 
                           maxLen: int, gamma1: int, gamma2: int,
                           temperature: float = 1, top_k: int = 0, top_p: float = 0.9) -> torch.Tensor:
    """
    Args:
        prefix: input sequence, (batch = 1, prefix_seqLen)
        q_model: small draft model (最底层建议模型)
        r_model: intermediate model (中间审核模型)
        p_model: target model (最终大模型)
        maxLen: the max overall generated tokens number.
        gamma1: the token number q_model guesses in one round.
        gamma2: the number of rounds r_model verifies q_model before p_model verifies.
        temperature (optional): Defaults to 1.
        top_k (optional): Defaults to 0.
        top_p (optional): Defaults to 0.
    Returns:
        torch.Tensor: generated tokens (batch, target_seqLen)
    """

    seqLen = prefix.shape[1]
    T = seqLen + maxLen
    assert prefix.shape[0] == 1
    
    with tqdm(total=T, desc="specSampling_multi") as pbar:
        while prefix.shape[1] < T:
            x = prefix
            original_len = prefix.shape[1]
            for _ in range(gamma2):
                curr_valid_len = x.shape[1]
                for _ in range(gamma1):
                    q = q_model(x).logits
                    next_tok = sample(norm_logits(q[:, -1, :], temperature, top_k, top_p))
                    x = torch.cat((x, next_tok), dim=1)
                q_logits = q_model(x).logits
                for i in range(q_logits.shape[1]):
                    q_logits[:, i, :] = norm_logits(q_logits[:, i, :], temperature, top_k, top_p)
                r_logits = r_model(x).logits
                for i in range(r_logits.shape[1]):
                    r_logits[:, i, :] = norm_logits(r_logits[:, i, :], temperature, top_k, top_p)
                flag_r = True
                n_r = curr_valid_len - 1 
                for i in range(gamma1):
                    r_rand = torch.rand(1, device=r_model.device)
                    j = x[:, curr_valid_len + i]
                    if r_rand < torch.min(torch.tensor([1], device=q_model.device), 
                                          r_logits[:, curr_valid_len + i - 1, j] / q_logits[:, curr_valid_len + i - 1, j]):
                        n_r += 1
                    else:
                        t = sample(max_fn(r_logits[:, n_r, :] - q_logits[:, n_r, :]))
                        flag_r = False
                        break
                x = x[:, :(n_r + 1)]
                if flag_r:
                    t = sample(r_logits[:, -1, :])
                
                x = torch.cat((x, t), dim=1)
                if x.shape[1] >= T:
                    break
            r_logits_final = r_model(x).logits
            for i in range(r_logits_final.shape[1]):
                r_logits_final[:, i, :] = norm_logits(r_logits_final[:, i, :], temperature, top_k, top_p)
            p_logits = p_model(x).logits
            for i in range(p_logits.shape[1]):
                p_logits[:, i, :] = norm_logits(p_logits[:, i, :], temperature, top_k, top_p)
            flag_p = True
            n_p = original_len - 1
            draft_len = x.shape[1] - original_len
            for i in range(draft_len):
                r_rand = torch.rand(1, device=p_model.device)
                j = x[:, original_len + i]
                if r_rand < torch.min(torch.tensor([1], device=p_model.device), 
                                      p_logits[:, original_len + i - 1, j] / r_logits_final[:, original_len + i - 1, j]):
                    n_p += 1
                else:
                    t = sample(max_fn(p_logits[:, n_p, :] - r_logits_final[:, n_p, :]))
                    flag_p = False
                    break
            prefix = x[:, :(n_p + 1)]
            if flag_p:
                t = sample(p_logits[:, -1, :])
            prefix = torch.cat((prefix, t), dim=1)
            pbar.update(prefix.shape[1] - original_len)

    return prefix