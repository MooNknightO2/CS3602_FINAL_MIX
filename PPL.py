from torch.nn import CrossEntropyLoss
import torch
from transformers import AutoTokenizer
from torch.nn import functional as F

def calculate_ppl_regr(model, text):
    ppl = None
    p_model_path = "./models/pythia-2.8b"
    tokenizer = AutoTokenizer.from_pretrained(p_model_path)
    inputs = tokenizer(text, return_tensors='pt')
    input_ids = inputs['input_ids']
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids.to(model.device))
        logits = outputs.logits
        labels = input_ids.to(logits.device)
        shift_logits = logits[..., :-1, :]
        shift_labels = labels[..., 1:]
        batchSize, sLen, vocabSize = shift_logits.shape
        logits = shift_logits.reshape(batchSize * sLen, vocabSize)
        labels = shift_labels.reshape(batchSize * sLen)
        loss = CrossEntropyLoss()(logits, labels)
        ppl = torch.exp(loss)
    return ppl.item()

def calculate_ppl_spec(p_model, q_model, text):
    ppl = None
    p_model_path = "./models/pythia-2.8b"
    tokenizer = AutoTokenizer.from_pretrained(p_model_path)
    inputs = tokenizer(text, return_tensors='pt')
    input_ids = inputs['input_ids'].to(p_model.device)
    q_model.eval()
    p_model.eval()
    device = p_model.device
    labels = input_ids.to(device)
    probs = []
    with torch.no_grad():
        l = input_ids.shape[1]
        for i in range(1, l):
            prefix = input_ids[:, :i]
            q = q_model(prefix).logits
            q[:, -1, :] = F.softmax(q[:, -1, :], dim = 1)
            p = p_model(prefix).logits
            p[:, -1, :] = F.softmax(p[:, -1, :], dim = 1)
            label = labels[:, i]
            prob1 = min(q[:, i - 1, label], p[:, i - 1, label])
            prob2 = p[:, i - 1, label] - min(q[:, i - 1, label], p[:, i - 1, label])
            prob =  prob1 + prob2
            probs.append(prob)
        nll = -torch.log(torch.stack(probs)).mean()
        ppl = torch.exp(nll)
    return ppl.item()