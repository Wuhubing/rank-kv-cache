"""Long-context Quest trace capture used for the pilot (run in Colab on a GPU).

Same selector as notebooks/first_trace.ipynb. Rankings and all-page scores are written into
preallocated GPU buffers and returned as numpy arrays (int16 [steps, L, K], float16 [steps, L, P]).
Prompt: Qwen3 chat template, thinking enabled, "summarise the whole document" (see
docs/long_trace_pilot.md). Stop: EOS, max_new, or the newest 32-gram repeating.
"""
import torch
from torch.nn import functional as F
from transformers import AttentionInterface

USER = {
    "qmsum": "Meeting transcript:\n\n{context}\n\nWrite a detailed summary of the whole meeting, "
             "covering every topic that was discussed, in at least eight paragraphs.",
    "gov_report": "Government report:\n\n{context}\n\nWrite a detailed summary of the whole report, "
                  "covering every section, in at least eight paragraphs.",
}

# Robustness check D only (docs/efficient_policy_dev.md): a differently worded summary prompt.
ALT_USER = {
    "qmsum": "Here is a meeting transcript.\n\n{context}\n\nSummarize this meeting for someone who missed it. "
             "Organize the summary by topic, with a short heading for each topic, and explain the decisions "
             "and open questions.",
    "gov_report": "Here is a government report.\n\n{context}\n\nSummarize this report for a policy maker. "
                  "Organize the summary by section, with a short heading for each section, and explain the "
                  "findings and recommendations.",
}


def chat_prompt(tok, task, row):
    msg = [{"role": "user", "content": USER[task].format(context=row["context"])}]
    return tok.apply_chat_template(msg, tokenize=True, add_generation_prompt=True, enable_thinking=True)


class QuestPages:
    def __init__(s, page_size, top_k):
        s.page_size, s.top_k, s.bounds, s.step = page_size, top_k, {}, None

    def __call__(s, module, query, key, value, attention_mask, scaling, **kw):
        layer = module.layer_idx
        kvh, length, dim = key.shape[1:]
        hp = (length - 1) // s.page_size
        past = s.bounds.get(layer)
        cached = 0 if past is None else past[0].shape[1]
        if hp > cached:
            b = key[0, :, cached*s.page_size:hp*s.page_size].reshape(kvh, hp-cached, s.page_size, dim)
            lo, hi = b.amin(2).float(), b.amax(2).float()
            if past is not None:
                lo, hi = torch.cat((past[0], lo), 1), torch.cat((past[1], hi), 1)
            s.bounds[layer] = lo, hi
        g = query.shape[1] // kvh
        lo, hi = s.bounds[layer]
        q = query[0, :, 0].reshape(kvh, g, dim).float()
        scores = (torch.einsum("hgd,hpd->hgp", q.clamp_min(0), hi)
                  + torch.einsum("hgd,hpd->hgp", q.clamp_max(0), lo)).mean((0, 1))
        ranked = scores.argsort(descending=True, stable=True)[:s.top_k]
        s.RANK[s.step, layer] = ranked.to(torch.int16)
        s.SCORE[s.step, layer, :hp] = scores.to(torch.float16)
        off = torch.arange(s.page_size, device=key.device)
        pos = torch.cat(((ranked.sort().values[:, None]*s.page_size + off).flatten(),
                         torch.arange(hp*s.page_size, length, device=key.device)))
        k = key.index_select(2, pos).repeat_interleave(g, 1)
        v = value.index_select(2, pos).repeat_interleave(g, 1)
        out = F.scaled_dot_product_attention(query, k, v, scale=scaling)
        return out.transpose(1, 2).contiguous(), None


def looping(gen, n=32):
    if len(gen) < 2*n:
        return False
    last = tuple(gen[-n:])
    return any(tuple(gen[i:i+n]) == last for i in range(len(gen)-n))


@torch.inference_mode()
def capture(model, prompt, page_size=16, top_k=64, max_new=1024, eos_ids=()):
    cfg = model.config
    L = cfg.num_hidden_layers
    be = QuestPages(page_size, top_k)
    pmax = (len(prompt) + max_new) // page_size
    be.RANK = torch.full((max_new, L, top_k), -1, dtype=torch.int16, device="cuda")
    be.SCORE = torch.full((max_new, L, pmax), float("nan"), dtype=torch.float16, device="cuda")
    AttentionInterface.register("quest_npz", be)
    ids = torch.tensor([prompt], device="cuda")
    orig = cfg._attn_implementation
    try:
        cfg._attn_implementation = "sdpa"
        out = model(input_ids=ids, use_cache=True, logits_to_keep=1)
        cache = out.past_key_values
        tokn = out.logits[:, -1].argmax(-1, keepdim=True)
        gen = [tokn.item()]
        cfg._attn_implementation = "quest_npz"
        stop = "length"
        for step in range(max_new - 1):
            if gen[-1] in eos_ids:
                stop = "eos"
                break
            if looping(gen):
                stop = "loop"
                break
            be.step = step
            out = model(input_ids=tokn, past_key_values=cache, use_cache=True, logits_to_keep=1)
            cache = out.past_key_values
            tokn = out.logits[:, -1].argmax(-1, keepdim=True)
            gen.append(tokn.item())
        n = len(gen) - 1
        npages = (len(prompt) + n - 1) // page_size
        return be.RANK[:n].cpu().numpy(), be.SCORE[:n, :, :npages].cpu().numpy(), gen, stop
    finally:
        cfg._attn_implementation = orig
        be.bounds.clear()
        del be.RANK, be.SCORE
