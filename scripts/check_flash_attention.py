
import torch
import torch.nn.functional as F
import time
import math
from collections import defaultdict
from tqdm import tqdm
from einops import rearrange
import numpy as np

def attention_pytorch(qkv, dropout_p=0.0, causal=True):
    """
    Arguments:
        qkv: (batch_size, seqlen, 3, nheads, head_dim)
        dropout_p: float
    Output:
        output: (batch_size, seqlen, nheads, head_dim)
    """
    batch_size, seqlen, _, nheads, d = qkv.shape
    q, k, v = qkv.unbind(dim=2)
    q = rearrange(q, 'b t h d -> (b h) t d')
    k = rearrange(k, 'b s h d -> (b h) d s')
    softmax_scale = 1.0 / math.sqrt(d)

    scores = torch.empty(batch_size * nheads, seqlen, seqlen, dtype=qkv.dtype, device=qkv.device)
    scores = rearrange(torch.baddbmm(scores, q, k, beta=0, alpha=softmax_scale),
                       '(b h) t s -> b h t s', h=nheads)
    if causal:
        causal_mask = torch.triu(torch.full((seqlen, seqlen), float('-inf'), device=scores.device), 1)
        scores = scores + causal_mask.to(dtype=scores.dtype)
    attention = torch.softmax(scores, dim=-1)
    attention_drop = F.dropout(attention, dropout_p)
    output = torch.einsum('bhts,bshd->bthd', attention_drop, v)
    return output.to(dtype=qkv.dtype)



def flash_attention(qkv, dropout_p=0.0, causal=True):
    """
    A wrapper function that utilizes PyTorch's scaled_dot_product_attention internally.
    
    Arguments:
        qkv: Tensor of shape (batch_size, seq_length, 3, num_heads, head_dim)
        dropout_p: Dropout probability
        is_causal: Whether to apply causal masking
    
    Returns:
        output: Tensor after applying attention
    """
    batch_size, seq_length, _, num_heads, head_dim = qkv.shape
    q, k, v = qkv.unbind(dim=2)  # Split the combined qkv tensor

    # Flatten the batch and head dimensions for compatibility with scaled_dot_product_attention
    q = rearrange(q, 'b t h d -> b h t d')
    k = rearrange(k, 'b s h d -> b h s d')
    v = rearrange(v, 'b s h d -> b h s d')

    # Compute scaled dot product attention with causal masking if needed
    attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=causal)

    # Apply dropout to the attention output
    attn_output = F.dropout(attn_output, p=dropout_p)

    attn_output = rearrange(attn_output, 'b h s d -> b s h d')

    return attn_output


def run_attention_test(qkv, attention_func, dropout_p=0.0, causal=True):
    """
    Run attention test using specified attention function and measure performance.
    """
    qkv = qkv.to(torch.bfloat16)  # Ensure BF16 for the test

    torch.cuda.synchronize()  # Synchronize CUDA to ensure accurate timing
    start_time = time.time()

    # Perform the attention operation 100 times
    with torch.no_grad():
        for _ in range(100):
            output = attention_func(qkv, dropout_p=dropout_p, causal=causal)

    torch.cuda.synchronize()  # Synchronize again after operations
    duration = time.time() - start_time

    # Measure memory usage
    peak_memory = torch.cuda.max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()

    return output, duration, peak_memory

def prepare_data(batch_size, seq_length, nheads, head_dim, device):
    """
    Generate a random tensor for qkv.
    """
    qkv = torch.randn(batch_size, seq_length, 3, nheads, head_dim, device=device, dtype=torch.bfloat16)
    return qkv


if __name__ == "__main__":
    # Configuration
    batch_size = 4
    seq_length = 5600
    head_dim = 64  # Must be a multiple of 8 for potential FlashAttention
    nheads = 12 # total dim = 64*12 = 768
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    flash = defaultdict(list)
    non_flash = defaultdict(list) # non flash mem will take ~3.8GB of GPU mem

    seq_lengths = [2**i for i in range(8,13)]
    print(seq_lengths)

    for seq_length in tqdm(seq_lengths):

        qkv = prepare_data(batch_size, seq_length, nheads, head_dim, device)
        
        output1, time1, memory1 = run_attention_test(qkv, flash_attention)
        
        output2, time2, memory2 = run_attention_test(qkv, attention_pytorch)

        print((output1 - output2).abs().mean())

        flash['mem'].append(memory1)
        flash['time'].append(time1)
        
        non_flash['mem'].append(memory2)
        non_flash['time'].append(time2)

    print('mem')
    print('flash \t\t', np.array(flash['mem']) / 1024**3)
    print('non_flash \t', np.array(non_flash['mem']) / 1024**3)

    print('time')
    print('flash \t\t', np.array(flash['time']))
    print('non_flash \t', np.array(non_flash['time'])