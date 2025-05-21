from pathlib import Path
import torch
import time
import sys

from einops import rearrange
from tqdm import tqdm

from vam.video_pretraining.mup_gpt2 import load_pretrained_gpt
from vam.utils import expand_path, nvtx

device  = "cuda"
dtype   = torch.bfloat16 # or torch.float16
BS      = 1  # batch
CTX_T   = 2  # context frames
PRED_T  = 1  # frames to generate
TOPK    = 3
TEMP    = 0.95
NRUNS   = 10 if len(sys.argv) <= 1 else int(sys.argv[1])  # timed runs
WARMUP  = 3 if len(sys.argv) <= 2 else int(sys.argv[2])   # compiled graph warm-up
COMPILE = True if len(sys.argv) <= 2 else sys.argv[3] in ("true", "yes", "t", "y", "T", "Y")

MUP_GPT2_COLOR = nvtx.get_domain_color("benchmark")

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32       = True
torch.backends.cuda.sdp_kernel        = True
torch.set_default_dtype(dtype)

# -----------------------------------------------------------
# 1.  Load model
nvtx.push_range("setup", color=MUP_GPT2_COLOR, domain="benchmark")
ckpt = expand_path("~/scratch/vavim1/width_768_pretrained_139k_total_155k.pt")
gpt  = load_pretrained_gpt(ckpt, device=device).to(dtype)
gpt.compile_forward = COMPILE
nvtx.pop_range(domain="benchmark")

nvtx.push_range("burnin", color=MUP_GPT2_COLOR, domain="benchmark")

# 2.  Dummy burn-in tokens ----------------------------------
HEIGHT, WIDTH = 18, 32
VOCAB_SIZE    = gpt.transformer.wie.weight.size(0)
burnin_tokens = torch.randint(
    0, VOCAB_SIZE,
    (BS, CTX_T, HEIGHT, WIDTH),
    dtype=torch.long, device=device)

# -----------------------------------------------------------
# 3.  Warm-up (fills the kernel cache, avoids first-run noise)
for _ in tqdm(range(WARMUP)):
    _ = gpt.forward_inference(PRED_T, burnin_tokens,
                  temperature=TEMP, topk_sampler=TOPK,
                  use_kv_cache=True, verbose=0)
torch.cuda.synchronize()

nvtx.pop_range(domain="benchmark")

# 4.  Timed runs -------------------------------------------
nvtx.push_range("bench", color=MUP_GPT2_COLOR, domain="benchmark")
times = []
for _ in tqdm(range(NRUNS)):
    start = time.perf_counter()          # wall clock
    _ = gpt.forward_inference(PRED_T, burnin_tokens,
                  temperature=TEMP, topk_sampler=TOPK,
                  use_kv_cache=True, verbose=0)
    torch.cuda.synchronize()             # wait for GPU
    times.append(time.perf_counter() - start)
nvtx.pop_range(domain="benchmark")

#TODO: PREFILL TIME IS ACTUALLY TAKEN IN ACCOUNT HERE
latency  = sum(times) / NRUNS
ntokens  = PRED_T * gpt.nb_tokens_per_timestep
throughput = ntokens / latency          # tokens / second

print(f"Walltime per call : {latency*1e3:.2f} ms")
print(f"Throughput        : {throughput:.1f} tokens/s "
      f"({ntokens} tokens per pass)")
