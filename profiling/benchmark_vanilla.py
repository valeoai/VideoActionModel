import argparse
from pathlib import Path
import torch
import time
import sys

from einops import rearrange
from tqdm import tqdm

from vam.video_pretraining.mup_gpt2 import load_pretrained_gpt
from vam.utils import expand_path, nvtx

parser = argparse.ArgumentParser(description="Model run configuration")
parser.add_argument('--device', type=str, default='cuda', help='Device to run on, e.g. "cuda" or "cpu"')
parser.add_argument('--dtype', type=str, choices=['bfloat16', 'float16'], default='bfloat16',
                    help='Data type: "bfloat16" or "float16"')
parser.add_argument('--bs', type=int, default=32, help='Batch size')
parser.add_argument('--ctx_t', type=int, default=1, help='Number of context frames')
parser.add_argument('--pred_t', type=int, default=1, help='Number of frames to generate')
parser.add_argument('--topk', type=int, default=3, help='Top-K sampling')
parser.add_argument('--temp', type=float, default=0.95, help='Temperature for sampling')
parser.add_argument('--nruns', type=int, default=10, help='Number of timed runs')
parser.add_argument('--warmup', type=int, default=3, help='Number of warm-up runs')
parser.add_argument('--compile', action="store_true", help='Whether to compile the model')

args = parser.parse_args()

# Convert dtype string to torch dtype
args.dtype = torch.bfloat16 if args.dtype == 'bfloat16' else torch.float16

device  = args.device
dtype   = args.dtype
BS      = args.bs
CTX_T   = args.ctx_t
PRED_T  = args.pred_t
TOPK    = args.topk
TEMP    = args.temp
NRUNS   = args.nruns
WARMUP  = args.warmup
COMPILE = args.compile

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
ntokens  = BS * PRED_T * gpt.nb_tokens_per_timestep
throughput = ntokens / latency          # tokens / second

print(f"Walltime per call : {latency*1e3:.2f} ms")
print(f"Throughput        : {throughput:.1f} tokens/s "
      f"({ntokens} tokens per pass)")
