# ComfyUI-ZImage-Triton

Triton-accelerated W8A8 quantization for Z-Image (Base & Turbo) S3-DiT -- ComfyUI node with Hadamard rotation, LoRA & ControlNet support

Custom Triton kernel acceleration for [Z-Image](https://huggingface.co/Tongyi-MAI/Z-Image) S3-DiT diffusion transformer in ComfyUI. Works directly with your existing BF16 model -- no extra model downloads, no custom CUDA builds. Just install and go.

> **[Benchmark Results](benchmark/BENCHMARK.md)** -- RTX 5090, Z-Image Base (30 steps) & Turbo (4 steps), with LoRA and ControlNet scenarios.

## When to Use

- **Z-Image Base users**: This is the **only kernel-level acceleration** available for Z-Image Base. [Nunchaku](https://github.com/nunchaku-ai/nunchaku) supports Turbo only ([Base was requested but closed as inactive](https://github.com/nunchaku-ai/nunchaku/issues/898)), and other options (GGUF, FP8) are weight-only -- they reduce VRAM but don't accelerate compute. This node gives Base users actual inference speedup for the first time.
- **Rapid prompt iteration (PoC)**: Faster feedback loop when tuning prompts and parameters
- **Batch image generation**: Initial Triton compilation cost (~3.6s) is amortized across many generations
- **Production serving**: Consistent speedup without manual optimization

## Key Advantages

| Feature | Details |
|---------|---------|
| **No extra model downloads** | Uses your existing BF16 safetensors as-is -- quantization happens at runtime |
| **Easy installation** | Pure Triton (Python) -- no custom CUDA builds, no version-matching wheels, just `pip install` |
| **Near-lossless quality** | W8A8 preserves quality better than W4A4 -- ideal when image fidelity matters |
| **Drop-in** | Add one node to your existing workflow -- no code changes, no new model files |
| **INT8 quantization** | W8A8 + Hadamard rotation saves ~3.5GB VRAM while preserving quality. Rotation is applied offline (absorbed into weights), so runtime overhead is virtually zero |
| **Dynamic shapes** | Resolution changes require no recompilation (unlike `torch.compile`) |
| **LoRA / ControlNet** | Works with any LoRA and ControlNet workflows |

## Comparison with Alternatives

| Method | Quantization | Quality | Speedup | Extra Downloads | Install | Z-Image Base |
|--------|-------------|---------|---------|-----------------|---------|:------------:|
| **This (Triton W8A8)** | W8A8 Hadamard | Near-lossless | 20-30% | **None** (uses existing BF16) | `pip install` | Yes |
| [Nunchaku](https://github.com/nunchaku-ai/nunchaku) (SVDQuant) | W4A4 | Good (minor loss) | [20-30%+](https://github.com/nunchaku-ai/nunchaku/blob/main/README.md) | [Quantized model](https://huggingface.co/nunchaku-ai/nunchaku-z-image-turbo) required | Custom CUDA whl | No (Turbo only) |
| GGUF ([city96](https://github.com/city96/ComfyUI-GGUF)) | Weight-only (Q2-Q8) | Varies by level | No compute speedup | GGUF model required | `pip install` | Yes |
| [TensorRT](https://developer.nvidia.com/tensorrt) | FP8/BF16 | Lossless | Not tested | Engine build | Engine build | Unknown |
| `torch.compile` | N/A | N/A | N/A | None | Built-in | Broken |

**Notes:**
- **Nunchaku** provides excellent acceleration with W4A4 quantization and achieves similar or greater speedups. However, it requires [downloading separate quantized model files](https://huggingface.co/nunchaku-ai/nunchaku-z-image-turbo) (~6GB per variant) and installing a [platform-specific CUDA wheel](https://github.com/nunchaku-ai/nunchaku/releases) matched to your PyTorch + Python version. It also only supports Z-Image Turbo -- [Z-Image Base support was requested but closed](https://github.com/nunchaku-ai/nunchaku/issues/898). Our approach works directly with your existing BF16 safetensors for both Base and Turbo -- no extra downloads, no build hassles. If maximum speed on Turbo matters more than simplicity, consider Nunchaku. See [community benchmarks](https://huggingface.co/nunchaku-ai/nunchaku-z-image-turbo/discussions/11) for comparisons.
- **GGUF** (Q4/Q5/Q8) reduces VRAM by storing weights at lower precision, but dequantizes back to BF16 during inference -- meaning no actual compute acceleration. Useful for fitting the model into limited VRAM, not for faster generation.
- **TensorRT** has not been publicly tested with Z-Image S3-DiT. Similar architectures (e.g., FLUX.1-dev with RoPE) [work with Torch-TensorRT](https://developer.nvidia.com/blog/double-pytorch-inference-speed-for-diffusion-models-using-torch-tensorrt/), but no Z-Image results are available.
- **torch.compile** has [known issues with Z-Image in ComfyUI](https://github.com/comfyanonymous/ComfyUI/issues/10965) -- output images are broken. Root cause appears to be SageAttention/Triton conflicts rather than fundamental incompatibility. Workarounds exist (e.g., [TorchCompileModelAdvanced](https://github.com/kijai/ComfyUI-KJNodes)), but reliability is inconsistent.

## Installation

### Option 1: ComfyUI Manager (Recommended)

Search for **"ZImage Triton"** in ComfyUI Manager and install directly. Also available on the [ComfyUI Registry](https://registry.comfy.org).

### Option 2: Manual

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/newgrit1004/ComfyUI-ZImage-Triton.git
pip install -r ComfyUI-ZImage-Triton/requirements.txt
```

Requires: NVIDIA GPU with CUDA support, PyTorch (provided by ComfyUI).

## Nodes

### ZImage Triton Accelerate

Replaces standard PyTorch operations with fused Triton kernels for faster inference.

| Input | Type | Description |
|-------|------|-------------|
| `model` | MODEL | Z-Image model from UNETLoader |
| `enable_int8` | BOOLEAN | Enable W8A8+Hadamard INT8 quantization (default: ON) |

**INT8 ON** (default): Best speedup + VRAM savings. Uses Hadamard rotation to preserve quality.

**INT8 OFF**: Triton kernels only, best quality, still faster than baseline.

## Workflow Examples

Pre-built workflow JSONs are in `workflows/`:

```
workflows/
├── base/          # Z-Image Base (30 steps, CFG 4.0)
│   ├── t2i.json                        # Baseline
│   ├── t2i_triton.json                 # + Triton
│   ├── t2i_triton_int8.json            # + Triton + INT8
│   ├── t2i_lora_famegrid.json          # + LoRA
│   ├── t2i_lora_famegrid_triton.json   # + LoRA + Triton
│   ├── t2i_multi_lora.json             # + Multi-LoRA
│   ├── t2i_multi_lora_triton.json      # + Multi-LoRA + Triton
│   └── ...
└── turbo/         # Z-Image Turbo (4 steps, CFG 1.0)
    ├── t2i.json
    ├── t2i_multi_lora.json             # + Multi-LoRA
    ├── t2i_controlnet_lora.json        # + ControlNet + LoRA
    └── ...
```

## First Run Note

The very first inference after loading the model will be slower due to one-time setup costs:

- **Triton kernel compilation**: ~3.6 seconds. Compiled kernels are cached automatically by Triton -- subsequent runs (even across restarts) start instantly. This is a one-time cost per GPU architecture.
- **INT8 quantization** (if enabled): ~10-20 seconds for Hadamard rotation and weight quantization. This runs each time the model is loaded.
- **Model loading**: Loading the BF16 model (~12GB), CLIP text encoder, and VAE from disk adds additional time on cold start.

After the first run, inference speed stabilizes to the benchmarked times.

## Triton Kernels

Six fused kernels optimized for the S3-DiT architecture:

| Kernel | Replaces |
|--------|----------|
| Fused RMSNorm | `torch` RMSNorm forward |
| Fused SwiGLU | FFN gating + activation |
| Fused QK-Norm + RoPE | Separate QK normalization and rotary embedding |
| Fused Norm + Gate + Residual | Post-attention norm, gate, and residual add |
| AdaLN Modulation | Adaptive layer norm shift/scale |
| RoPE 3D | 3D rotary position embedding |

## INT8 Quantization: W8A8 + Hadamard Rotation

- **W8A8**: Both weights and activations quantized to INT8, using INT8 Tensor Cores
- **Hadamard rotation**: Offline rotation spreads outlier values evenly, preserving model quality ([QuaRot](https://arxiv.org/abs/2404.00456), NeurIPS 2024). For DiT models, group-wise rotation is applied to handle row-wise outliers ([ConvRot](https://arxiv.org/abs/2512.03673))
- **Sensitive layer skip**: Embedding, AdaLN, and first/last transformer blocks excluded automatically
- **VRAM savings**: INT8 weights replace BF16 weights, reducing total VRAM by ~3.5GB (~4.8GB on transformer weights alone)

## INT8 + LoRA Behavior

When INT8 is enabled, LoRA effects are **partially applied**: quantized layers (~80% of transformer) use pre-LoRA INT8 weights, while sensitive layers (embedders, AdaLN, first/last blocks, ~20%) receive full LoRA application. This means LoRA styling is present but slightly weaker compared to non-INT8 mode. For full LoRA fidelity, disable INT8 (Triton-only mode still provides ~1.08x speedup).

This zero-overhead approach matches how other quantization frameworks handle LoRA compatibility (see [Nunchaku](https://github.com/nunchaku-ai/nunchaku), [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF)) and provides consistent speedup regardless of LoRA count. **Multiple LoRAs are fully supported** -- stack as many as needed.

## Compatibility

| Component | Supported |
|-----------|-----------|
| Z-Image Base | Yes |
| Z-Image Turbo | Yes |
| LoRA | Yes |
| Multi-LoRA | Yes |
| ControlNet | Yes |
| Dynamic resolution | Yes (no recompilation) |
| GPU | NVIDIA with INT8 Tensor Cores (RTX 2060+) |

## References

- [QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs](https://arxiv.org/abs/2404.00456) (NeurIPS 2024) -- Hadamard rotation for quantization
- [ConvRot: Rotation-Based Plug-and-Play 4-bit Quantization for Diffusion Transformers](https://arxiv.org/abs/2512.03673) -- Group-wise Hadamard rotation for DiT
- [Z-Image: An Efficient Image Generation Foundation Model](https://arxiv.org/abs/2511.22699) -- S3-DiT architecture
- [Nunchaku / SVDQuant](https://github.com/nunchaku-ai/nunchaku) -- Alternative W4A4 acceleration for Z-Image

## License

[MIT](LICENSE)
