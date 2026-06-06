# Remote CLIP for ComfyUI

Run the CLIP text encoder on one machine and use it from another. This is useful for offloading CLIP to a separate GPU or host to free up VRAM on the machine doing image generation.

As well as text encoding (conditioning), the remote CLIP also supports text generation, so LLM-style text encoders (such as Gemma) can drive nodes like **Generate Text** and **Generate LTX2 Prompt** from across the network. Image, video, and audio inputs are forwarded to the Sender for both encoding and generation.

There are two roles:

- **Sender** — the machine that holds the CLIP model and runs the encoder.
- **Loader** — the machine running the workflow, which connects to the Sender in place of a local CLIP.

## Installation

Clone the repository into your `ComfyUI/custom_nodes` folder on both machines:

```bash
git clone https://github.com/nyueki/ComfyUI-RemoteCLIPLoader.git
```

Restart ComfyUI on both machines.

## Usage

### Sender

On the machine holding the CLIP model:

1. Add the **Send Remote CLIP** node.
2. Connect your CLIP model to it.
3. Set `listen_port` (default `8181`).
4. Optionally set an `auth_token` to require a matching secret from Loaders, and a `bind_host` to restrict the listening interface (default `0.0.0.0`, all interfaces).
5. Queue the workflow to start listening.

### Loader

On the machine running the workflow:

1. Add the **Load Remote CLIP** node.
2. Set `worker_ip` to the Sender's IP address.
3. Set `port` to match the Sender (default `8181`).
4. Connect the output anywhere a normal CLIP is expected.

Optional inputs:

- `auth_token` — A shared secret. If the Sender was started with a token, the same value must be set here or requests are rejected. Leave blank when the Sender has no token. Can also be supplied through the `REMOTE_CLIP_TOKEN` environment variable.
- `transport_precision` — Precision of embeddings sent over the network:
  - `auto` (default) — sends fp16 to remote hosts to save bandwidth, but keeps full precision when the worker is on localhost, where bandwidth is not a constraint.
  - `fp16` — always send half precision. Halves bandwidth at the cost of a small precision loss; useful on slower links.
  - `fp32` — never downcast; full precision regardless of host.

The Sender and Loader must run the same plugin version. A protocol mismatch is rejected with a clear error rather than producing incorrect output.

## Text generation

The remote CLIP works as a drop-in for generation-capable text encoders, not just plain conditioning. Connect **Load Remote CLIP** to a generation node such as **Generate Text** or **Generate LTX2 Prompt** and the prompt, sampling settings, and any image/video/audio inputs are sent to the Sender, which runs the generation and returns the text.

- The model held by the Sender must actually support generation (for example a Gemma-class encoder). If it only does encoding, the generation nodes will error.
- Generation is not cached, since sampling and seeds make each run distinct.
- Generation can take much longer than encoding (autoregressive decoding runs token by token), so its network read uses an extended timeout and is not retried, to avoid kicking off a second expensive run on a brief hiccup. Generation speed is bound by the Sender's model and hardware, not by this plugin.

## LoRAs

The text encoder runs on the Sender, so any LoRA affecting CLIP is ultimately applied there. There are two ways to set this up.

### Apply on the Sender

Place the **LoraLoaderCLIPOnly** node between the Checkpoint or CLIP Loader and the **Send Remote CLIP** node on the Sender. Select the LoRA, set `strength_clip`, and connect the output to **Send Remote CLIP**.

### Apply from the Loader

You can also place **LoraLoaderCLIPOnly** on the Loader, right after **Load Remote CLIP**. When it detects a remote connection it does not patch locally; instead it forwards the LoRA name and strength to the Sender, which applies them.

- The LoRA file must exist in the Sender's `loras` folder. The Loader only sends the name and strength, so a missing file fails with a "LoRA not found on worker" error.
- Multiple `LoraLoaderCLIPOnly` nodes can be stacked; they accumulate and are all applied on the Sender.
- The standard ComfyUI LoRA Loader does not work against a remote CLIP. Use **LoraLoaderCLIPOnly**.

### Model-only LoRAs

If a LoRA only needs to affect image generation (the UNet) and not the text prompt, use the standard **LoraLoaderModelOnly** node on the Loader as usual.

## Performance

The Sender caches encoding results to keep repeated requests fast:

- Embedding cache — an identical prompt and settings return without re-encoding. Requests carrying image/video/audio inputs are not cached.
- Patched-CLIP cache — a given LoRA stack is applied once and reused, avoiding repeated patching.

The Loader reconnects automatically if the connection drops, so brief network interruptions do not break a running encode. Generation is the exception: it is not auto-retried, so a dropped connection mid-generation surfaces as an error rather than silently running twice.

## Notes

- Authentication: when exposing the Sender on a shared network, set the same `auth_token` on both nodes. If the Sender binds to `0.0.0.0` without a token, anyone on the network can use the model, and the Sender logs a warning.
- Bind host: the Sender binds to `0.0.0.0` by default. Set `bind_host` to a specific address to restrict access.
- Firewall: allow the listening port (default 8181) through the firewall on the Sender.
- Network: intended for use on a local network.
- Encryption: traffic is not encrypted. Do not expose this directly over the public internet; tunnel it through a VPN or SSH instead.

Node category: `Remote CLIP`

## Contributing

Contributions are welcome, whether bug fixes, improvements, or new features.

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/new-feature`
3. Commit your changes: `git commit -m 'Add a new feature'`
4. Push the branch: `git push origin feature/new-feature`
5. Open a pull request against the original repository.

If you find a bug or have a suggestion but aren't ready to implement it, please open an issue.
