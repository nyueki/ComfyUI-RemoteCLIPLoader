# 🌐 Remote CLIP for ComfyUI

**Offload heavy CLIP models to a different machine to save VRAM.**

These custom nodes allow you to run the **CLIP model** on one computer (The **Sender**) and use it wirelessly on another computer (The **Loader**).

---

## ⚙️ Installation

Run this command in your `ComfyUI/custom_nodes` folder on **both computers**:

```bash
git clone https://github.com/nyueki/ComfyUI-RemoteCLIPLoader.git

```

*Restart ComfyUI on both machines.*

---

## 🚀 How to Use

### 1️⃣ On the "Sender" Machine

*(The PC holding the CLIP model)*

1. Add the **Send Remote CLIP** node.
2. Connect your desired **CLIP model** to it.
3. Set the `listen_port` (default `8181`).
4. **Queue the workflow** to start listening.

### 2️⃣ On the "Loader" Machine

*(The PC generating images)*

1. Add the **Load Remote CLIP** node.
2. Enter the **IP address** of the Sender machine.
3. Match the `port` number (default `8181`).
4. Connect it to your workflow wherever a normal CLIP node is expected.

---

## 🎨 How to use LoRAs

Because the **Loader** machine does not have the actual CLIP model loaded (it only has a "remote connection"), you cannot use a standard LoRA Loader on that machine to patch the CLIP. You have two options:

### Option A: Apply LoRA to CLIP (On the Sender)

If you need the LoRA to affect the text encoding (CLIP), you must load it on the **Sender** machine *before* transmitting the connection.

1. On the **Sender** machine, place the **LoraLoaderCLIPOnly** node between your Checkpoint/CLIP Loader and the **Send Remote CLIP** node.
2. Select your LoRA.
3. Connect the output to the **Send Remote CLIP** node.

### Option B: Apply LoRA to Model Only (On the Loader)

If you only need the LoRA to affect the image generation (UNet/Model) and not the text prompt, do this on the **Loader** machine.

1. Use the standard **LoraLoaderModelOnly** node (built into ComfyUI).
2. Connect it to your Model/UNet on the Loader machine.

---

## 🛡️ Important Notes

* **Firewall:** Ensure the port (default 8181) is allowed through the firewall on the Sender machine.
* **Network:** Works best on a local home network (LAN).
* **Security:** Do not use this over the public internet; it is not encrypted.

**Node Category:** `Remote CLIP`