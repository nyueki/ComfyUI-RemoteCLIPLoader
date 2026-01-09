# 🌐 Remote CLIP for ComfyUI

Run **CLIP text encoding on a different machine** and use it seamlessly inside **ComfyUI** 🚀
Perfect for offloading heavy CLIP models to another PC or server.

---

## ✨ What is this?

This project adds **two custom ComfyUI nodes** that let you:

* 🖥️ Run a **CLIP Worker** on one machine
* 🎛️ Connect to it from another machine using a **CLIP Loader**
* 🔗 Send text prompts over the network
* 📦 Receive CLIP embeddings (`cond` + `pooled`) back in real time

Both machines run **ComfyUI**, but only **one needs the CLIP model loaded**.

---

## 🧠 How it works (simple view)

```
[ ComfyUI (Client) ]  --->  text prompt  --->  [ ComfyUI (Worker) ]
[ Remote CLIP Loader ] <--- embeddings  <---  [ Remote CLIP Worker ]
```

* Uses **TCP sockets**
* Sends metadata as **JSON**
* Sends tensors as **binary blobs**
* Automatically handles tensor shapes & dtypes

---

## 🧩 Nodes Included

### 🔹 Remote CLIP Worker

🖥️ **Runs on the machine that has the CLIP model**

* Listens on a TCP port
* Receives text prompts
* Runs CLIP encoding
* Sends embeddings back to clients

**Inputs**

* `clip` → Any CLIP model
* `listen_port` → Port to listen on (default: `8002`)

---

### 🔹 Remote CLIP Loader

🎛️ **Runs on the client machine**

* Connects to the worker
* Acts like a normal CLIP object in ComfyUI
* Transparently forwards encoding requests

**Inputs**

* `worker_ip` → IP address of the worker machine
* `port` → Worker port (default: `8002`)

---

## ⚙️ Setup Instructions

### 1️⃣ Install on both machines

Clone this repository into your ComfyUI `custom_nodes` directory:

```
cd ComfyUI/custom_nodes
git clone https://github.com/nyueki/ComfyUI-RemoteCLIPLoader.git
```

Restart ComfyUI on **both machines**.

---

### 2️⃣ Start the Worker (Machine A)

1. Add **Remote CLIP Worker** node
2. Connect a CLIP model
3. Choose a port (e.g. `8002`)
4. Queue the workflow

✅ The worker is now listening for connections

---

### 3️⃣ Connect from Client (Machine B)

1. Add **Remote CLIP Loader** node
2. Enter the worker’s IP address
3. Match the port number
4. Use it anywhere a CLIP node is expected 🎉

---

## 📦 What gets transmitted?

* 📝 Prompt text
* 🔢 Tokenization options
* 🧠 CLIP embeddings:

  * `cond` → `(1, 77, 768)`
  * `pooled` → `(1, 768)`

All tensors are:

* Converted to CPU
* Sent as raw bytes
* Reconstructed safely on the client

---

## 🛡️ Notes & Tips

* 🔌 Make sure the port is open on the worker machine
* 🏠 Best used on local networks (LAN)
* ⚠️ No encryption — don’t expose to the public internet
* 🧪 Designed for simplicity & reliability

---

## 🧰 Category in ComfyUI

```
Remote CLIP
```

---

## ❤️ Why use this?

* Save VRAM on your main machine
* Share one powerful CLIP server
* Experiment with distributed ComfyUI setups
* Clean, minimal, no external dependencies
