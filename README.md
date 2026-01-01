# ComfyUI Remote CLIP Loader

A small ComfyUI custom node set to offload CLIP encoding to a remote machine.

Motivation
I created this because I have a PC with 16GB VRAM and some large models must unload and reload the text encoder during generation, which causes inefficient generation times. I decided to run CLIP on my Mac as a remote encoder and keep my PC as the image generator.

What this does
- Runs a simple TCP worker that accepts prompt encode requests and returns tensors (`cond`, `pooled`).
- Worker code and client loader are in [__init__.py](__init__.py).
- Key symbols: [`RemoteCLIPWorker`](__init__.py), [`RemoteCLIPWorker.start_worker`](__init__.py), [`RemoteCLIPLoader`](__init__.py), [`RemoteCLIPLoader.load_remote`](__init__.py), [`RemoteCLIPProxy`](__init__.py), [`pack_tensors`](__init__.py), [`unpack_tensors`](__init__.py).

Tested with
- Flux.2 Dev
- Qwen Image
- Wan2.2

Requirements
- See [requirements.txt](requirements.txt). At minimum you need PyTorch and numpy.

Installation
1. Go to your ComfyUI installation's custom_nodes folder:
   cd /path/to/ComfyUI/custom_nodes
2. Clone this repository:
   git clone 
3. Restart ComfyUI.

Usage
- On the machine that will serve CLIP (your Mac), add the "Remote CLIP Worker" node (see [`RemoteCLIPWorker`](__init__.py)) and run it (set the CLIP model and listen port).
- On your generator PC, add the "Remote CLIP Loader" node (see [`RemoteCLIPLoader`](__init__.py)), point it to the worker's IP and port, and connect it as you would a normal CLIP model.
- The protocol and tensor packing are implemented by [`pack_tensors`](__init__.py) / [`unpack_tensors`](__init__.py).