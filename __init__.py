import socket
import struct
import threading
import json
import torch
import time
import folder_paths
import comfy.sd
import comfy.utils

class TensorEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.cpu().tolist()
        return super().default(obj)
    
class TensorEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, torch.Tensor):
            return {"__tensor__": True, "value": obj.cpu().tolist()}
        return super().default(obj)

def tensor_hook(dct):
    if "__tensor__" in dct:
        return torch.tensor(dct["value"])
    return dct

def log(msg):
    print(f"[RemoteCLIP {time.strftime('%H:%M:%S')}] {msg}", flush=True)

def recv_exact(sock, size):
    buf = b""
    while len(buf) < size:
        chunk = sock.recv(size - len(buf))
        if not chunk:
            raise ConnectionError("Socket closed")
        buf += chunk
    return buf

def send_packet(sock, meta, blob):
    meta_bytes = json.dumps(meta, cls=TensorEncoder).encode("utf-8")
    sock.sendall(struct.pack(">Q", len(meta_bytes)))
    sock.sendall(meta_bytes)
    sock.sendall(blob)

def recv_packet(sock):
    meta_size = struct.unpack(">Q", recv_exact(sock, 8))[0]
    meta = json.loads(recv_exact(sock, meta_size).decode("utf-8"), object_hook=tensor_hook)
    blob = recv_exact(sock, meta.get("blob_size", 0))
    return meta, blob

def pack_tensors(tensors):
    meta = {}
    blobs = []
    for name, t in tensors.items():
        if t is None:
            log(f"Warning: tensor '{name}' is None, replacing with zeros")
            if name == "cond":
                t = torch.zeros((1, 77, 768))
            else:
                t = torch.zeros((1, 768))
        t = t.contiguous().cpu()
        raw = t.numpy().tobytes()
        meta[name] = {"dtype": str(t.dtype), "shape": list(t.shape), "size": len(raw)}
        blobs.append(raw)
    return meta, b"".join(blobs)

def unpack_tensors(meta, blob):
    out = {}
    offset = 0
    for name, info in meta.items():
        size = info["size"]
        raw = bytearray(blob[offset:offset + size])
        offset += size
        dtype = getattr(torch, info["dtype"].split(".")[-1])
        t = torch.frombuffer(raw, dtype=dtype).clone()
        out[name] = t.reshape(info["shape"])
    return out

class RemoteCLIPProxy:
    def __init__(self, ip, port):
        self.ip = ip
        self.port = port
        self.sock = socket.create_connection((ip, port), timeout=60)
        log(f"Connected to worker {ip}:{port}")

    def clone(self):
        return self

    def add_patches(self, *_, **__):
        return self

    def tokenize(self, text, **kwargs):
        return {"text": text, "kwargs": kwargs}

    def _infer(self, tokens):
        meta = {"cmd": "encode", "text": tokens["text"], "kwargs": tokens["kwargs"], "blob_size": 0}
        try:
            log("Sending encode request")
            send_packet(self.sock, meta, b"")
            resp_meta, blob = recv_packet(self.sock)
            tensors = unpack_tensors(resp_meta["tensors"], blob)
            cond = tensors.get("cond", torch.zeros((1, 77, 768)))
            pooled = tensors.get("pooled", torch.zeros((1, 768)))
            log("Received embeddings")
            return cond, pooled
        except Exception as e:
            log(f"Remote CLIP inference failed: {e}")
            return torch.zeros((1, 77, 768)), torch.zeros((1, 768))

    def encode_from_tokens(self, tokens, return_pooled=True, return_dict=False):
        cond, pooled = self._infer(tokens)
        if return_dict:
            return {"cond": cond, "pooled_output": pooled}
        return cond, pooled

    def encode_from_tokens_scheduled(self, tokens):
        cond, pooled = self._infer(tokens)
        return [[cond, {"pooled_output": pooled}]]

class SendRemoteCLIP:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"CLIP": ("CLIP",), "listen_port": ("INT", {"default": 8181})}}

    RETURN_TYPES = ()
    FUNCTION = "start_worker"
    OUTPUT_NODE = True
    CATEGORY = "Remote CLIP"

    def start_worker(self, CLIP, listen_port):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("0.0.0.0", listen_port))
        server.listen(16)
        log(f"Worker listening on port {listen_port}")

        def handle(conn):
            addr = conn.getpeername()
            log(f"Client connected: {addr}")
            with conn:
                while True:
                    try:
                        meta, _ = recv_packet(conn)
                        if meta is None or "text" not in meta:
                            log("Warning: invalid request, skipping")
                            continue
                        log(f"Encoding prompt ({len(meta['text'])} chars)")
                        tokens = CLIP.tokenize(meta["text"], **meta["kwargs"])
                        cond, pooled = CLIP.encode_from_tokens(tokens, return_pooled=True)
                        tensor_meta, blob = pack_tensors({"cond": cond, "pooled": pooled})
                        reply_meta = {"tensors": tensor_meta, "blob_size": len(blob)}
                        send_packet(conn, reply_meta, blob)
                        log(f"Sent embeddings ({len(blob)} bytes)")
                    except Exception as e:
                        log(f"Client disconnected: {e}")
                        break

        while True:
            conn, _ = server.accept()
            threading.Thread(target=handle, args=(conn,), daemon=True).start()

class LoadRemoteCLIP:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"worker_ip": ("STRING", {"default": "127.0.0.1"}), "port": ("INT", {"default": 8181})}}

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load_remote"
    CATEGORY = "Remote CLIP"

    def load_remote(self, worker_ip, port):
        return (RemoteCLIPProxy(worker_ip, port),)
    
class LoraLoaderCLIPOnly:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "CLIP": ("CLIP",),
                "lora_name": (folder_paths.get_filename_list("loras"), ),
                "strength_clip": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load_lora"
    CATEGORY = "Remote CLIP"

    def load_lora(self, CLIP, lora_name, strength_clip):
        lora_path = folder_paths.get_full_path("loras", lora_name)
        lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
        _, clip_lora = comfy.sd.load_lora_for_models(None, CLIP, lora, 0, strength_clip)
        return (clip_lora,)

NODE_CLASS_MAPPINGS = {"SendRemoteCLIP": SendRemoteCLIP, "LoadRemoteCLIP": LoadRemoteCLIP, "LoraLoaderCLIPOnly": LoraLoaderCLIPOnly}
NODE_DISPLAY_NAME_MAPPINGS = {"SendRemoteCLIP": "Send Remote CLIP", "LoadRemoteCLIP": "Load Remote CLIP", "LoraLoaderCLIPOnly": "LoraLoaderCLIPOnly"}