import socket
import struct
import threading
import json
import torch
import time

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
    meta_bytes = json.dumps(meta).encode("utf-8")
    sock.sendall(struct.pack(">Q", len(meta_bytes)))
    sock.sendall(meta_bytes)
    sock.sendall(blob)

def recv_packet(sock):
    meta_size = struct.unpack(">Q", recv_exact(sock, 8))[0]
    meta = json.loads(recv_exact(sock, meta_size).decode("utf-8"))
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
        raw = blob[offset:offset + size]
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

class RemoteCLIPWorker:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"clip": ("CLIP",), "listen_port": ("INT", {"default": 8002})}}

    RETURN_TYPES = ()
    FUNCTION = "start_worker"
    OUTPUT_NODE = True
    CATEGORY = "Remote CLIP"

    def start_worker(self, clip, listen_port):
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
                        tokens = clip.tokenize(meta["text"], **meta["kwargs"])
                        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
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

class RemoteCLIPLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"worker_ip": ("STRING", {"default": "10.0.0.37"}), "port": ("INT", {"default": 8002})}}

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load_remote"
    CATEGORY = "Remote CLIP"

    def load_remote(self, worker_ip, port):
        return (RemoteCLIPProxy(worker_ip, port),)

NODE_CLASS_MAPPINGS = {"RemoteCLIPWorker": RemoteCLIPWorker, "RemoteCLIPLoader": RemoteCLIPLoader}
NODE_DISPLAY_NAME_MAPPINGS = {"RemoteCLIPWorker": "Remote CLIP Worker", "RemoteCLIPLoader": "Remote CLIP Loader"}