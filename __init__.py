import os
import json
import time
import hmac
import socket
import struct
import threading
from collections import OrderedDict
import torch
import folder_paths
import comfy.sd
import comfy.utils
DEFAULT_PORT = 8181
PROTOCOL_VERSION = 1                   
HEADER_LEN_FMT = ">Q"                 
HEADER_LEN_SIZE = struct.calcsize(HEADER_LEN_FMT)
SOCKET_TIMEOUT = 120                  
CONNECT_RETRIES = 3
CONNECT_BACKOFF = 1.0                 
RECV_CHUNK = 1 << 20                  
MAX_HEADER_BYTES = 8 * 1024 * 1024    
MAX_BLOB_BYTES = 512 * 1024 * 1024
PATCHED_CLIP_CACHE = 4                
EMBED_CACHE = 64                      
ALLOWED_DTYPES = {
    "torch.float32": torch.float32,
    "torch.float16": torch.float16,
    "torch.bfloat16": torch.bfloat16,
}
def log(msg):
    print(f"[RemoteCLIP {time.strftime('%H:%M:%S')}] {msg}", flush=True)
_LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1", "0.0.0.0"}
def resolve_transport_dtype(mode, ip):
    """Map a transport mode + worker host to a transport dtype string.
    Returns a key into ALLOWED_DTYPES, or None to send tensors as-is (no
    downcast, full precision).
      - "fp16": always send float tensors as float16 (half the bandwidth).
      - "fp32": never downcast; preserve full precision.
      - "auto": fp16 for remote workers (bandwidth matters), full precision
                on localhost (bandwidth is free, so don't lose precision).
    """
    if mode == "fp16":
        return "torch.float16"
    if mode == "fp32":
        return None
    # auto
    is_local = ip in _LOCAL_HOSTS
    return None if is_local else "torch.float16"
def _recv_exact(sock, size):
    chunks = []
    remaining = size
    while remaining > 0:
        chunk = sock.recv(min(remaining, RECV_CHUNK))
        if not chunk:
            raise ConnectionError("Socket closed while reading")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)
def send_packet(sock, header, blob=b""):
    header_bytes = json.dumps(header).encode("utf-8")
    prefix = struct.pack(HEADER_LEN_FMT, len(header_bytes))
    sock.sendall(prefix + header_bytes)
    if blob:
        sock.sendall(blob)
def recv_header(sock):
    """Read and parse just the JSON header (no blob)."""
    header_len = struct.unpack(HEADER_LEN_FMT, _recv_exact(sock, HEADER_LEN_SIZE))[0]
    if header_len > MAX_HEADER_BYTES:
        raise ValueError(f"Header too large: {header_len} bytes")
    return json.loads(_recv_exact(sock, header_len).decode("utf-8"))
def recv_blob(sock, header):
    """Read the blob declared by a previously-read header."""
    blob_size = header.get("blob_size", 0)
    if blob_size > MAX_BLOB_BYTES:
        raise ValueError(f"Blob too large: {blob_size} bytes")
    return _recv_exact(sock, blob_size) if blob_size else b""
def recv_packet(sock):
    header = recv_header(sock)
    blob = recv_blob(sock, header)
    return header, blob
def _set_socket_opts(sock):
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    sock.settimeout(SOCKET_TIMEOUT)
def pack_tensors(tensors, transport_dtype=None):
    """Serialize {name: tensor} into (metadata, blob).
    transport_dtype: optional torch dtype to cast floating tensors to before
    transfer (e.g. float16 to halve bandwidth). The original dtype is recorded
    so the receiver can restore it.
    """
    meta = {}
    blobs = []
    offset = 0
    for name, t in tensors.items():
        if t is None:
            raise ValueError(f"Tensor '{name}' is None")
        t = t.detach().cpu()
        orig_dtype = str(t.dtype)
        if t.is_floating_point():
            if transport_dtype is not None:
                t = t.to(transport_dtype)
            elif t.dtype not in (torch.float16, torch.float32):
                t = t.to(torch.float32)
        t = t.contiguous()
        raw = t.numpy().tobytes()
        meta[name] = {
            "dtype": str(t.dtype),
            "orig_dtype": orig_dtype,
            "shape": list(t.shape),
            "offset": offset,
            "size": len(raw),
        }
        offset += len(raw)
        blobs.append(raw)
    return meta, b"".join(blobs)
def unpack_tensors(meta, blob):
    out = {}
    view = memoryview(blob)
    for name, info in meta.items():
        dtype_name = info["dtype"]
        if dtype_name not in ALLOWED_DTYPES:
            raise ValueError(f"Refusing to decode disallowed dtype: {dtype_name}")
        dtype = ALLOWED_DTYPES[dtype_name]
        start = info["offset"]
        raw = bytearray(view[start:start + info["size"]])
        # .clone() so the tensor owns its memory instead of aliasing `raw`.
        t = torch.frombuffer(raw, dtype=dtype).reshape(info["shape"]).clone()
        orig = info.get("orig_dtype")
        if orig in ALLOWED_DTYPES and ALLOWED_DTYPES[orig] != dtype:
            t = t.to(ALLOWED_DTYPES[orig])
        out[name] = t
    return out
class _Connection:
    """Thread-safe, reconnecting TCP connection to a worker. Shared by proxy clones."""
    def __init__(self, ip, port, auth_token=""):
        self.ip = ip
        self.port = port
        self.auth_token = auth_token
        self._sock = None
        self._lock = threading.Lock()
    def _connect(self):
        last_err = None
        backoff = CONNECT_BACKOFF
        for attempt in range(1, CONNECT_RETRIES + 1):
            try:
                sock = socket.create_connection((self.ip, self.port), timeout=SOCKET_TIMEOUT)
                _set_socket_opts(sock)
                self._sock = sock
                log(f"Connected to worker {self.ip}:{self.port}")
                return
            except OSError as e:
                last_err = e
                log(f"Connect attempt {attempt}/{CONNECT_RETRIES} failed: {e}")
                if attempt < CONNECT_RETRIES:
                    time.sleep(backoff)
                    backoff *= 2
        raise ConnectionError(
            f"Could not connect to worker {self.ip}:{self.port}: {last_err}"
        )
    def _close(self):
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None
    def request(self, header, blob=b""):
        """Send a request, return (header, blob). Reconnects once on failure."""
        if self.auth_token:
            header = {**header, "auth": self.auth_token}
        with self._lock:
            for attempt in (1, 2):
                try:
                    if self._sock is None:
                        self._connect()
                    send_packet(self._sock, header, blob)
                    return recv_packet(self._sock)
                except (ConnectionError, OSError, struct.error, ValueError) as e:
                    log(f"Request failed ({e}); reconnecting (attempt {attempt})")
                    self._close()
                    if attempt == 2:
                        raise
class RemoteCLIPProxy:
    """Stands in for a ComfyUI CLIP object but encodes on a remote worker."""
    def __init__(self, ip, port, auth_token="", transport_mode="auto",
                 connection=None, lora_stack=None):
        self.ip = ip
        self.port = port
        self.transport_mode = transport_mode
        self.transport_dtype = resolve_transport_dtype(transport_mode, ip)
        self._conn = connection or _Connection(ip, port, auth_token)
        self.lora_stack = list(lora_stack or [])
    def clone(self):
        return RemoteCLIPProxy(
            self.ip, self.port,
            transport_mode=self.transport_mode,
            connection=self._conn,
            lora_stack=self.lora_stack,
        )
    def with_lora(self, lora_name, strength_clip):
        c = self.clone()
        c.lora_stack = self.lora_stack + [[lora_name, float(strength_clip)]]
        return c
    def add_patches(self, *_args, **_kwargs):
        raise NotImplementedError(
            "Apply LoRAs to a remote CLIP with the 'LoraLoaderCLIPOnly' node, "
            "which forwards them to the worker."
        )
    def tokenize(self, text, **kwargs):
        return {"text": text, "kwargs": kwargs, "lora_stack": self.lora_stack}
    def _infer(self, tokens):
        header = {
            "cmd": "encode",
            "proto": PROTOCOL_VERSION,
            "text": tokens["text"],
            "kwargs": tokens.get("kwargs", {}),
            "lora_stack": tokens.get("lora_stack", self.lora_stack),
            "transport_dtype": self.transport_dtype,
            "blob_size": 0,
        }
        log("Sending encode request")
        resp, blob = self._conn.request(header)
        if resp.get("error"):
            raise RuntimeError(f"Remote CLIP worker error: {resp['error']}")
        tensors = unpack_tensors(resp["tensors"], blob)
        log("Received embeddings")
        return tensors["cond"], tensors["pooled"]
    def encode_from_tokens(self, tokens, return_pooled=True, return_dict=False):
        cond, pooled = self._infer(tokens)
        if return_dict:
            return {"cond": cond, "pooled_output": pooled}
        if return_pooled:
            return cond, pooled
        return cond
    def encode_from_tokens_scheduled(self, tokens):
        cond, pooled = self._infer(tokens)
        return [[cond, {"pooled_output": pooled}]]
class _Worker:
    def __init__(self, clip, auth_token=""):
        self.base_clip = clip
        self.auth_token = auth_token
        self._cache_lock = threading.Lock()   
        self._infer_lock = threading.Lock()    
        self._patched_clips = OrderedDict()   
        self._embed_cache = OrderedDict()      
    @staticmethod
    def _sig(obj):
        return json.dumps(obj, sort_keys=True)
    def _get_clip(self, lora_stack):
        if not lora_stack:
            return self.base_clip
        sig = self._sig(lora_stack)
        clip = self._patched_clips.get(sig)
        if clip is not None:
            self._patched_clips.move_to_end(sig)
            return clip
        clip = self.base_clip
        for lora_name, strength in lora_stack:
            if strength == 0:
                continue
            lora_path = folder_paths.get_full_path("loras", lora_name)
            if lora_path is None:
                raise FileNotFoundError(f"LoRA not found on worker: {lora_name}")
            lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
            _, clip = comfy.sd.load_lora_for_models(None, clip, lora, 0, strength)
        self._patched_clips[sig] = clip
        while len(self._patched_clips) > PATCHED_CLIP_CACHE:
            self._patched_clips.popitem(last=False)
        return clip
    def encode(self, text, kwargs, lora_stack):
        cache_key = self._sig([text, kwargs, lora_stack])
        # Fast path: serve cached embeddings without blocking on GPU inference.
        with self._cache_lock:
            cached = self._embed_cache.get(cache_key)
            if cached is not None:
                self._embed_cache.move_to_end(cache_key)
                return cached
        # Slow path: serialize the actual GPU work on its own lock so that
        # concurrent cache hits aren't stuck behind an in-flight encode.
        with self._infer_lock:
            # Re-check: another thread may have computed this while we waited.
            with self._cache_lock:
                cached = self._embed_cache.get(cache_key)
                if cached is not None:
                    self._embed_cache.move_to_end(cache_key)
                    return cached
            clip = self._get_clip(lora_stack)
            with torch.inference_mode():
                tokens = clip.tokenize(text, **kwargs)
                cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
            cond = cond.detach().cpu()
            if pooled is None:
                pooled = torch.zeros((cond.shape[0], cond.shape[-1]))
            else:
                pooled = pooled.detach().cpu()
            # Clone before caching so a downstream in-place mutation on the
            # returned tensors can never poison the cache.
            result = (cond.clone(), pooled.clone())
            with self._cache_lock:
                self._embed_cache[cache_key] = result
                while len(self._embed_cache) > EMBED_CACHE:
                    self._embed_cache.popitem(last=False)
            return result
class SendRemoteCLIP:
    _servers = {}  
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "CLIP": ("CLIP",),
                "listen_port": ("INT", {"default": DEFAULT_PORT, "min": 1, "max": 65535}),
            },
            "optional": {
                "bind_host": ("STRING", {"default": "0.0.0.0"}),
                "auth_token": ("STRING", {"default": ""}),
            },
        }
    RETURN_TYPES = ()
    FUNCTION = "start_worker"
    OUTPUT_NODE = True
    CATEGORY = "Remote CLIP"
    def start_worker(self, CLIP, listen_port, bind_host="0.0.0.0", auth_token=""):
        token = auth_token or os.environ.get("REMOTE_CLIP_TOKEN", "")
        existing = SendRemoteCLIP._servers.pop(listen_port, None)
        if existing is not None:
            try:
                existing.close()
            except OSError:
                pass
        worker = _Worker(CLIP, token)
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((bind_host, listen_port))
        server.listen(16)
        SendRemoteCLIP._servers[listen_port] = server
        if bind_host == "0.0.0.0" and not token:
            log("WARNING: worker bound to 0.0.0.0 with no auth token. Anyone on "
                "the network can use this CLIP. Set auth_token or bind to a "
                "specific host.")
        log(f"Worker listening on {bind_host}:{listen_port}")
        def handle(conn, addr):
            log(f"Client connected: {addr}")
            try:
                _set_socket_opts(conn)
                while True:
                    # Read only the header first. The blob is read after the
                    # client is authorized, so an unauthenticated peer can't
                    # make us buffer a large (up to MAX_BLOB_BYTES) payload.
                    header = recv_header(conn)
                    if token and not hmac.compare_digest(header.get("auth", ""), token):
                        send_packet(conn, {"error": "unauthorized", "blob_size": 0})
                        log(f"Rejected unauthorized client {addr}")
                        break
                    client_proto = header.get("proto", 0)
                    if client_proto != PROTOCOL_VERSION:
                        send_packet(conn, {
                            "error": (f"protocol mismatch: worker speaks v{PROTOCOL_VERSION}, "
                                      f"client sent v{client_proto}"),
                            "blob_size": 0,
                        })
                        log(f"Rejected client {addr}: protocol v{client_proto} "
                            f"!= v{PROTOCOL_VERSION}")
                        break
                    # Authorized: now it's safe to read any declared blob.
                    recv_blob(conn, header)
                    if header.get("cmd") != "encode" or "text" not in header:
                        send_packet(conn, {"error": "bad request", "blob_size": 0})
                        continue
                    try:
                        transport = header.get("transport_dtype")
                        transport_dtype = ALLOWED_DTYPES.get(transport) if transport else None
                        cond, pooled = worker.encode(
                            header["text"],
                            header.get("kwargs", {}),
                            header.get("lora_stack", []),
                        )
                        meta, blob = pack_tensors(
                            {"cond": cond, "pooled": pooled}, transport_dtype
                        )
                        send_packet(conn, {"tensors": meta, "blob_size": len(blob)}, blob)
                        log(f"Sent embeddings ({len(blob)} bytes)")
                    except Exception as e:
                        log(f"Encode failed: {e}")
                        send_packet(conn, {"error": str(e), "blob_size": 0})
            except (ConnectionError, OSError) as e:
                log(f"Client {addr} disconnected: {e}")
            finally:
                try:
                    conn.close()
                except OSError:
                    pass
        def accept_loop():
            while True:
                try:
                    conn, addr = server.accept()
                except OSError:
                    log("Server socket closed; stopping accept loop")
                    break
                threading.Thread(target=handle, args=(conn, addr), daemon=True).start()
        threading.Thread(target=accept_loop, daemon=True).start()
        return {"ui": {"text": [f"Remote CLIP worker on {bind_host}:{listen_port}"]}}
class LoadRemoteCLIP:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "worker_ip": ("STRING", {"default": "127.0.0.1"}),
                "port": ("INT", {"default": DEFAULT_PORT, "min": 1, "max": 65535}),
            },
            "optional": {
                "auth_token": ("STRING", {"default": ""}),
                "transport_precision": (["auto", "fp16", "fp32"], {"default": "auto"}),
            },
        }
    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load_remote"
    CATEGORY = "Remote CLIP"
    def load_remote(self, worker_ip, port, auth_token="", transport_precision="auto"):
        token = auth_token or os.environ.get("REMOTE_CLIP_TOKEN", "")
        return (RemoteCLIPProxy(worker_ip, port, token, transport_precision),)
class LoraLoaderCLIPOnly:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "CLIP": ("CLIP",),
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "strength_clip": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
            }
        }
    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load_lora"
    CATEGORY = "Remote CLIP"
    def load_lora(self, CLIP, lora_name, strength_clip):
        if isinstance(CLIP, RemoteCLIPProxy):
            return (CLIP.with_lora(lora_name, strength_clip),)
        if strength_clip == 0:
            return (CLIP,)
        lora_path = folder_paths.get_full_path("loras", lora_name)
        lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
        _, clip_lora = comfy.sd.load_lora_for_models(None, CLIP, lora, 0, strength_clip)
        return (clip_lora,)
NODE_CLASS_MAPPINGS = {
    "SendRemoteCLIP": SendRemoteCLIP,
    "LoadRemoteCLIP": LoadRemoteCLIP,
    "LoraLoaderCLIPOnly": LoraLoaderCLIPOnly,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SendRemoteCLIP": "Send Remote CLIP",
    "LoadRemoteCLIP": "Load Remote CLIP",
    "LoraLoaderCLIPOnly": "LoraLoaderCLIPOnly",
}
