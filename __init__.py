import os
import math
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
PROTOCOL_VERSION = 3                   
HEADER_LEN_FMT = ">Q"                 
HEADER_LEN_SIZE = struct.calcsize(HEADER_LEN_FMT)
SOCKET_TIMEOUT = 120                  
GENERATE_TIMEOUT = 3600               
CONNECT_RETRIES = 3
CONNECT_BACKOFF = 1.0                 
RECV_CHUNK = 1 << 20                  
MAX_HEADER_BYTES = 8 * 1024 * 1024    
MAX_BLOB_BYTES = 512 * 1024 * 1024
PATCHED_CLIP_CACHE = 4                
ALLOWED_DTYPES = {
    "torch.float32": torch.float32,
    "torch.float16": torch.float16,
    "torch.bfloat16": torch.bfloat16,
    "torch.int64": torch.int64,
    "torch.int32": torch.int32,
    "torch.uint8": torch.uint8,
    "torch.bool": torch.bool,
}
def log(msg):
    print(f"[RemoteCLIP {time.strftime('%H:%M:%S')}] {msg}", flush=True)
_LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1", "0.0.0.0"}
def resolve_transport_dtype(mode, ip):
    if mode == "fp16":
        return "torch.float16"
    if mode == "fp32":
        return "torch.float32"
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
    if len(header_bytes) > MAX_HEADER_BYTES or len(blob) > MAX_BLOB_BYTES:
        raise ValueError("Packet exceeds protocol size limits")
    prefix = struct.pack(HEADER_LEN_FMT, len(header_bytes))
    sock.sendall(prefix + header_bytes)
    if blob:
        sock.sendall(blob)
def recv_header(sock):
    header_len = struct.unpack(HEADER_LEN_FMT, _recv_exact(sock, HEADER_LEN_SIZE))[0]
    if header_len > MAX_HEADER_BYTES:
        raise ValueError(f"Header too large: {header_len} bytes")
    header = json.loads(_recv_exact(sock, header_len).decode("utf-8"))
    if not isinstance(header, dict):
        raise ValueError("Packet header must be an object")
    return header
def recv_blob(sock, header):
    blob_size = header.get("blob_size", 0)
    if type(blob_size) is not int or not 0 <= blob_size <= MAX_BLOB_BYTES:
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
        start, size, shape = info["offset"], info["size"], info["shape"]
        if (type(start) is not int or type(size) is not int or
                start < 0 or size < 0 or start + size > len(blob)):
            raise ValueError("Invalid tensor byte range")
        if not isinstance(shape, list) or any(type(d) is not int or d < 0 for d in shape):
            raise ValueError("Invalid tensor shape")
        count = math.prod(shape)
        if count * torch.empty((), dtype=dtype).element_size() != size:
            raise ValueError("Tensor shape does not match byte size")
        if count == 0:
            t = torch.empty(shape, dtype=dtype)
        else:
            raw = bytearray(view[start:start + size])
            t = torch.frombuffer(raw, dtype=dtype).reshape(shape).clone()
        orig = info.get("orig_dtype")
        if orig in ALLOWED_DTYPES and ALLOWED_DTYPES[orig] != dtype:
            t = t.to(ALLOWED_DTYPES[orig])
        out[name] = t
    return out
def extract_tensors(obj, prefix, out):
    # Tag every container so user dictionaries cannot mimic tensor references.
    if isinstance(obj, torch.Tensor):
        key = f"tensor_{len(out)}"
        out[key] = obj
        return {"kind": "tensor", "value": key}
    if isinstance(obj, dict):
        return {"kind": "dict", "value": [
            [extract_tensors(k, prefix, out), extract_tensors(v, prefix, out)]
            for k, v in obj.items()]}
    if isinstance(obj, (list, tuple)):
        return {"kind": "tuple" if isinstance(obj, tuple) else "list",
                "value": [extract_tensors(v, prefix, out) for v in obj]}
    return {"kind": "value", "value": obj}


def restore_tensors(obj, tensors):
    kind, value = obj["kind"], obj["value"]
    if kind == "tensor":
        return tensors[value]
    if kind == "dict":
        return {restore_tensors(k, tensors): restore_tensors(v, tensors) for k, v in value}
    if kind in ("list", "tuple"):
        values = [restore_tensors(v, tensors) for v in value]
        return tuple(values) if kind == "tuple" else values
    if kind != "value":
        raise ValueError(f"Invalid structure kind: {kind}")
    return value


def pack_object(obj, transport_dtype=None):
    tensors = {}
    structure = extract_tensors(obj, "", tensors)
    meta, blob = pack_tensors(tensors, transport_dtype)
    return {"struct": structure, "tensors": meta, "blob_size": len(blob)}, blob


def unpack_object(header, blob):
    return restore_tensors(header["struct"], unpack_tensors(header["tensors"], blob))


class _Connection:
    def __init__(self, ip, port, auth_token="", connect_timeout=SOCKET_TIMEOUT,
                 request_attempts=CONNECT_RETRIES, connect_backoff=CONNECT_BACKOFF,
                 encode_timeout=SOCKET_TIMEOUT):
        self.ip, self.port, self.auth_token = ip, port, auth_token
        self.connect_timeout = float(connect_timeout)
        self.request_attempts = int(request_attempts)
        self.connect_backoff = float(connect_backoff)
        self.encode_timeout = float(encode_timeout)
        for value in (self.connect_timeout, self.encode_timeout):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Timeouts must be finite and positive")
        if self.request_attempts < 1:
            raise ValueError("Request attempts must be at least 1")
        if not math.isfinite(self.connect_backoff) or self.connect_backoff < 0:
            raise ValueError("Retry delay must be finite and nonnegative")
        self._sock = None
        self._lock = threading.Lock()

    def _connect(self):
        sock = socket.create_connection((self.ip, self.port), timeout=self.connect_timeout)
        try:
            _set_socket_opts(sock)
        except Exception:
            sock.close()
            raise
        self._sock = sock
        log(f"Connected to worker {self.ip}:{self.port}")

    def _close(self):
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None

    def request(self, header, blob=b"", response_timeout=None, replay_safe=True):
        header = {**header, "auth": self.auth_token}
        timeout = self.encode_timeout if response_timeout is None else response_timeout
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Response timeout must be finite and positive")
        with self._lock:
            backoff = self.connect_backoff
            for attempt in range(1, self.request_attempts + 1):
                sending = False
                try:
                    if self._sock is None:
                        self._connect()
                    self._sock.settimeout(timeout)
                    sending = True
                    send_packet(self._sock, header, blob)
                    result = recv_packet(self._sock)
                    self._sock.settimeout(self.encode_timeout)
                    return result
                except (OSError, ConnectionError, struct.error, ValueError):
                    self._close()
                    # Generation may have executed even when its response was lost.
                    if attempt == self.request_attempts or (sending and not replay_safe):
                        raise
                    log(f"Request attempt {attempt}/{self.request_attempts} failed; retrying")
                    time.sleep(backoff)
                    backoff *= 2


class RemoteCLIPProxy:
    def __init__(self, ip, port, auth_token="", transport_mode="auto",
                 connection=None, lora_stack=None, connect_timeout=SOCKET_TIMEOUT,
                 request_attempts=CONNECT_RETRIES, connect_backoff=CONNECT_BACKOFF,
                 encode_timeout=SOCKET_TIMEOUT, generate_timeout=GENERATE_TIMEOUT,
                 layer_idx=None):
        self.ip, self.port = ip, port
        self.transport_mode = transport_mode
        self.transport_dtype_name = resolve_transport_dtype(transport_mode, ip)
        self.transport_dtype = ALLOWED_DTYPES.get(self.transport_dtype_name)
        self.generate_timeout = float(generate_timeout)
        if not math.isfinite(self.generate_timeout) or self.generate_timeout <= 0:
            raise ValueError("Generation timeout must be finite and positive")
        self._conn = connection or _Connection(
            ip, port, auth_token, connect_timeout, request_attempts,
            connect_backoff, encode_timeout)
        self.lora_stack = list(lora_stack or [])
        self.layer_idx = layer_idx

    def clone(self, disable_dynamic=False):
        return RemoteCLIPProxy(
            self.ip, self.port, transport_mode=self.transport_mode,
            connection=self._conn, lora_stack=self.lora_stack,
            generate_timeout=self.generate_timeout, layer_idx=self.layer_idx)

    def clip_layer(self, layer_idx):
        self.layer_idx = layer_idx

    def with_lora(self, lora_name, strength_clip):
        c = self.clone()
        c.lora_stack = self.lora_stack + [[lora_name, float(strength_clip)]]
        return c

    def add_patches(self, *_args, **_kwargs):
        raise NotImplementedError(
            "Apply LoRAs to a remote CLIP with the 'LoraLoaderCLIPOnly' node.")

    def _rpc(self, cmd, payload, generation=False):
        # Token tensors retain their original numerical values. Transport precision
        # applies to image/tokenizer inputs and embedding responses, not token IDs.
        header, blob = pack_object(payload)
        header.update(cmd=cmd, proto=PROTOCOL_VERSION,
                      lora_stack=self.lora_stack, layer_idx=self.layer_idx,
                      transport_dtype=self.transport_dtype_name)
        response, data = self._conn.request(
            header, blob,
            response_timeout=self.generate_timeout if generation else None,
            replay_safe=not generation)
        if response.get("error"):
            raise RuntimeError(f"Remote CLIP worker error: {response['error']}")
        return unpack_object(response, data)

    def tokenize(self, text, return_word_ids=False, **kwargs):
        # Return actual token dictionaries: callers such as SDXL edit l/g streams.
        return self._rpc("tokenize", {"text": text, "kwargs": {
            **kwargs, "return_word_ids": return_word_ids}})

    def encode_from_tokens(self, tokens, return_pooled=False, return_dict=False):
        out = self._rpc("encode", {"tokens": tokens, "return_pooled": return_pooled})
        if return_dict:
            return out
        if return_pooled:
            return out["cond"], out.get("pooled_output")
        return out["cond"]

    def encode_from_tokens_scheduled(self, tokens, unprojected=False, add_dict=None, show_pbar=True):
        return self._rpc("encode_scheduled", {
            "tokens": tokens, "unprojected": unprojected,
            "add_dict": {} if add_dict is None else add_dict, "show_pbar": show_pbar})

    def generate(self, tokens, **gen_kwargs):
        return self._rpc("generate", {"tokens": tokens, "gen_kwargs": gen_kwargs}, generation=True)

    def decode(self, generated, skip_special_tokens=True):
        return generated


class _Worker:
    def __init__(self, clip, auth_token=""):
        self.base_clip = clip
        self.auth_token = auth_token
        self._infer_lock = threading.Lock()
        self._patched_clips = OrderedDict()

    def _get_clip(self, lora_stack, layer_idx):
        sig = json.dumps([lora_stack, layer_idx], sort_keys=True)
        if sig in self._patched_clips:
            self._patched_clips.move_to_end(sig)
            return self._patched_clips[sig]
        clip = self.base_clip.clone()
        if layer_idx is not None:
            clip.clip_layer(layer_idx)
        for lora_name, strength in lora_stack:
            if strength == 0:
                continue
            path = folder_paths.get_full_path("loras", lora_name)
            if path is None:
                raise FileNotFoundError(f"LoRA not found on worker: {lora_name}")
            lora = comfy.utils.load_torch_file(path, safe_load=True)
            _, clip = comfy.sd.load_lora_for_models(None, clip, lora, 0, strength)
        self._patched_clips[sig] = clip
        while len(self._patched_clips) > PATCHED_CLIP_CACHE:
            self._patched_clips.popitem(last=False)
        return clip

    def execute(self, cmd, payload, lora_stack, layer_idx):
        with self._infer_lock, torch.inference_mode():
            clip = self._get_clip(lora_stack, layer_idx)
            if cmd == "tokenize":
                return clip.tokenize(payload["text"], **payload["kwargs"])
            if cmd == "generate":
                return clip.decode(clip.generate(payload["tokens"], **payload["gen_kwargs"]))
            if cmd == "encode":
                out = clip.encode_from_tokens(payload["tokens"],
                    return_pooled=payload["return_pooled"], return_dict=True)
            elif cmd == "encode_scheduled":
                out = clip.encode_from_tokens_scheduled(**payload)
            else:
                raise ValueError(f"Unknown command: {cmd}")
            return out


class _Server:
    def __init__(self, clip, host, port, token):
        self.worker = _Worker(clip, token)
        self.token = token
        self.stopped = threading.Event()
        self._clients = set()
        self._clients_lock = threading.Lock()
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.socket.bind((host, port))
            self.socket.listen(16)
            self.socket.settimeout(0.5)
        except Exception:
            self.socket.close()
            raise
        self.thread = threading.Thread(target=self._accept_loop, daemon=True)
        self.thread.start()

    @staticmethod
    def _close_socket(sock):
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        sock.close()

    def close(self):
        self.stopped.set()
        self._close_socket(self.socket)
        with self._clients_lock:
            clients = list(self._clients)
        for conn in clients:
            self._close_socket(conn)
        self.thread.join(timeout=1)

    def _accept_loop(self):
        while not self.stopped.is_set():
            try:
                conn, addr = self.socket.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            with self._clients_lock:
                if self.stopped.is_set():
                    conn.close()
                    break
                self._clients.add(conn)
            threading.Thread(target=self._handle, args=(conn, addr), daemon=True).start()

    def _handle(self, conn, addr):
        try:
            _set_socket_opts(conn)
            while not self.stopped.is_set():
                header = recv_header(conn)
                auth = header.get("auth", "")
                if self.token and (not isinstance(auth, str) or not hmac.compare_digest(
                        auth.encode("utf-8"), self.token.encode("utf-8"))):
                    send_packet(conn, {"error": "unauthorized", "blob_size": 0})
                    break
                if header.get("proto") != PROTOCOL_VERSION:
                    send_packet(conn, {"error": f"Protocol mismatch: install v{PROTOCOL_VERSION} on both ends", "blob_size": 0})
                    break
                blob = recv_blob(conn, header)
                try:
                    payload = unpack_object(header, blob)
                    if self.stopped.is_set():
                        break
                    out = self.worker.execute(header["cmd"], payload,
                        header.get("lora_stack", []), header.get("layer_idx"))
                    if self.stopped.is_set():
                        break
                    transport = header.get("transport_dtype")
                    if transport not in (None, "torch.float16", "torch.float32"):
                        raise ValueError("Invalid transport precision")
                    dtype = ALLOWED_DTYPES.get(transport) if header["cmd"] in ("encode", "encode_scheduled") else None
                    response, data = pack_object(out, dtype)
                    send_packet(conn, response, data)
                except Exception as e:
                    if self.stopped.is_set():
                        break
                    log(f"Worker request failed: {e}")
                    send_packet(conn, {"error": str(e), "blob_size": 0})
        except (ConnectionError, OSError, ValueError, TypeError, KeyError, struct.error) as e:
            log(f"Client {addr} disconnected: {e}")
        finally:
            with self._clients_lock:
                self._clients.discard(conn)
            self._close_socket(conn)


class SendRemoteCLIP:
    _servers = {}
    _servers_lock = threading.Lock()

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "CLIP": ("CLIP",),
            "listen_port": ("INT", {"default": DEFAULT_PORT, "min": 1, "max": 65535}),
        }, "optional": {
            "bind_host": ("STRING", {"default": "0.0.0.0"}),
            "auth_token": ("STRING", {"default": ""}),
        }}

    RETURN_TYPES = ()
    FUNCTION = "start_worker"
    OUTPUT_NODE = True
    CATEGORY = "Remote CLIP"

    def start_worker(self, CLIP, listen_port, bind_host="0.0.0.0", auth_token=""):
        token = auth_token or os.environ.get("REMOTE_CLIP_TOKEN", "")
        with self._servers_lock:
            previous = self._servers.pop(listen_port, None)
            if previous is not None:
                previous.close()
            self._servers[listen_port] = _Server(CLIP, bind_host, listen_port, token)
        if bind_host == "0.0.0.0" and not token:
            log("WARNING: worker is accessible on all interfaces without authentication")
        log(f"Worker listening on {bind_host}:{listen_port}")
        return {"ui": {"text": [f"Remote CLIP worker on {bind_host}:{listen_port}"]}}


class LoadRemoteCLIP:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "worker_ip": ("STRING", {"default": "127.0.0.1"}),
            "port": ("INT", {"default": DEFAULT_PORT, "min": 1, "max": 65535}),
        }, "optional": {
            "auth_token": ("STRING", {"default": ""}),
            "transport_precision": (["auto", "fp16", "fp32"], {"default": "auto"}),
            "connect_timeout": ("FLOAT", {"default": SOCKET_TIMEOUT, "min": 0.1, "max": 3600.0, "step": 0.1, "tooltip": "Seconds per connection attempt."}),
            "request_attempts": ("INT", {"default": CONNECT_RETRIES, "min": 1, "max": 10, "tooltip": "Total attempts per operation, including the first. Connection failures and safe request retries share this single limit. Generation is never replayed after sending starts."}),
            "connect_backoff": ("FLOAT", {"default": CONNECT_BACKOFF, "min": 0.0, "max": 60.0, "step": 0.1, "tooltip": "Initial retry delay in seconds; doubles after each failure."}),
            "encode_timeout": ("FLOAT", {"default": SOCKET_TIMEOUT, "min": 0.1, "max": 86400.0, "step": 0.1, "tooltip": "Socket timeout for tokenization and encoding; not a total operation deadline."}),
            "generate_timeout": ("FLOAT", {"default": GENERATE_TIMEOUT, "min": 0.1, "max": 86400.0, "step": 0.1, "tooltip": "Generation socket timeout; not a total operation deadline."}),
        }}

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load_remote"
    CATEGORY = "Remote CLIP"

    def load_remote(self, worker_ip, port, auth_token="", transport_precision="auto",
                    connect_timeout=SOCKET_TIMEOUT, request_attempts=CONNECT_RETRIES,
                    connect_backoff=CONNECT_BACKOFF, encode_timeout=SOCKET_TIMEOUT,
                    generate_timeout=GENERATE_TIMEOUT):
        token = auth_token or os.environ.get("REMOTE_CLIP_TOKEN", "")
        return (RemoteCLIPProxy(worker_ip, port, token, transport_precision,
            connect_timeout=connect_timeout, request_attempts=request_attempts,
            connect_backoff=connect_backoff, encode_timeout=encode_timeout,
            generate_timeout=generate_timeout),)

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
