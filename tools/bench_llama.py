"""llama-server throughput at several slot counts.

    python tools/bench_llama.py [slots] [ctx_per_slot] [conc,conc,...]
"""
import json, os, subprocess, sys, threading, time, urllib.request

OLLAMA_DIR = os.path.join(os.environ["LOCALAPPDATA"], "Programs", "Ollama")
LIB = os.path.join(OLLAMA_DIR, "lib", "ollama")
EXE = os.path.join(LIB, "llama-server.exe")
MODELS = os.path.expanduser("~/.ollama/models")
PORT = 11511
URL = "http://127.0.0.1:%d" % PORT


def gguf(tag="qwen3.5", ver="4b"):
    mf = os.path.join(MODELS, "manifests", "registry.ollama.ai", "library", tag, ver)
    for l in json.load(open(mf))["layers"]:
        if l["mediaType"].endswith("image.model"):
            return os.path.join(MODELS, "blobs", l["digest"].replace(":", "-"))


def alive():
    try:
        return urllib.request.urlopen(URL + "/health", timeout=2).status == 200
    except Exception:
        return False


def launch(slots, ctx, backend):
    env = dict(os.environ)
    env["PATH"] = LIB + os.pathsep + os.path.dirname(backend) + os.pathsep + env["PATH"]
    env["GGML_BACKEND_PATH"] = backend
    args = [EXE, "--model", gguf(), "--host", "127.0.0.1", "--port", str(PORT), "--no-webui",
            "-c", str(slots * ctx), "-np", str(slots), "-ngl", "999", "--flash-attn", "on",
            "--cache-type-k", "q8_0", "--cache-type-v", "q8_0", "-b", "1024", "-ub", "512"]
    log = open(os.path.join(os.path.dirname(__file__), "bench_llama.log"), "ab")
    p = subprocess.Popen(args, env=env, stdout=log, stderr=log,
                         creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
    for _ in range(180):
        if alive():
            return p
        if p.poll() is not None:
            return None
        time.sleep(1)
    subprocess.run(["taskkill", "/F", "/T", "/PID", str(p.pid)], capture_output=True)
    return None


PROMPT = ("<|im_start|>user\nWrite a realistic customer support ticket (about 120 words) from a "
          "user whose online order arrived damaged. Variant %d.<|im_end|>\n<|im_start|>assistant\n"
          "<think>\n\n</think>\n\n")


def one(i, out):
    body = {"prompt": PROMPT % i, "n_predict": 160, "temperature": 0.8, "seed": i,
            "cache_prompt": False, "stop": ["<|im_end|>"]}
    req = urllib.request.Request(URL + "/completion", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.loads(r.read())
    out.append(d.get("tokens_predicted", 0))


def main():
    slots = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    ctx = int(sys.argv[2]) if len(sys.argv) > 2 else 2048
    concs = [int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "1,4,8").split(",")]
    for sub in ("cuda_v13", "cuda_v12"):
        backend = os.path.join(LIB, sub, "ggml-cuda.dll")
        p = launch(slots, ctx, backend)
        if p:
            print("backend", sub, "slots", slots, "ctx/slot", ctx, flush=True)
            break
        print("backend", sub, "failed to start", flush=True)
    else:
        sys.exit(1)
    try:
        smi = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total",
                              "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
        print("VRAM after load:", smi, flush=True)
        one(0, [])
        for conc in concs:
            out = []
            t0 = time.time()
            th = [threading.Thread(target=one, args=(k + 1, out)) for k in range(conc)]
            [t.start() for t in th]
            [t.join() for t in th]
            wall = time.time() - t0
            print("conc=%-3d tokens=%-5d wall=%5.1fs  aggregate=%6.1f tok/s"
                  % (conc, sum(out), wall, sum(out) / wall), flush=True)
    finally:
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(p.pid)], capture_output=True)


if __name__ == "__main__":
    main()
