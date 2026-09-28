"""Inference backends used INSIDE the worker process only. The parent never imports llama_cpp (so it never holds a CUDA context)."""
from __future__ import annotations
import json, os, re, shutil, socket, subprocess, time


class BaseRunner:
    def generate(self, messages, max_tokens=512, temperature=0.2, stop=None, json_mode=False, on_token=None, reset=False) -> dict:
        raise NotImplementedError

    def close(self):
        pass


# ---------------------------------------------------------------- llama-cpp-python
class LlamaCppRunner(BaseRunner):
    def __init__(self, spec):
        from llama_cpp import Llama  # noqa
        kw = dict(model_path=spec["path"], n_ctx=spec["n_ctx"], n_gpu_layers=spec["n_gpu_layers"],
                  n_batch=int(spec.get("n_batch", 512)), n_threads=os.cpu_count() or 4,
                  flash_attn=bool(spec.get("flash_attn", False)), use_mmap=True, verbose=True)
        if spec.get("kv_cache_type") == "q8_0":
            kw.update(type_k=8, type_v=8)  # GGML_TYPE_Q8_0 (requires flash_attn)
        if spec.get("chat_format"):
            kw["chat_format"] = spec["chat_format"]
        self.llm = Llama(**kw)

    def generate(self, messages, max_tokens=512, temperature=0.2, stop=None, json_mode=False, on_token=None, reset=False):
        if reset:
            self.llm.reset()
        kw = dict(messages=messages, max_tokens=max_tokens, temperature=temperature, stop=stop or None, stream=True)
        if json_mode:
            kw["response_format"] = {"type": "json_object"}
        t0 = time.perf_counter()
        first = None
        n_out = 0
        parts = []
        for ch in self.llm.create_chat_completion(**kw):
            piece = ch["choices"][0].get("delta", {}).get("content")
            if piece:
                if first is None:
                    first = time.perf_counter()
                n_out += 1  # one streamed chunk ~ one token
                parts.append(piece)
                if on_token:
                    on_token(piece)
        t1 = time.perf_counter()
        first = first or t1
        # n_tokens = tokens now in the KV sequence (prompt + generated); prompt includes any reused prefix
        prompt_tokens = max(0, int(getattr(self.llm, "n_tokens", 0)) - n_out)
        return {"output": "".join(parts), "prompt_tokens": prompt_tokens, "output_tokens": n_out,
                "prompt_ms": int((first - t0) * 1000), "gen_ms": int((t1 - first) * 1000)}


# ---------------------------------------------------------------- llama-server (child of the worker)
class LlamaServerRunner(BaseRunner):
    def __init__(self, spec):
        import requests
        self.requests = requests
        binp = spec.get("llama_server_bin")
        if not binp or not os.path.exists(binp):
            raise RuntimeError("llama-server binary not found (set llama_server_bin / LLAMA_SERVER_BIN or run bootstrap --build-runtime)")
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        self.port = s.getsockname()[1]
        s.close()
        cmd = [binp, "-m", spec["path"], "-c", str(spec["n_ctx"]), "-ngl", str(spec["n_gpu_layers"] if spec["n_gpu_layers"] >= 0 else 999),
               "-b", str(spec.get("n_batch", 512)), "--host", "127.0.0.1", "--port", str(self.port), "--no-webui"]
        if spec.get("kv_cache_type") == "q8_0":
            cmd += ["--cache-type-k", "q8_0", "--cache-type-v", "q8_0"]
        cmd += [str(a) for a in spec.get("extra_args", [])]
        self.proc = subprocess.Popen(cmd, stdout=2, stderr=2)  # its logs go to the worker's stderr log
        deadline = time.time() + float(spec.get("load_timeout_s", 900))
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"llama-server exited during load (rc={self.proc.returncode})")
            try:
                if requests.get(f"http://127.0.0.1:{self.port}/health", timeout=2).status_code == 200:
                    return
            except Exception:
                pass
            time.sleep(1.0)
        raise RuntimeError("llama-server load timeout")

    def generate(self, messages, max_tokens=512, temperature=0.2, stop=None, json_mode=False, on_token=None, reset=False):
        body = {"messages": messages, "max_tokens": max_tokens, "temperature": temperature, "stream": True,
                "stream_options": {"include_usage": True}}
        if stop:
            body["stop"] = stop
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        if reset:
            body["cache_prompt"] = False
        t0 = time.perf_counter()
        first = None
        parts, n_out, timings, usage = [], 0, {}, {}
        with self.requests.post(f"http://127.0.0.1:{self.port}/v1/chat/completions", json=body, stream=True, timeout=(10, None)) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line or not line.startswith(b"data: "):
                    continue
                data = line[6:]
                if data == b"[DONE]":
                    break
                j = json.loads(data)
                timings = j.get("timings", timings)
                usage = j.get("usage", usage)
                for c in j.get("choices", []):
                    piece = (c.get("delta") or {}).get("content")
                    if piece:
                        first = first or time.perf_counter()
                        n_out += 1
                        parts.append(piece)
                        if on_token:
                            on_token(piece)
        t1 = time.perf_counter()
        first = first or t1
        out = {"output": "".join(parts), "prompt_tokens": usage.get("prompt_tokens", 0), "output_tokens": usage.get("completion_tokens", n_out),
               "prompt_ms": int((first - t0) * 1000), "gen_ms": int((t1 - first) * 1000)}
        if timings:  # exact numbers reported by llama-server override wall-clock estimates
            out["prompt_ms"] = int(timings.get("prompt_ms", out["prompt_ms"]))
            out["gen_ms"] = int(timings.get("predicted_ms", out["gen_ms"]))
            out["prompt_tokens"] = timings.get("prompt_n", out["prompt_tokens"])
            out["output_tokens"] = timings.get("predicted_n", out["output_tokens"])
        return out

    def close(self):
        try:
            self.proc.terminate()
            self.proc.wait(timeout=10)
        except Exception:
            self.proc.kill()


# ---------------------------------------------------------------- mock (tests / CI only)
class MockRunner(BaseRunner):
    """Deterministic fake model so the scheduler/lifecycle can be tested without a GPU. NEVER used unless backend: mock."""
    def __init__(self, spec):
        self.role = spec.get("role", "")
        self.key = spec["key"]
        time.sleep(float(os.environ.get("ASTRA_MOCK_LOAD_S", "0.05")))

    def generate(self, messages, max_tokens=512, temperature=0.2, stop=None, json_mode=False, on_token=None, reset=False):
        last = messages[-1]["content"]
        sysm = messages[0]["content"] if messages[0]["role"] == "system" else ""
        if os.environ.get("ASTRA_MOCK_OOM") == self.key:
            raise MemoryError("CUDA error: out of memory (mock)")
        if os.environ.get("ASTRA_MOCK_HANG") == self.key:
            time.sleep(3600)
        if self.key == "orchestrator" and "ASTRA-ORCHESTRATOR" in sysm:
            steps = [{"id": "step_1", "agent": "engineer", "capability": "coding", "action": "inspect_repository", "description": "inspect", "depends_on": []}]
            if "math" in last.lower():
                steps.append({"id": "step_2", "agent": "logic", "capability": "mathematics", "action": "analyze", "description": "analyze", "depends_on": ["step_1"]})
            steps.append({"id": f"step_{len(steps)+1}", "agent": "assistant", "capability": "summarization", "action": "summarize",
                          "description": "summarize", "depends_on": [steps[-1]["id"]]})
            out = json.dumps({"goal": "mock goal", "steps": steps})
        elif self.key == "engineer":
            if os.environ.get("ASTRA_MOCK_ASK") == "engineer" and "<tool_response>" not in last:
                out = '<tool_call>{"name":"ask_user","arguments":{"question":"A or B?"}}</tool_call>'
            elif "<tool_response>" in last:
                out = json.dumps({"status": "success", "changes": [], "tests": [], "notes": ["mock engineer done"]})
            else:
                out = '<tool_call>\n{"name": "list_dir", "arguments": {"path": "."}}\n</tool_call>' 
        elif self.key == "logic":
            out = json.dumps({"answer": "42", "algorithm": "mock", "complexity": "O(1)", "risks": []})
        elif self.key == "assistant":
            out = json.dumps({"response": "mock: all steps complete"})
        else:
            out = "{}"
        if on_token:
            on_token(out)
        n = max(1, len(out) // 4)
        return {"output": out, "prompt_tokens": max(1, len(last) // 4), "output_tokens": n, "prompt_ms": 5, "gen_ms": 10 * n}


def make_runner(spec) -> BaseRunner:
    b = spec["backend"]
    if b == "mock":
        return MockRunner(spec)
    if b == "llama_server":
        return LlamaServerRunner(spec)
    return LlamaCppRunner(spec)
