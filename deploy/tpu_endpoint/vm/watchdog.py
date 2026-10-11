"""Self-delete the TPU VM when idle or too old. Runs on the VM; uses the VM service account (cloud-platform scope).

Idle = the vLLM request counters have not changed for idle_min minutes AFTER the server first became healthy
(load/compile time is never counted as idle, but max_up_min bounds the whole lifetime and is the cost safety net).
"""
import json
import sys
import time
import urllib.error
import urllib.request

MD = "http://metadata.google.internal/computeMetadata/v1/"
WATCHED = ("vllm:request_success_total", "vllm:num_requests_running", "vllm:prompt_tokens_total")


def md(path):
    r = urllib.request.Request(MD + path, headers={"Metadata-Flavor": "Google"})
    return urllib.request.urlopen(r, timeout=5).read().decode()


def counters(port, key):
    r = urllib.request.Request(f"http://127.0.0.1:{port}/metrics", headers={"Authorization": f"Bearer {key}"})
    txt = urllib.request.urlopen(r, timeout=5).read().decode()
    total = 0.0
    for ln in txt.splitlines():
        if ln.startswith(WATCHED):
            try:
                total += float(ln.rsplit(" ", 1)[1])
            except ValueError:
                pass
    return total


def delete_self(project, node):
    zone = md("instance/zone").rsplit("/", 1)[-1]
    tok = json.loads(md("instance/service-accounts/default/token"))["access_token"]
    url = f"https://tpu.googleapis.com/v2/projects/{project}/locations/{zone}/nodes/{node}"
    req = urllib.request.Request(url, method="DELETE", headers={"Authorization": f"Bearer {tok}"})
    print("deleting", url, flush=True)
    urllib.request.urlopen(req, timeout=30).read()


def decide(now, t0, max_up_s, idle_s, last_change):
    """'max_up' | 'idle' | None. last_change is None until the server has been healthy once."""
    if now - t0 > max_up_s:
        return "max_up"
    if last_change is not None and now - last_change > idle_s:
        return "idle"
    return None


def main(argv):
    port, key, idle_min, max_up_min, project, node = argv[:6]
    idle_s, max_up_s = float(idle_min) * 60, float(max_up_min) * 60
    t0 = time.time()
    last_val, last_change = None, None
    while True:
        now = time.time()
        try:
            v = counters(port, key)
            if last_change is None or v != last_val:
                last_val, last_change = v, now
        except (urllib.error.URLError, OSError, ValueError):
            pass  # server not up yet (or restarting); max_up_s still bounds the lifetime
        why = decide(now, t0, max_up_s, idle_s, last_change)
        if why:
            print(why, flush=True)
            break
        time.sleep(20)
    try:
        delete_self(project, node)
    except Exception as exc:  # noqa: BLE001
        print("delete failed:", exc, flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
