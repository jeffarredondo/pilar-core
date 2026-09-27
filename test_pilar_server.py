#!/usr/bin/env python3
"""
Full test client for pilar-server. Speaks raw MCP JSON-RPC over
Streamable HTTP directly -- no MCP client library, no OpenClaw, no
middleman. Requires only `requests` (pip install requests).

Exercises all three tools:
    - list_shards  (registry read)
    - save_concept (scratch write -- verified on disk, not just by the
                     tool's own success message, if --km-dir is given)
    - query        (full RAG round trip -- needs Ollama actually running
                     with the configured models pulled)

Usage:
    python3 test_pilar_server.py
    python3 test_pilar_server.py --url http://127.0.0.1:8090/mcp --km-dir ./km_output
    python3 test_pilar_server.py --question "What was SpaceX's Q1 2026 revenue?"
    python3 test_pilar_server.py --skip-query          # skip the Ollama-dependent test
    python3 test_pilar_server.py --skip-save

Exit code 0 if every test passes, 1 otherwise -- scriptable.
"""

import argparse
import json
import sys
import time
import uuid

import requests

HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json, text/event-stream",
}

_next_id = 0


def rpc(base_url: str, method: str, params: dict, timeout: float = 30) -> dict:
    global _next_id
    _next_id += 1
    body = {"jsonrpc": "2.0", "id": _next_id, "method": method, "params": params}

    t0 = time.monotonic()
    resp = requests.post(base_url, headers=HEADERS, json=body, timeout=timeout)
    elapsed = time.monotonic() - t0

    content_type = resp.headers.get("content-type", "")
    if not content_type.startswith("application/json"):
        raise AssertionError(
            f"expected content-type application/json, got '{content_type}' "
            f"(HTTP {resp.status_code}), body: {resp.text[:500]}"
        )
    if resp.status_code != 200:
        raise AssertionError(f"HTTP {resp.status_code}: {resp.text[:500]}")

    print(f"    ({elapsed:.3f}s)")
    return resp.json()


def call_tool(base_url: str, name: str, arguments: dict, timeout: float = 30) -> dict:
    """Calls a tool and returns its parsed inner JSON payload. Raises if
    the JSON-RPC call itself errored, or if the tool reported a soft
    (isError:true) failure -- callers that expect a soft failure should
    catch AssertionError and check .args instead."""
    resp = rpc(base_url, "tools/call", {"name": name, "arguments": arguments}, timeout=timeout)
    if "error" in resp:
        raise AssertionError(f"{name}: JSON-RPC error: {resp['error']}")

    result = resp["result"]
    text = result["content"][0]["text"]
    if result.get("isError"):
        raise ToolError(text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"_raw_text": text}


class ToolError(Exception):
    pass


class TestRunner:
    def __init__(self):
        self.passed = 0
        self.failed = 0

    def run(self, name: str, fn):
        print(f"[{self.passed + self.failed + 1}] {name}")
        try:
            fn()
            print("    PASS\n")
            self.passed += 1
        except Exception as e:
            print(f"    FAIL: {e}\n")
            self.failed += 1

    def summary(self) -> int:
        total = self.passed + self.failed
        print(f"{'=' * 60}")
        print(f"{self.passed}/{total} passed")
        return 0 if self.failed == 0 else 1


def test_initialize(base_url: str):
    resp = rpc(base_url, "initialize", {
        "protocolVersion": "2025-11-25",
        "capabilities": {},
        "clientInfo": {"name": "pilar-test-client", "version": "1"},
    })
    assert "error" not in resp, resp.get("error")
    server_info = resp["result"]["serverInfo"]
    print(f"    server: {server_info['name']} {server_info['version']}")


def test_tools_list(base_url: str):
    resp = rpc(base_url, "tools/list", {})
    assert "error" not in resp, resp.get("error")
    names = {t["name"] for t in resp["result"]["tools"]}
    expected = {"list_shards", "query", "save_concept"}
    assert names == expected, f"expected exactly {expected}, got {names}"
    print(f"    tools: {sorted(names)}")


def test_list_shards(base_url: str) -> dict:
    result = call_tool(base_url, "list_shards", {})
    assert "shard_count" in result, f"missing shard_count in {result}"
    assert result["shard_count"] == len(result["shards"]), "shard_count doesn't match len(shards)"
    print(f"    shard_count: {result['shard_count']}")
    print(f"    first 5 shard ids: {[s['shard_id'] for s in result['shards'][:5]]}")
    return result


def test_save_concept(base_url: str, km_dir: str | None):
    marker = f"pilar-test-marker-{uuid.uuid4().hex[:8]}"
    text = f"Automated test note ({marker}): this line was written by test_pilar_server.py."
    result = call_tool(base_url, "save_concept", {"text": text, "tag": "automated-test"})
    print(f"    tool response: {result}")

    if km_dir is None:
        print("    (no --km-dir given -- trusting the tool's own success response, not verifying the file on disk)")
        return

    import pathlib
    scratch_path = pathlib.Path(km_dir) / "scratch.jsonl"
    assert scratch_path.exists(), f"expected {scratch_path} to exist after save_concept"
    lines = scratch_path.read_text().splitlines()
    matching = [json.loads(line) for line in lines if marker in line]
    assert len(matching) == 1, f"expected exactly 1 matching line in {scratch_path}, found {len(matching)}"
    record = matching[0]
    assert record["source"] == "scratch", f"expected source=scratch, got {record['source']}"
    assert record["tag"] == "automated-test"
    print(f"    verified on disk at {scratch_path}: {record}")


def test_query(base_url: str, question: str):
    result = call_tool(base_url, "query", {"question": question, "top_k": 5}, timeout=90)
    assert "answer" in result, f"missing 'answer' in {result}"
    assert "concepts" in result, f"missing 'concepts' in {result}"
    print(f"    question: {question}")
    print(f"    concepts used: {len(result['concepts'])}")
    for c in result["concepts"][:3]:
        print(f"      [{c['distance']:.4f}] {c['raw_term']} -> \"{c['label']}\"")
    print(f"    answer: {result['answer'][:300]}{'...' if len(result['answer']) > 300 else ''}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://127.0.0.1:8090/mcp")
    parser.add_argument("--km-dir", default=None, help="local km_output path, to verify save_concept's write on disk")
    parser.add_argument("--question", default="What was SpaceX's Q1 2026 revenue?", help="question to send to the query tool")
    parser.add_argument("--skip-query", action="store_true", help="skip the query test (needs live Ollama)")
    parser.add_argument("--skip-save", action="store_true", help="skip the save_concept test")
    args = parser.parse_args()

    print(f"Testing pilar-server at {args.url}\n")
    runner = TestRunner()

    runner.run("initialize", lambda: test_initialize(args.url))
    runner.run("tools/list exposes exactly the 3 expected tools", lambda: test_tools_list(args.url))
    runner.run("list_shards returns real registry data", lambda: test_list_shards(args.url))

    if not args.skip_save:
        runner.run("save_concept writes a real, verifiable record", lambda: test_save_concept(args.url, args.km_dir))

    if not args.skip_query:
        runner.run("query performs a real RAG round trip via Ollama", lambda: test_query(args.url, args.question))

    sys.exit(runner.summary())


if __name__ == "__main__":
    main()