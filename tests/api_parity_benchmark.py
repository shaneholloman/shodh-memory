#!/usr/bin/env python3
"""
API Parity Benchmark: Native Python vs REST API
Tests all API endpoints and compares timing between native and REST implementations.
"""

import json
import os
import sys
import time
import tempfile
import shutil
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

import requests

# Try to import native bindings
try:
    from shodh_memory import Memory
    NATIVE_AVAILABLE = True
except ImportError:
    NATIVE_AVAILABLE = False
    print("WARNING: Native shodh_memory not available. Install with: pip install shodh-memory")

# Configuration
REST_BASE_URL = os.environ.get("SHODH_REST_URL", "http://127.0.0.1:3030")  # IPv4 to avoid Windows IPv6 delay
API_KEY = os.environ.get("SHODH_API_KEY", "sk-shodh-dev-4f8b2c1d9e3a7f5b6d2c8e4a1b9f7d3c")
USER_ID = "benchmark-user"
NUM_ITERATIONS = 5

class BenchmarkResult:
    def __init__(self, name: str):
        self.name = name
        self.native_times: List[float] = []
        self.rest_times: List[float] = []
        self.native_success = True
        self.rest_success = True
        self.native_error: Optional[str] = None
        self.rest_error: Optional[str] = None

    def add_native(self, elapsed: float):
        self.native_times.append(elapsed)

    def add_rest(self, elapsed: float):
        self.rest_times.append(elapsed)

    @property
    def native_avg(self) -> float:
        return sum(self.native_times) / len(self.native_times) if self.native_times else 0

    @property
    def rest_avg(self) -> float:
        return sum(self.rest_times) / len(self.rest_times) if self.rest_times else 0

    @property
    def speedup(self) -> float:
        if self.native_avg == 0:
            return 0
        return self.rest_avg / self.native_avg if self.native_avg > 0 else 0


class RESTClient:
    """Simple REST client for shodh-memory API"""

    def __init__(self, base_url: str, api_key: str, user_id: str):
        self.base_url = base_url
        self.api_key = api_key
        self.user_id = user_id
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "X-API-Key": api_key
        })

    def remember(self, content: str, memory_type: str = "Observation", tags: List[str] = None) -> Dict:
        resp = self.session.post(f"{self.base_url}/api/remember", json={
            "user_id": self.user_id,
            "content": content,
            "memory_type": memory_type,
            "tags": tags or []
        })
        resp.raise_for_status()
        return resp.json()

    def recall(self, query: str, limit: int = 10) -> Dict:
        resp = self.session.post(f"{self.base_url}/api/recall", json={
            "user_id": self.user_id,
            "query": query,
            "limit": limit
        })
        resp.raise_for_status()
        return resp.json()

    def list_memories(self, limit: int = 100) -> Dict:
        resp = self.session.get(f"{self.base_url}/api/list/{self.user_id}", params={"limit": limit})
        resp.raise_for_status()
        return resp.json()

    def context_summary(self, max_items: int = 5) -> Dict:
        resp = self.session.post(f"{self.base_url}/api/context_summary", json={
            "user_id": self.user_id,
            "max_items": max_items,
            "include_decisions": True,
            "include_learnings": True,
            "include_context": True
        })
        resp.raise_for_status()
        return resp.json()

    def brain_state(self) -> Dict:
        resp = self.session.get(f"{self.base_url}/api/brain/{self.user_id}")
        resp.raise_for_status()
        return resp.json()

    def get_stats(self) -> Dict:
        resp = self.session.get(f"{self.base_url}/api/users/{self.user_id}/stats")
        resp.raise_for_status()
        return resp.json()

    def forget_by_tags(self, tags: List[str]) -> Dict:
        resp = self.session.post(f"{self.base_url}/api/forget/tags", json={
            "user_id": self.user_id,
            "tags": tags
        })
        resp.raise_for_status()
        return resp.json()

    def forget_by_age(self, days: int) -> Dict:
        resp = self.session.post(f"{self.base_url}/api/forget/age", json={
            "user_id": self.user_id,
            "days_old": days
        })
        resp.raise_for_status()
        return resp.json()

    def forget_by_importance(self, threshold: float) -> Dict:
        resp = self.session.post(f"{self.base_url}/api/forget/importance", json={
            "user_id": self.user_id,
            "threshold": threshold
        })
        resp.raise_for_status()
        return resp.json()

    def recall_by_tags(self, tags: List[str], limit: int = 20) -> Dict:
        resp = self.session.post(f"{self.base_url}/api/recall/tags", json={
            "user_id": self.user_id,
            "tags": tags,
            "limit": limit
        })
        resp.raise_for_status()
        return resp.json()

    def recall_by_date(self, start: str, end: str, limit: int = 20) -> Dict:
        resp = self.session.post(f"{self.base_url}/api/recall/date", json={
            "user_id": self.user_id,
            "start": start,
            "end": end,
            "limit": limit
        })
        resp.raise_for_status()
        return resp.json()

    def graph_stats(self) -> Dict:
        resp = self.session.get(f"{self.base_url}/api/graph/{self.user_id}/stats")
        resp.raise_for_status()
        return resp.json()

    def health(self) -> Dict:
        resp = self.session.get(f"{self.base_url}/health")
        resp.raise_for_status()
        return resp.json()


def time_operation(func, *args, **kwargs):
    """Time a single operation and return (elapsed_ms, result)"""
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = (time.perf_counter() - start) * 1000  # Convert to ms
    return elapsed, result


def check_rest_server():
    """Check if REST server is available"""
    try:
        resp = requests.get(f"{REST_BASE_URL}/health", timeout=2)
        return resp.status_code == 200
    except:
        return False


def run_benchmarks():
    """Run all benchmarks"""
    results: List[BenchmarkResult] = []

    # Check prerequisites
    rest_available = check_rest_server()

    print("=" * 70)
    print("SHODH-MEMORY API PARITY BENCHMARK")
    print("=" * 70)
    print(f"Native Python available: {NATIVE_AVAILABLE}")
    print(f"REST Server available:   {rest_available} ({REST_BASE_URL})")
    print(f"Iterations per test:     {NUM_ITERATIONS}")
    print("=" * 70)

    if not NATIVE_AVAILABLE and not rest_available:
        print("ERROR: Neither native nor REST API is available!")
        return results

    # Setup
    temp_dir = tempfile.mkdtemp(prefix="shodh_bench_")
    native_mem = None
    rest_client = None

    try:
        if NATIVE_AVAILABLE:
            native_mem = Memory(storage_path=temp_dir)

        if rest_available:
            rest_client = RESTClient(REST_BASE_URL, API_KEY, USER_ID)

        # =========================================================================
        # WARMUP PHASE - Load ONNX models and initialize caches
        # =========================================================================
        print("\n--- WARMUP (model loading, not timed) ---")
        if native_mem:
            native_mem.remember("Warmup memory for model loading", memory_type="Context")
            native_mem.recall("warmup query", limit=1)
        if rest_client:
            rest_client.remember("Warmup memory for REST", memory_type="Context")
            rest_client.recall("warmup query", limit=1)
        print("    Warmup complete.")

        # =========================================================================
        # 1. REMEMBER (Store memory)
        # =========================================================================
        print("\n--- 1. REMEMBER (Store memory) ---")
        result = BenchmarkResult("remember")

        test_contents = [
            "User prefers dark mode for better visibility",
            "API endpoint changed to /v2/auth",
            "Database migration completed successfully",
            "JWT tokens expire after 24 hours",
            "Performance optimization reduced latency by 40%"
        ]

        for i in range(NUM_ITERATIONS):
            content = test_contents[i % len(test_contents)]
            tags = ["test", f"iteration-{i}"]

            if native_mem:
                try:
                    elapsed, _ = time_operation(
                        native_mem.remember,
                        content,
                        memory_type="Decision",
                        tags=tags
                    )
                    result.add_native(elapsed)
                except Exception as e:
                    result.native_success = False
                    result.native_error = str(e)

            if rest_client:
                try:
                    elapsed, _ = time_operation(
                        rest_client.remember,
                        content,
                        memory_type="Decision",
                        tags=tags
                    )
                    result.add_rest(elapsed)
                except Exception as e:
                    result.rest_success = False
                    result.rest_error = str(e)

        results.append(result)

        # =========================================================================
        # 2. RECALL (Semantic search)
        # =========================================================================
        print("--- 2. RECALL (Semantic search) ---")
        result = BenchmarkResult("recall")

        queries = [
            "user preferences",
            "authentication",
            "database changes",
            "token expiration",
            "performance improvements"
        ]

        for i in range(NUM_ITERATIONS):
            query = queries[i % len(queries)]

            if native_mem:
                try:
                    elapsed, _ = time_operation(native_mem.recall, query, limit=5)
                    result.add_native(elapsed)
                except Exception as e:
                    result.native_success = False
                    result.native_error = str(e)

            if rest_client:
                try:
                    elapsed, _ = time_operation(rest_client.recall, query, limit=5)
                    result.add_rest(elapsed)
                except Exception as e:
                    result.rest_success = False
                    result.rest_error = str(e)

        results.append(result)

        # =========================================================================
        # 3. LIST_MEMORIES
        # =========================================================================
        print("--- 3. LIST_MEMORIES ---")
        result = BenchmarkResult("list_memories")

        for i in range(NUM_ITERATIONS):
            if native_mem:
                try:
                    elapsed, _ = time_operation(native_mem.list_memories, limit=50)
                    result.add_native(elapsed)
                except Exception as e:
                    result.native_success = False
                    result.native_error = str(e)

            if rest_client:
                try:
                    elapsed, _ = time_operation(rest_client.list_memories, limit=50)
                    result.add_rest(elapsed)
                except Exception as e:
                    result.rest_success = False
                    result.rest_error = str(e)

        results.append(result)

        # =========================================================================
        # 4. CONTEXT_SUMMARY
        # =========================================================================
        print("--- 4. CONTEXT_SUMMARY ---")
        result = BenchmarkResult("context_summary")

        for i in range(NUM_ITERATIONS):
            if native_mem:
                try:
                    elapsed, _ = time_operation(native_mem.context_summary, max_items=5)
                    result.add_native(elapsed)
                except Exception as e:
                    result.native_success = False
                    result.native_error = str(e)

            if rest_client:
                try:
                    elapsed, _ = time_operation(rest_client.context_summary, max_items=5)
                    result.add_rest(elapsed)
                except Exception as e:
                    result.rest_success = False
                    result.rest_error = str(e)

        results.append(result)

        # =========================================================================
        # 5. BRAIN_STATE
        # =========================================================================
        print("--- 5. BRAIN_STATE ---")
        result = BenchmarkResult("brain_state")

        for i in range(NUM_ITERATIONS):
            if native_mem:
                try:
                    elapsed, _ = time_operation(native_mem.brain_state, longterm_limit=50)
                    result.add_native(elapsed)
                except Exception as e:
                    result.native_success = False
                    result.native_error = str(e)

            if rest_client:
                try:
                    elapsed, _ = time_operation(rest_client.brain_state)
                    result.add_rest(elapsed)
                except Exception as e:
                    result.rest_success = False
                    result.rest_error = str(e)

        results.append(result)

        # =========================================================================
        # 6. GET_STATS
        # =========================================================================
        print("--- 6. GET_STATS ---")
        result = BenchmarkResult("get_stats")

        for i in range(NUM_ITERATIONS):
            if native_mem:
                try:
                    elapsed, _ = time_operation(native_mem.get_stats)
                    result.add_native(elapsed)
                except Exception as e:
                    result.native_success = False
                    result.native_error = str(e)

            if rest_client:
                try:
                    elapsed, _ = time_operation(rest_client.get_stats)
                    result.add_rest(elapsed)
                except Exception as e:
                    result.rest_success = False
                    result.rest_error = str(e)

        results.append(result)

        # =========================================================================
        # 7. FORGET_BY_TAGS
        # =========================================================================
        print("--- 7. FORGET_BY_TAGS ---")
        result = BenchmarkResult("forget_by_tags")

        for i in range(NUM_ITERATIONS):
            # First add a memory with specific tag, then delete it
            tag = f"delete-me-{i}"

            if native_mem:
                try:
                    native_mem.remember(f"Memory to delete {i}", memory_type="Context", tags=[tag])
                    elapsed, _ = time_operation(native_mem.forget_by_tags, [tag])
                    result.add_native(elapsed)
                except Exception as e:
                    result.native_success = False
                    result.native_error = str(e)

            if rest_client:
                try:
                    rest_client.remember(f"Memory to delete {i}", memory_type="Context", tags=[tag])
                    elapsed, _ = time_operation(rest_client.forget_by_tags, [tag])
                    result.add_rest(elapsed)
                except Exception as e:
                    result.rest_success = False
                    result.rest_error = str(e)

        results.append(result)

        # =========================================================================
        # 8. FORGET_BY_IMPORTANCE
        # =========================================================================
        print("--- 8. FORGET_BY_IMPORTANCE ---")
        result = BenchmarkResult("forget_by_importance")

        for i in range(NUM_ITERATIONS):
            if native_mem:
                try:
                    # Use high threshold so nothing gets deleted
                    elapsed, _ = time_operation(native_mem.forget_by_importance, 0.99)
                    result.add_native(elapsed)
                except Exception as e:
                    result.native_success = False
                    result.native_error = str(e)

            if rest_client:
                try:
                    elapsed, _ = time_operation(rest_client.forget_by_importance, 0.99)
                    result.add_rest(elapsed)
                except Exception as e:
                    result.rest_success = False
                    result.rest_error = str(e)

        results.append(result)

        # =========================================================================
        # 9. FORGET_BY_AGE
        # =========================================================================
        print("--- 9. FORGET_BY_AGE ---")
        result = BenchmarkResult("forget_by_age")

        for i in range(NUM_ITERATIONS):
            if native_mem:
                try:
                    # Use 999 days so nothing gets deleted
                    elapsed, _ = time_operation(native_mem.forget_by_age, 999)
                    result.add_native(elapsed)
                except Exception as e:
                    result.native_success = False
                    result.native_error = str(e)

            if rest_client:
                try:
                    elapsed, _ = time_operation(rest_client.forget_by_age, 999)
                    result.add_rest(elapsed)
                except Exception as e:
                    result.rest_success = False
                    result.rest_error = str(e)

        results.append(result)

        # =========================================================================
        # 10. RECALL_BY_TAGS
        # =========================================================================
        print("--- 10. RECALL_BY_TAGS ---")
        result = BenchmarkResult("recall_by_tags")

        for i in range(NUM_ITERATIONS):
            if native_mem:
                try:
                    elapsed, _ = time_operation(native_mem.recall_by_tags, ["test"], limit=10)
                    result.add_native(elapsed)
                except Exception as e:
                    result.native_success = False
                    result.native_error = str(e)

            if rest_client:
                try:
                    elapsed, _ = time_operation(rest_client.recall_by_tags, ["test"], limit=10)
                    result.add_rest(elapsed)
                except Exception as e:
                    result.rest_success = False
                    result.rest_error = str(e)

        results.append(result)

        # =========================================================================
        # 11. RECALL_BY_DATE
        # =========================================================================
        print("--- 11. RECALL_BY_DATE ---")
        result = BenchmarkResult("recall_by_date")

        from datetime import timezone
        start_date = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
        end_date = datetime.now(timezone.utc).isoformat()

        for i in range(NUM_ITERATIONS):
            if native_mem:
                try:
                    elapsed, _ = time_operation(
                        native_mem.recall_by_date,
                        start_date,
                        end_date,
                        limit=10
                    )
                    result.add_native(elapsed)
                except Exception as e:
                    result.native_success = False
                    result.native_error = str(e)

            if rest_client:
                try:
                    elapsed, _ = time_operation(
                        rest_client.recall_by_date,
                        start_date,
                        end_date,
                        limit=10
                    )
                    result.add_rest(elapsed)
                except Exception as e:
                    result.rest_success = False
                    result.rest_error = str(e)

        results.append(result)

        # =========================================================================
        # 12. GRAPH_STATS
        # =========================================================================
        print("--- 12. GRAPH_STATS ---")
        result = BenchmarkResult("graph_stats")

        for i in range(NUM_ITERATIONS):
            if native_mem:
                try:
                    elapsed, _ = time_operation(native_mem.graph_stats)
                    result.add_native(elapsed)
                except Exception as e:
                    result.native_success = False
                    result.native_error = str(e)

            if rest_client:
                try:
                    elapsed, _ = time_operation(rest_client.graph_stats)
                    result.add_rest(elapsed)
                except Exception as e:
                    result.rest_success = False
                    result.rest_error = str(e)

        results.append(result)

    finally:
        # Cleanup
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)

    return results


def print_results(results: List[BenchmarkResult]):
    """Print benchmark results in a nice table"""
    print("\n")
    print("=" * 90)
    print("BENCHMARK RESULTS")
    print("=" * 90)
    print(f"{'Operation':<25} {'Native (ms)':<15} {'REST (ms)':<15} {'Speedup':<12} {'Status'}")
    print("-" * 90)

    for r in results:
        native_str = f"{r.native_avg:.2f}" if r.native_times else "N/A"
        rest_str = f"{r.rest_avg:.2f}" if r.rest_times else "N/A"

        if r.native_avg > 0 and r.rest_avg > 0:
            speedup_str = f"{r.speedup:.2f}x"
        else:
            speedup_str = "N/A"

        status = ""
        if not r.native_success:
            status += f"[Native FAIL: {r.native_error[:30]}] "
        if not r.rest_success:
            status += f"[REST FAIL: {r.rest_error[:30]}] "
        if not status:
            status = "OK"

        print(f"{r.name:<25} {native_str:<15} {rest_str:<15} {speedup_str:<12} {status}")

    print("-" * 90)

    # Summary
    total_native = sum(r.native_avg for r in results if r.native_times)
    total_rest = sum(r.rest_avg for r in results if r.rest_times)

    if total_native > 0 and total_rest > 0:
        print(f"{'TOTAL':<25} {total_native:.2f} ms        {total_rest:.2f} ms        {total_rest/total_native:.2f}x")

    print("=" * 90)
    print("\nNotes:")
    print("- Native Python: Direct bindings without network overhead")
    print("- REST: HTTP API with JSON serialization overhead")
    print("- Speedup: REST time / Native time (higher = native is faster)")
    print("=" * 90)


def main():
    results = run_benchmarks()
    print_results(results)

    # Return exit code based on success
    all_success = all(r.native_success and r.rest_success for r in results)
    sys.exit(0 if all_success else 1)


if __name__ == "__main__":
    main()
