
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests


URL = "http://127.0.0.1:8082/invocations"
PAYLOAD_PATH = Path("tests/payloads/lk10_prediction.json")


def send_request(payload, url):
    start = time.perf_counter()

    try:
        response = requests.post(
            url,
            json=payload,
            timeout=30,
        )

        latency_ms = (time.perf_counter() - start) * 1000

        return {
            "success": response.status_code == 200,
            "latency_ms": latency_ms,
            "status_code": response.status_code,
        }

    except requests.RequestException:
        return {
            "success": False,
            "latency_ms": (time.perf_counter() - start) * 1000,
            "status_code": None,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--url",
        type=str,
        default=URL,
    )
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--requests", type=int, default=60)
    args = parser.parse_args()

    if args.concurrency < 1 or args.requests < 1:
        parser.error("concurrency and requests must be positive")

    payload = json.loads(PAYLOAD_PATH.read_text(encoding="utf-8"))

    start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [
            executor.submit(send_request, payload, args.url)
            for _ in range(args.requests)
        ]
        results = [future.result() for future in as_completed(futures)]

    duration = time.perf_counter() - start

    successful = [r for r in results if r["success"]]
    latencies = sorted(r["latency_ms"] for r in results)

    def percentile(values, percentile):
        if not values:
            return None
        index = int((len(values) - 1) * percentile)
        return values[index]

    print(f"Concurrency: {args.concurrency}")
    print(f"Total requests: {args.requests}")
    print(f"Success: {len(successful)}")
    print(f"Failed: {len(results) - len(successful)}")
    print(f"Duration: {duration:.3f} seconds")
    print(f"Throughput: {len(successful) / duration:.3f} req/s")
    print(f"Mean latency: {sum(latencies) / len(latencies):.2f} ms")
    print(f"P95 latency: {percentile(latencies, 0.95):.2f} ms")


if __name__ == "__main__":
    main()
