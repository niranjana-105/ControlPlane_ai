"""
ControlPlane.ai - Quick API Test Client
Demonstrates sending requests to the running FastAPI Governance Proxy (/v1/chat/completions).
"""

import requests
import json

GATEWAY_URL = "http://localhost:8000/v1/chat/completions"

def test_api_scenario(scenario="composite", prompt="Please process this transaction."):
    print(f"\n==========================================")
    print(f"Testing Scenario: {scenario.upper()}")
    print(f"Prompt: {prompt}")
    print(f"==========================================")

    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "scenario": scenario,
        "policy_profile": "Customer Support Bot"
    }

    try:
        response = requests.post(GATEWAY_URL, json=payload, stream=True, timeout=10)
        print(f"HTTP Status: {response.status_code}")
        print("Streaming Governed Response Tokens:")
        
        for line in response.iter_lines():
            if line:
                decoded = line.decode('utf-8')
                if decoded.startswith("data: "):
                    data_str = decoded[6:]
                    if data_str == "[DONE]":
                        print("\n--- Stream Complete [DONE] ---")
                        break
                    try:
                        chunk = json.loads(data_str)
                        if "choices" in chunk and len(chunk["choices"]) > 0:
                            content = chunk["choices"][0].get("delta", {}).get("content", "")
                            print(content, end="", flush=True)
                        elif "event" in chunk:
                            print(f"\n[Event: {chunk['event']}] => {chunk.get('data')}")
                    except Exception:
                        pass
        print()
    except requests.exceptions.ConnectionError:
        print("[!] Gateway not running on http://localhost:8000. Start it with `python run.py` first!")

if __name__ == "__main__":
    print("Testing ControlPlane.ai Gateway API...")
    test_api_scenario("pii_leak", "Show customer details")
    test_api_scenario("bias", "Evaluate leadership candidates")
    test_api_scenario("clean", "How do I reset my password?")
