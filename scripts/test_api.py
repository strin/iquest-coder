#!/usr/bin/env python3
"""
Test script for the IQuest-Coder vLLM API endpoint.

This script tests the OpenAI-compatible API served by vLLM to verify
the endpoint is working correctly and can handle various request types.

Usage:
    python test_api.py                          # Uses default localhost:8000
    python test_api.py --base-url http://host:port/v1
    python test_api.py --quick                  # Quick health check only
"""

import argparse
import json
import sys
import time
from typing import Optional

try:
    import requests
except ImportError:
    print("❌ 'requests' library not found. Install with: pip install requests")
    sys.exit(1)

# Default configuration
DEFAULT_BASE_URL = "http://localhost:8000/v1"
DEFAULT_MODEL = "IQuestLab/IQuest-Coder-V1-40B-Instruct"


def test_health(base_url: str) -> bool:
    """Test if the server is responding."""
    print("🏥 Testing server health...")
    
    # vLLM health endpoint (non-standard, try root)
    health_url = base_url.replace("/v1", "/health")
    
    try:
        response = requests.get(health_url, timeout=10)
        if response.status_code == 200:
            print(f"   ✅ Health endpoint OK")
            return True
    except requests.exceptions.RequestException:
        pass
    
    # Fallback: try models endpoint
    try:
        response = requests.get(f"{base_url}/models", timeout=10)
        if response.status_code == 200:
            print(f"   ✅ Server responding (models endpoint)")
            return True
        else:
            print(f"   ❌ Server returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"   ❌ Connection refused. Is the server running?")
        return False
    except requests.exceptions.Timeout:
        print(f"   ❌ Connection timed out")
        return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def test_list_models(base_url: str) -> Optional[str]:
    """Test the /models endpoint and return the first available model."""
    print("\n📋 Testing /models endpoint...")
    
    try:
        response = requests.get(f"{base_url}/models", timeout=30)
        response.raise_for_status()
        
        data = response.json()
        models = data.get("data", [])
        
        if models:
            print(f"   ✅ Found {len(models)} model(s):")
            for model in models:
                model_id = model.get("id", "unknown")
                print(f"      - {model_id}")
            return models[0].get("id")
        else:
            print("   ⚠️  No models found")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"   ❌ Request failed: {e}")
        return None


def test_chat_completion(base_url: str, model: str, stream: bool = False) -> bool:
    """Test the /chat/completions endpoint."""
    mode = "streaming" if stream else "non-streaming"
    print(f"\n💬 Testing /chat/completions ({mode})...")
    
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": "Say 'Hello, IQuest!' and nothing else."}
        ],
        "temperature": 0.1,
        "max_tokens": 50,
        "stream": stream,
    }
    
    try:
        start_time = time.time()
        
        if stream:
            response = requests.post(
                f"{base_url}/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json"},
                stream=True,
                timeout=120,
            )
            response.raise_for_status()
            
            print("   Response (streamed):")
            full_content = ""
            for line in response.iter_lines():
                if line:
                    line_str = line.decode("utf-8")
                    if line_str.startswith("data: "):
                        data_str = line_str[6:]
                        if data_str.strip() == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data_str)
                            delta = chunk.get("choices", [{}])[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                full_content += content
                                print(content, end="", flush=True)
                        except json.JSONDecodeError:
                            pass
            print()  # Newline after streaming
            
            elapsed = time.time() - start_time
            print(f"   ✅ Streaming completed in {elapsed:.2f}s")
            return True
            
        else:
            response = requests.post(
                f"{base_url}/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=120,
            )
            response.raise_for_status()
            
            elapsed = time.time() - start_time
            data = response.json()
            
            # Extract response content
            choices = data.get("choices", [])
            if choices:
                message = choices[0].get("message", {})
                content = message.get("content", "")
                print(f"   Response: {content}")
                
                # Show usage stats
                usage = data.get("usage", {})
                if usage:
                    print(f"   Usage: {usage.get('prompt_tokens', '?')} prompt + "
                          f"{usage.get('completion_tokens', '?')} completion = "
                          f"{usage.get('total_tokens', '?')} total tokens")
            
            print(f"   ✅ Completed in {elapsed:.2f}s")
            return True
            
    except requests.exceptions.Timeout:
        print(f"   ❌ Request timed out (>120s)")
        return False
    except requests.exceptions.RequestException as e:
        print(f"   ❌ Request failed: {e}")
        return False


def test_completion(base_url: str, model: str) -> bool:
    """Test the /completions endpoint (legacy)."""
    print("\n📝 Testing /completions endpoint (legacy)...")
    
    payload = {
        "model": model,
        "prompt": "def hello_world():\n    ",
        "temperature": 0.1,
        "max_tokens": 30,
        "stop": ["\n\n"],
    }
    
    try:
        start_time = time.time()
        response = requests.post(
            f"{base_url}/completions",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=120,
        )
        response.raise_for_status()
        
        elapsed = time.time() - start_time
        data = response.json()
        
        choices = data.get("choices", [])
        if choices:
            text = choices[0].get("text", "")
            print(f"   Prompt: def hello_world():\\n    ")
            print(f"   Completion: {text.strip()}")
        
        print(f"   ✅ Completed in {elapsed:.2f}s")
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"   ❌ Request failed: {e}")
        return False


def test_code_generation(base_url: str, model: str) -> bool:
    """Test code generation capability."""
    print("\n🧑‍💻 Testing code generation...")
    
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": "Write a Python function that calculates the factorial of a number. Only output the code, no explanations."
            }
        ],
        "temperature": 0.1,
        "max_tokens": 200,
    }
    
    try:
        start_time = time.time()
        response = requests.post(
            f"{base_url}/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=120,
        )
        response.raise_for_status()
        
        elapsed = time.time() - start_time
        data = response.json()
        
        choices = data.get("choices", [])
        if choices:
            content = choices[0].get("message", {}).get("content", "")
            print(f"   Generated code:")
            for line in content.split("\n")[:15]:  # Show first 15 lines
                print(f"   | {line}")
            if content.count("\n") > 15:
                print(f"   | ... ({content.count(chr(10)) - 15} more lines)")
        
        print(f"   ✅ Completed in {elapsed:.2f}s")
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"   ❌ Request failed: {e}")
        return False


def run_tests(base_url: str, model: Optional[str] = None, quick: bool = False) -> int:
    """Run all tests and return exit code."""
    print("=" * 60)
    print("  IQuest-Coder API Test Suite")
    print("=" * 60)
    print(f"\n  Base URL: {base_url}")
    print()
    
    results = {}
    
    # Test 1: Health check
    results["health"] = test_health(base_url)
    if not results["health"]:
        print("\n❌ Server not responding. Aborting tests.")
        return 1
    
    # Test 2: List models
    detected_model = test_list_models(base_url)
    results["models"] = detected_model is not None
    
    # Use detected model or fallback to provided/default
    test_model = model or detected_model or DEFAULT_MODEL
    print(f"\n  Using model: {test_model}")
    
    if quick:
        print("\n✅ Quick health check passed!")
        return 0
    
    # Test 3: Chat completion (non-streaming)
    results["chat"] = test_chat_completion(base_url, test_model, stream=False)
    
    # Test 4: Chat completion (streaming)
    results["chat_stream"] = test_chat_completion(base_url, test_model, stream=True)
    
    # Test 5: Legacy completions
    results["completions"] = test_completion(base_url, test_model)
    
    # Test 6: Code generation
    results["code_gen"] = test_code_generation(base_url, test_model)
    
    # Summary
    print("\n" + "=" * 60)
    print("  Test Summary")
    print("=" * 60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, passed_test in results.items():
        status = "✅ PASS" if passed_test else "❌ FAIL"
        print(f"  {test_name:20} {status}")
    
    print()
    print(f"  Results: {passed}/{total} tests passed")
    print("=" * 60)
    
    return 0 if passed == total else 1


def main():
    parser = argparse.ArgumentParser(
        description="Test IQuest-Coder vLLM API endpoint",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with default settings (localhost:8000)
  python test_api.py
  
  # Test a specific endpoint
  python test_api.py --base-url http://10.0.0.1:8000/v1
  
  # Quick health check only
  python test_api.py --quick
  
  # Specify model explicitly
  python test_api.py --model IQuestLab/IQuest-Coder-V1-40B-Instruct
"""
    )
    
    parser.add_argument(
        "--base-url", "-b",
        default=DEFAULT_BASE_URL,
        help=f"Base URL for the API (default: {DEFAULT_BASE_URL})"
    )
    parser.add_argument(
        "--model", "-m",
        default=None,
        help="Model name to use for tests (auto-detected if not specified)"
    )
    parser.add_argument(
        "--quick", "-q",
        action="store_true",
        help="Quick mode: only test health and list models"
    )
    
    args = parser.parse_args()
    
    # Ensure base_url ends with /v1
    base_url = args.base_url.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url += "/v1"
    
    return run_tests(base_url, args.model, args.quick)


if __name__ == "__main__":
    sys.exit(main())
