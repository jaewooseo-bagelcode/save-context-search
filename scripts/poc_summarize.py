#!/usr/bin/env python3
"""
PoC: Function summarization using Gemini 2.5 Flash-Lite API via AI Proxy.

Usage:
    export AI_PROXY_BASE_URL=https://your-proxy-url
    export AI_PROXY_PERSONAL_TOKEN=your_token
    python scripts/poc_summarize.py

Test mode (no API calls):
    python scripts/poc_summarize.py --test
"""

import os
import re
import json
import ssl
import urllib.request
import urllib.error
from dataclasses import dataclass
from typing import Optional

# SSL context for self-signed certificates
SSL_CONTEXT = ssl.create_default_context()
SSL_CONTEXT.check_hostname = False
SSL_CONTEXT.verify_mode = ssl.CERT_NONE

# Configuration
AI_PROXY_BASE_URL = os.environ.get("AI_PROXY_BASE_URL", "")
AI_PROXY_TOKEN = os.environ.get("AI_PROXY_PERSONAL_TOKEN", "")
MODEL = "gemini-2.5-flash-lite"

def get_endpoint():
    if not AI_PROXY_BASE_URL:
        raise ValueError("AI_PROXY_BASE_URL environment variable not set")
    return f"{AI_PROXY_BASE_URL}/google/v1beta/models/{MODEL}:generateContent"


@dataclass
class FunctionInfo:
    name: str
    signature: str
    body: str
    line_start: int
    line_end: int


def extract_rust_functions(source: str) -> list[FunctionInfo]:
    """Extract function definitions from Rust source code."""
    functions = []
    lines = source.split('\n')

    # Pattern to match function definitions
    fn_pattern = re.compile(r'^\s*(pub\s+)?(async\s+)?fn\s+(\w+)')

    i = 0
    while i < len(lines):
        line = lines[i]
        match = fn_pattern.match(line)

        if match:
            fn_name = match.group(3)
            fn_start = i + 1  # 1-indexed

            # Find the function signature (may span multiple lines until '{')
            sig_lines = [line]
            brace_count = line.count('{') - line.count('}')
            j = i + 1

            # If no opening brace yet, continue reading signature
            while brace_count == 0 and j < len(lines):
                sig_lines.append(lines[j])
                brace_count += lines[j].count('{') - lines[j].count('}')
                j += 1

            # Now find the end of the function (matching braces)
            while brace_count > 0 and j < len(lines):
                brace_count += lines[j].count('{') - lines[j].count('}')
                j += 1

            fn_end = j  # 1-indexed

            # Extract signature (first line up to opening brace)
            signature = '\n'.join(sig_lines).split('{')[0].strip()

            # Extract body (everything between the first { and last })
            full_text = '\n'.join(lines[i:fn_end])
            body_start = full_text.find('{')
            if body_start != -1:
                body = full_text[body_start+1:].rsplit('}', 1)[0].strip()
            else:
                body = ""

            functions.append(FunctionInfo(
                name=fn_name,
                signature=signature,
                body=body,
                line_start=fn_start,
                line_end=fn_end
            ))

            i = j
        else:
            i += 1

    return functions


def count_tokens_approx(text: str) -> int:
    """Approximate token count (rough estimate: ~4 chars per token)."""
    return len(text) // 4


def call_gemini_api(prompt: str, max_tokens: int = 100) -> tuple[Optional[str], dict]:
    """
    Call Gemini API via AI Proxy.

    Returns: (response_text, usage_info)
    """
    if not AI_PROXY_TOKEN:
        return None, {"error": "AI_PROXY_PERSONAL_TOKEN not set"}
    if not AI_PROXY_BASE_URL:
        return None, {"error": "AI_PROXY_BASE_URL not set"}

    endpoint = get_endpoint()
    headers = {
        "Authorization": f"Bearer {AI_PROXY_TOKEN}",
        "Content-Type": "application/json"
    }

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ],
        "generationConfig": {
            "maxOutputTokens": max_tokens,
            "temperature": 0.3
        }
    }

    try:
        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(endpoint, data=data, headers=headers, method='POST')

        with urllib.request.urlopen(req, timeout=30, context=SSL_CONTEXT) as response:
            result = json.loads(response.read().decode('utf-8'))

        # Extract response text
        candidates = result.get("candidates", [])
        if candidates:
            content = candidates[0].get("content", {})
            parts = content.get("parts", [])
            if parts:
                text = parts[0].get("text", "")
            else:
                text = ""
        else:
            text = ""

        # Extract usage metadata
        usage = result.get("usageMetadata", {})

        return text.strip(), {
            "promptTokenCount": usage.get("promptTokenCount", 0),
            "candidatesTokenCount": usage.get("candidatesTokenCount", 0),
            "totalTokenCount": usage.get("totalTokenCount", 0)
        }

    except urllib.error.HTTPError as e:
        error_body = e.read().decode('utf-8') if e.fp else str(e)
        return None, {"error": f"HTTP {e.code}: {error_body}"}
    except urllib.error.URLError as e:
        return None, {"error": str(e)}
    except Exception as e:
        return None, {"error": str(e)}


def generate_summary_prompt(fn_name: str, body: str) -> str:
    """Generate the prompt for function summarization."""
    # Truncate body if too long
    max_body_chars = 2000
    if len(body) > max_body_chars:
        body = body[:max_body_chars] + "\n... (truncated)"

    return f"""Summarize this function in one concise sentence (max 80 chars).
Focus on WHAT it does, not HOW.

Context: {fn_name}
```
{body}
```

Summary:"""


def main():
    # Read lib.rs
    lib_rs_path = "/Users/jaewooseo/git/save-context-search/src/lib.rs"

    print(f"Reading {lib_rs_path}...")
    with open(lib_rs_path, 'r') as f:
        source = f.read()

    # Extract functions
    functions = extract_rust_functions(source)
    print(f"Found {len(functions)} functions\n")

    # Select a few interesting functions for the PoC
    target_functions = [
        "intern",
        "rebuild_index",
        "load_or_create",
        "ensure_fresh",
        "search",
        "lookup",
        "status",
        "generate_embeddings"
    ]

    selected = [f for f in functions if f.name in target_functions]

    if not selected:
        # Fallback: take first 5 functions
        selected = functions[:5]

    print(f"Selected {len(selected)} functions for summarization:\n")
    print("-" * 60)

    total_input_tokens = 0
    total_output_tokens = 0

    for fn in selected:
        print(f"\n## {fn.name} (lines {fn.line_start}-{fn.line_end})")
        print(f"   Signature: {fn.signature[:80]}...")

        prompt = generate_summary_prompt(fn.name, fn.body)
        input_tokens_approx = count_tokens_approx(prompt)

        summary, usage = call_gemini_api(prompt)

        if summary:
            print(f"   Summary: {summary}")
            print(f"   Tokens: input={usage.get('promptTokenCount', 'N/A')}, output={usage.get('candidatesTokenCount', 'N/A')}")
            total_input_tokens += usage.get('promptTokenCount', 0)
            total_output_tokens += usage.get('candidatesTokenCount', 0)
        else:
            print(f"   Error: {usage.get('error', 'Unknown error')}")

    print("\n" + "=" * 60)
    print("COST ESTIMATION")
    print("=" * 60)

    # Gemini 2.5 Flash-Lite pricing (approximate, check actual pricing)
    # Input: $0.075 per 1M tokens, Output: $0.30 per 1M tokens
    input_cost = (total_input_tokens / 1_000_000) * 0.075
    output_cost = (total_output_tokens / 1_000_000) * 0.30
    total_cost = input_cost + output_cost

    print(f"Total input tokens:  {total_input_tokens:,}")
    print(f"Total output tokens: {total_output_tokens:,}")
    print(f"Estimated cost:      ${total_cost:.6f}")
    print(f"  - Input cost:      ${input_cost:.6f}")
    print(f"  - Output cost:     ${output_cost:.6f}")


def test_extraction():
    """Test function extraction without API calls."""
    lib_rs_path = "/Users/jaewooseo/git/save-context-search/src/lib.rs"

    print("=" * 60)
    print("TEST MODE: Function extraction only (no API calls)")
    print("=" * 60)

    with open(lib_rs_path, 'r') as f:
        source = f.read()

    functions = extract_rust_functions(source)
    print(f"\nFound {len(functions)} functions in lib.rs:\n")

    for fn in functions[:10]:  # Show first 10
        body_preview = fn.body[:100].replace('\n', ' ')
        print(f"  - {fn.name} (lines {fn.line_start}-{fn.line_end})")
        print(f"    Signature: {fn.signature[:60]}...")
        print(f"    Body preview: {body_preview}...")
        print()

    # Show what prompt would look like
    sample_fn = functions[0]
    prompt = generate_summary_prompt(sample_fn.name, sample_fn.body)
    print("=" * 60)
    print("SAMPLE PROMPT (would be sent to API):")
    print("=" * 60)
    print(prompt[:500])
    if len(prompt) > 500:
        print(f"... ({len(prompt)} chars total)")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_extraction()
    else:
        main()
