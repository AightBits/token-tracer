#!/usr/bin/env python3
"""
vLLM token-pool tracer (pure REST), assistant-aware tokenization, with top_k disabled by default.

- Calls /v1/chat/completions with logprobs enabled.
- Tokenizes prompt as USER and completion/candidates as ASSISTANT using the server tokenizer.
- If detokenize endpoints are absent, prints IDs only for token lists.
- Top-k sampling is DISABLED by default (top_k = -1) to avoid altering sampling vs your earlier runs.
"""

import math
import argparse
import requests


def fmt_prob(logprob: float) -> str:
    """Format log probability for human readability"""
    try:
        p = math.exp(logprob)
    except OverflowError:
        p = 0.0
    return f"logp={logprob: .4f} p={p*100:6.2f}%"


def print_separator(width: int = 100):
    print("-" * width)


def extract_ids(obj: dict):
    """Extract token IDs from various response formats"""
    return (
        obj.get("tokens")
        or obj.get("token_ids")
        or obj.get("input_ids")
        or []
    )


def format_token_display(token_ids, max_display, description):
    """Consistent token display formatting"""
    n = len(token_ids)
    show_count = n if (max_display <= 0 or max_display >= n) else max_display
    display_tokens = token_ids[:show_count]
    
    print(f"{description}:")
    print(display_tokens)
    if show_count < n:
        print(f"... ({n - show_count} more tokens not shown)")
    print_separator()


def main():
    ap = argparse.ArgumentParser(description="vLLM token-pool tracer (REST) with assistant-aware tokenization")
    ap.add_argument("--base-url", default="http://localhost:8000", help="Base URL without trailing /v1")
    ap.add_argument("--max-new-tokens", type=int, default=64, help="Max new tokens to generate")
    ap.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    ap.add_argument("--top-p", type=float, default=0.9, help="Top-p nucleus sampling")
    ap.add_argument("--top-k", type=int, default=-1, help="Top-k cutoff; -1 disables")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    ap.add_argument("--top-logprobs", type=int, default=5, help="Candidate tokens to display per step")
    ap.add_argument("--prompt", default="In a single sentence, what is a cat?", help="Prompt text")
    ap.add_argument("--max-print-tokens", type=int, default=0, help="Limit tokens to print (0=all)")
    args = ap.parse_args()

    # Use session for connection pooling
    session = requests.Session()
    
    def post_json(url: str, payload: dict, headers: dict):
        r = session.post(url, headers=headers, json=payload, timeout=120)
        r.raise_for_status()
        return r.json()

    def _try_tokenize(base_url: str, headers: dict, body: dict):
        """Try multiple tokenize endpoints"""
        for path in ("/tokenize", "/v1/tokenize"):
            url = f"{base_url}{path}"
            try:
                return post_json(url, body, headers), url
            except requests.HTTPError:
                continue
        raise requests.HTTPError("All tokenize endpoints failed")

    def tokenize_text(base_url: str, headers: dict, *, model: str | None, text: str, role: str = "user"):
        """Unified tokenization function with role support"""
        # Try messages format first
        body = {"messages": [{"role": role, "content": text}]}
        if model:
            body["model"] = model
        try:
            data, used = _try_tokenize(base_url, headers, body)
            return data, used, f"messages({role})"
        except requests.HTTPError:
            pass

        # Fallback to prompt format
        body = {"prompt": text}
        if model:
            body["model"] = model
        data, used = _try_tokenize(base_url, headers, body)
        return data, used, "prompt(fallback)"

    def get_single_token_ids(base_url: str, headers: dict, model: str | None, text: str):
        """Get token IDs for a single token, handling BOS token properly"""
        if text == "":
            return []  # Empty string has no tokens
            
        data, _, _ = tokenize_text(base_url, headers, model=model, text=text, role="assistant")
        token_ids = extract_ids(data)
        
        # Remove BOS token (1) if present and we have multiple tokens
        if len(token_ids) > 1 and token_ids[0] == 1:
            return token_ids[1:]
        return token_ids

    chat_url = f"{args.base_url}/v1/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": "Bearer EMPTY"}
    
    # Build payload
    payload = {
        "messages": [{"role": "user", "content": args.prompt}],
        "max_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "logprobs": True,
        "top_logprobs": args.top_logprobs,
        "seed": args.seed,
    }
    if args.top_k >= 0:
        payload["top_k"] = args.top_k

    # Request completion
    data = post_json(chat_url, payload, headers)
    choice = data["choices"][0]
    final_text = choice["message"].get("content", "") or ""
    model_name = data.get("model") or choice.get("model") or None

    # Tokenize prompt and completion using unified function
    prompt_tok, used_url_prompt, prompt_mode = tokenize_text(
        args.base_url, headers, model=model_name, text=args.prompt, role="user"
    )
    prompt_token_ids = extract_ids(prompt_tok)

    out_tok, used_url_out, completion_mode = tokenize_text(
        args.base_url, headers, model=model_name, text=final_text, role="assistant"
    )
    out_token_ids = extract_ids(out_tok)
    assistant_mode_ok = completion_mode.startswith("messages(")

    # Setup token cache for performance
    token_cache = {}
    def cached_get_single_token_ids(text: str):
        if text not in token_cache:
            token_cache[text] = get_single_token_ids(
                args.base_url, headers, model=model_name, text=text
            )
        return token_cache[text]

    # ----------------------- OUTPUT -----------------------
    print_separator()
    print("Configuration:")
    print(f"  Seed: {args.seed}, Temperature: {args.temperature}, Top-p: {args.top_p}")
    print(f"  Top-k: {'(disabled)' if args.top_k < 0 else args.top_k}")
    print(f"  Max new tokens: {args.max_new_tokens}")
    print(f"  Server model: {model_name or '(not returned)'}")
    print(f"  Tokenizer modes: prompt={prompt_mode}, completion={completion_mode}")
    if not assistant_mode_ok:
        print("  Note: completion tokenization fell back to 'prompt' form")

    print_separator()
    print("PROMPT:")
    print(args.prompt)
    print_separator()
    format_token_display(prompt_token_ids, args.max_print_tokens, "PROMPT TOKEN IDs")

    print("COMPLETION:")
    print(final_text)
    print_separator()
    format_token_display(out_token_ids, args.max_print_tokens, "COMPLETION TOKEN IDs")

    # Per-token breakdown with cached tokenization
    steps = choice.get("logprobs", {}).get("content") or []
    
    # Remove BOS token from completion tokens for proper alignment
    completion_tokens_without_bos = out_token_ids[1:] if out_token_ids and out_token_ids[0] == 1 else out_token_ids
    
    print(f"Logprob steps: {len(steps)}, Completion tokens (without BOS): {len(completion_tokens_without_bos)}")
    if len(steps) != len(completion_tokens_without_bos):
        print(f"Note: Step count ({len(steps)}) differs from token count ({len(completion_tokens_without_bos)})")
    print_separator()
    print("Per-token candidate breakdown:")
    print_separator()

    for idx, step in enumerate(steps):
        tok_text = step.get("token", "")
        tok_lp = step.get("logprob", float("-inf"))
        tok_ids = cached_get_single_token_ids(tok_text)
        
        # Get the actual token ID from completion (aligned properly)
        actual_id = completion_tokens_without_bos[idx] if idx < len(completion_tokens_without_bos) else "N/A"
        
        # Verify alignment
        alignment_ok = (len(tok_ids) == 1 and tok_ids[0] == actual_id) if idx < len(completion_tokens_without_bos) else "N/A"
        alignment_marker = "✓" if alignment_ok == True else "✗" if alignment_ok == False else " "
        
        print(f"Step {idx + 1:>3} | chosen: {repr(tok_text):<20} ids={tok_ids} {alignment_marker} (actual: {actual_id})  ({fmt_prob(tok_lp)})")
        print("  candidates:")

        cands_sorted = sorted(step.get("top_logprobs", []), key=lambda c: c["logprob"], reverse=True)
        for rank, cand in enumerate(cands_sorted[:args.top_logprobs], start=1):
            ct = cand["token"]
            clp = cand["logprob"]
            c_ids = cached_get_single_token_ids(ct)
            marker = "*" if ct == tok_text else " "
            print(f"    {rank:>2}. {marker} {repr(ct):<20} ids={c_ids}  {fmt_prob(clp)}")

        print_separator()
    
    # Cleanup
    try:
        session.close()
    except:
        pass


if __name__ == "__main__":
    main()
