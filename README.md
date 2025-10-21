# Token Tracer

A Python script for analyzing token probabilities and tokenization in vLLM models via REST API.

## Overview

token_tracer.py calls vLLM's /v1/chat/completions endpoint with logprobs enabled to examine token sampling behavior. It tokenizes prompts as USER role and completions/candidates as ASSISTANT role using the server's tokenizer, falling back to prompt formatting when assistant-aware tokenization is unavailable.

## Features

- Token probability analysis with logprobs and top-k candidates
- Role-aware tokenization (USER for prompts, ASSISTANT for completions)  
- Configurable sampling parameters (temperature, top-p, top-k, seed)
- Token ID verification between individual tokens and full completions
- Connection pooling for performance

## Requirements

- Python 3.7+
- requests library
- vLLM server running with REST API enabled

## Installation

1. Download token_tracer.py
2. Install dependencies: pip install requests

## Usage

python3 token_tracer.py [options]

### Basic Example

python3 token_tracer.py --prompt="What is machine learning?" --seed=42

### Command Line Options

--base-url: vLLM server URL (default: http://localhost:8000)

--max-new-tokens: Maximum tokens to generate (default: 64)

--temperature: Sampling temperature (default: 0.7)

--top-p: Top-p nucleus sampling (default: 0.9)

--top-k: Top-k sampling (-1 to disable, default: -1)

--seed: Random seed for reproducibility (default: 42)

--top-logprobs: Number of candidate tokens to display per step (default: 5)

--prompt: Input prompt text

--max-print-tokens: Limit tokens displayed (0 for all, default: 0)

## Output

The script provides:

1. Configuration summary: Sampling parameters and model info
2. Prompt analysis: Original text and token IDs
3. Completion analysis: Generated text and token IDs
4. Per-token breakdown: For each generation step:
   - Chosen token with probability and token IDs
   - Top candidate tokens with probabilities
   - Token ID verification between individual and batch tokenization

## Tokenization Behavior

- Prompts are tokenized as USER role messages when possible
- Completions and candidate tokens are tokenized as ASSISTANT role
- Falls back to prompt formatting if role-aware endpoints fail
- Handles BOS (beginning of sequence) and EOS (end of sequence) tokens appropriately

## Notes

- Top-k sampling is disabled by default (top_k = -1) to maintain consistency with earlier sampling behavior
- The script requires vLLM server with enabled tokenize endpoints (/tokenize or /v1/tokenize)
- Token ID mismatches may occur if the server doesn't support role-aware tokenization

## Example Output

Step   1 | chosen: 'A'                  ids=[1065] ✓ (actual: 1065)  (logp=-0.0053 p= 99.47%)

  candidates:

     1. * 'A'                  ids=[1065]  logp=-0.0053 p= 99.47%
     
     2.   '"A'                 ids=[93192]  logp=-5.3803 p=  0.46%
     
     3.   '"'                  ids=[1034]  logp=-7.7553 p=  0.04%
     
## Limitations

- Requires vLLM server with logprobs support
- Dependent on server tokenizer behavior
- May show token count mismatches due to EOS handling
- Assistant role tokenization depends on server capabilities
