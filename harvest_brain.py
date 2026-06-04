#!/usr/bin/env python3
"""Brain harvesting script — analyze context.json for mistakes and improvement opportunities.

Reads the full conversation history, chunks it, and sends each chunk to the LLM
for analysis. Outputs a report of mistakes made and opportunities for improvement.

Usage:
    python3 harvest_brain.py                    # analyze context.json, print report
    python3 harvest_brain.py -o report.md       # write report to file
    python3 harvest_brain.py --context other.json  # use a different context file
"""

import argparse
import json
import os
import sys
import time

from config import PROJECT_DIR, TOKEN_ESTIMATE_DIVISOR
from context import Context, _ts_fmt
from ai_client import AIClient


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // TOKEN_ESTIMATE_DIVISOR)


def format_message(msg: dict) -> str:
    """Format a single message into a human-readable log line."""
    role = msg.get("role", "?")
    ts = msg.get("_ts", 0)
    ts_str = _ts_fmt(ts) if ts else "[??:??:??]"
    content = msg.get("content", "")

    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict) and p.get("type") == "text":
                parts.append(p["text"])
            elif isinstance(p, dict) and p.get("type") == "image_url":
                parts.append("[image]")
        content = " ".join(parts)

    tool_calls = msg.get("tool_calls", [])
    if tool_calls:
        tools = ", ".join(tc["function"]["name"] for tc in tool_calls)
        args_parts = []
        for tc in tool_calls:
            args = tc["function"].get("arguments", "{}")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    pass
            if isinstance(args, dict):
                # Show key args concisely
                short = {k: (v[:80] + "..." if isinstance(v, str) and len(v) > 80 else v)
                         for k, v in args.items()}
                args_parts.append(json.dumps(short))
            else:
                args_parts.append(str(args))
        content = f"[calls: {tools}] {'; '.join(args_parts)} {content}".strip()

    if role == "tool":
        name = msg.get("name", "?")
        # Truncate long tool results
        if len(content) > 300:
            content = content[:300] + "..."
        return f"{ts_str} tool({name}): {content}"

    return f"{ts_str} {role}: {content}"


def chunk_messages(messages: list, max_tokens_per_chunk: int = 12000) -> list[list[dict]]:
    """Split messages into chunks that fit within token limits."""
    chunks = []
    current = []
    current_tokens = 0

    for msg in messages:
        text = format_message(msg)
        tokens = _estimate_tokens(text)

        if current_tokens + tokens > max_tokens_per_chunk and current:
            chunks.append(current)
            current = []
            current_tokens = 0

        current.append(msg)
        current_tokens += tokens

    if current:
        chunks.append(current)

    return chunks


HARVEST_PROMPT = """\
You are reviewing a conversation log between an AI assistant (running on a Raspberry Pi with camera and e-ink display) and its user, Austin.

The AI's purpose: keeping Austin honest about daily habits — getting up from the desk, drinking water, staying on track with studying and applications. It's a friendly nudge buddy.

Analyze this conversation chunk and identify:

## 1. MISTAKES
Things the AI did wrong or poorly:
- Ignored user requests or misunderstood what they wanted
- Talked too much when the user wasn't engaging (violated pacing rules)
- Used emoji or AI tropes despite being told not to
- Showed notifications at bad times
- Gave wrong information from searches
- Failed to follow up on things the user said
- Was annoying, repetitive, or tone-deaf
- Made poor tool choices (e.g., capture_photo when take_photo would suffice)
- Missed opportunities to actually nudge on habits when it mattered

## 2. WINS
Things that worked well:
- Good timing on nudges that the user responded to
- Natural conversation that felt like a real friend
- Useful information shared at the right moment
- Appropriate use of display vs chat
- Good pacing — knew when to be quiet

## 3. USER PREFERENCES REVEALED
Things we learned about what Austin wants:
- Topics he engaged with vs ignored
- Times he was active vs away
- How he responded to different types of nudges
- Conversation styles he seemed to prefer
- Things that annoyed him

## 4. IMPROVEMENT OPPORTUNITIES
Concrete changes for the next generation:
- System prompt tweaks
- Tool usage patterns to change
- Timing/pacing adjustments
- New capabilities that would help
- Things to stop doing entirely

Be specific. Quote actual messages where relevant. Be honest — don't sugarcoat mistakes.
If this chunk is mostly routine monitoring with nothing notable, say so briefly and move on.

Here is the conversation log:

{log}
"""

SYNTHESIS_PROMPT = """\
You are synthesizing multiple analysis chunks from a brain harvesting session.
Each chunk analyzed a portion of an AI assistant's conversation history with its user Austin.

Combine the chunk analyses into a single coherent report. Deduplicate, prioritize the most important findings, and organize clearly.

Structure:
## Mistakes & Anti-patterns
Most impactful issues first. Group related mistakes.

## What Worked
Patterns to keep and reinforce.

## User Profile
What we learned about Austin's preferences, schedule, and personality.

## Actionable Improvements
Concrete, prioritized list of changes. Each item should be specific enough to implement.
For system prompt changes, suggest the actual text.
For behavioral changes, describe the rule clearly.

Be direct and concise. This report will be used to build the next generation of the AI.

Here are the chunk analyses:

{analyses}
"""


def harvest(context_file: str, verbose: bool = False) -> str:
    """Run the full brain harvest pipeline. Returns the final report."""
    # Load context
    if not os.path.exists(context_file):
        print(f"Error: {context_file} not found", file=sys.stderr)
        sys.exit(1)

    with open(context_file, "r") as f:
        messages = json.load(f)

    print(f"Loaded {len(messages)} messages from {context_file}")

    # Skip system prompt for analysis
    if messages and messages[0].get("role") == "system":
        messages = messages[1:]

    if not messages:
        print("No messages to analyze.")
        sys.exit(0)

    # Time range
    timestamps = [m.get("_ts", 0) for m in messages if m.get("_ts")]
    if timestamps:
        t_min = time.strftime("%Y-%m-%d %H:%M", time.localtime(min(timestamps)))
        t_max = time.strftime("%Y-%m-%d %H:%M", time.localtime(max(timestamps)))
        print(f"Time range: {t_min} to {t_max}")

    # Chunk
    chunks = chunk_messages(messages)
    print(f"Split into {len(chunks)} chunks for analysis")

    client = AIClient()
    analyses = []

    for i, chunk in enumerate(chunks):
        log_lines = [format_message(m) for m in chunk]
        log_text = "\n".join(log_lines)

        if verbose:
            print(f"\n--- Chunk {i+1}/{len(chunks)} ({len(chunk)} messages, ~{_estimate_tokens(log_text)} tokens) ---")

        prompt = HARVEST_PROMPT.format(log=log_text)

        try:
            payload = {
                "model": client.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 4096,
                "temperature": 0.3,
            }
            from ai_client import _api_call
            resp = _api_call(
                f"{client.base_url}/chat/completions",
                client._headers,
                payload,
                180,
                caller=f"harvest chunk {i+1}: ",
            )
            analysis = resp["choices"][0]["message"]["content"].strip()
            analyses.append(analysis)
            print(f"  Chunk {i+1}/{len(chunks)} done ({len(analysis)} chars)")
        except Exception as e:
            print(f"  Chunk {i+1}/{len(chunks)} FAILED: {e}", file=sys.stderr)
            analyses.append(f"[Analysis failed for chunk {i+1}: {e}]")

    # Synthesize if multiple chunks
    if len(analyses) == 1:
        report = analyses[0]
    else:
        print(f"\nSynthesizing {len(analyses)} chunk analyses...")
        combined = "\n\n---\n\n".join(
            f"### Chunk {i+1}\n{a}" for i, a in enumerate(analyses)
        )
        prompt = SYNTHESIS_PROMPT.format(analyses=combined)

        try:
            payload = {
                "model": client.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 8192,
                "temperature": 0.3,
            }
            resp = _api_call(
                f"{client.base_url}/chat/completions",
                client._headers,
                payload,
                240,
                caller="harvest synthesis: ",
            )
            report = resp["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"Synthesis failed: {e}", file=sys.stderr)
            report = "# Synthesis Failed\n\nIndividual chunk analyses:\n\n" + "\n\n---\n\n".join(analyses)

    # Add header
    header = f"# Brain Harvest Report\n\nGenerated: {time.strftime('%Y-%m-%d %H:%M')}\n"
    header += f"Source: {context_file}\n"
    header += f"Messages analyzed: {len(messages)}\n"
    if timestamps:
        header += f"Period: {t_min} to {t_max}\n"
    header += f"Chunks: {len(chunks)}\n\n---\n\n"

    return header + report


def main():
    parser = argparse.ArgumentParser(description="Harvest brain from AI conversation context")
    parser.add_argument("--context", "-c", default=os.path.join(PROJECT_DIR, "context.json"),
                        help="Path to context.json (default: %(default)s)")
    parser.add_argument("--output", "-o", default=None,
                        help="Write report to file (default: print to stdout)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show chunk details")
    args = parser.parse_args()

    report = harvest(args.context, verbose=args.verbose)

    if args.output:
        with open(args.output, "w") as f:
            f.write(report)
        print(f"\nReport written to {args.output}")
    else:
        print("\n" + "=" * 60)
        print(report)


if __name__ == "__main__":
    main()
