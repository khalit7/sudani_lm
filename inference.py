"""Chat with a trained checkpoint.

    python inference.py                                   # talk to the final model
    python inference.py --checkpoint .../stage_c/best.pt  # compare against Stage C
    python inference.py --temperature 0                   # greedy, for reproducible output
    echo "كيف الحال" | python inference.py --once         # one turn, non-interactive

Commands inside the session:
    /reset            start a new conversation
    /temp <value>     change sampling temperature
    /speaker <name>   change who you are (the model was trained on 1,169 named senders)
    /transcript       print the conversation so far
    /quit             exit
"""

import argparse
import sys
from pathlib import Path

import torch

from src.chat import DEFAULT_CHECKPOINT, ChatSession, load_model
from src.tokenizer.utils import get_tokenizer


def build_session(args):
    model, device, checkpoint = load_model(args.checkpoint)
    session = ChatSession(
        model, get_tokenizer(), device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        speaker=args.speaker,
    )
    return session, checkpoint


def handle_command(session, line) -> None:
    parts = line.split(maxsplit=1)
    command, argument = parts[0], (parts[1] if len(parts) > 1 else "")

    if command in ("/quit", "/exit"):
        raise SystemExit(0)
    if command == "/reset":
        session.reset()
        print("  (conversation cleared)")
    elif command == "/temp":
        try:
            session.temperature = float(argument)
            print(f"  (temperature = {session.temperature})")
        except ValueError:
            print("  usage: /temp 0.8")
    elif command == "/speaker":
        if argument.strip():
            session.speaker = argument.strip()
            print(f"  (you are now {session.speaker})")
        else:
            print("  usage: /speaker Mukh")
    elif command == "/transcript":
        print(session.transcript() or "  (nothing yet)")
    else:
        print(f"  unknown command {command}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--speaker", default="Friend",
                        help="the name you appear under in the conversation")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--once", action="store_true",
                        help="read one message from stdin, print one reply, exit")
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    session, checkpoint = build_session(args)
    print(
        f"sudani_lm · {args.checkpoint.parent.name} (step {checkpoint.get('step')}) · "
        f"T={args.temperature} · you are {args.speaker}",
        file=sys.stderr,
    )

    if args.once:
        message = sys.stdin.read().strip()
        if message:
            print(session.reply(message))
        return 0

    print("  /reset  /temp <v>  /speaker <name>  /transcript  /quit", file=sys.stderr)
    while True:
        try:
            line = input(f"{session.speaker}: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not line:
            continue
        if line.startswith("/"):
            handle_command(session, line)
            continue
        print(f"ME: {session.reply(line)}")


if __name__ == "__main__":
    raise SystemExit(main())
