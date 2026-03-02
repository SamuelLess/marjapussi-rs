#!/usr/bin/env python3
"""
Watch a run directory and send a Telegram Saved Messages notification on completion.

Requirements:
  pip install telethon

Env vars:
  TG_API_ID   (required)
  TG_API_HASH (required)
  TG_PHONE    (required only on first login/session creation)
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

from telethon import TelegramClient
from telethon.errors import SessionPasswordNeededError


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True, help="Run directory (e.g. ml/runs/my_run)")
    p.add_argument(
        "--session-file",
        default="ml/.telegram_saved_messages",
        help="Telethon session file path",
    )
    p.add_argument("--poll-seconds", type=int, default=20, help="Polling interval")
    p.add_argument("--timeout-hours", type=float, default=72.0, help="Max watch duration")
    p.add_argument(
        "--tag",
        default="training",
        help="Short tag included in the notification message",
    )
    return p.parse_args()


def completion_marker(run_dir: Path) -> Path | None:
    ckpt = run_dir / "checkpoints"
    if not ckpt.exists():
        return None

    # Preferred final markers.
    for pat in ("*_selfplay_final.pt", "phase_full_game_complete*.pt"):
        matches = sorted(ckpt.glob(pat))
        if matches:
            return matches[-1]

    # Fallback: if run has best+latest and a final log exists, accept latest.
    latest = ckpt / "latest.pt"
    best = ckpt / "best.pt"
    if latest.exists() and best.exists():
        return latest
    return None


async def send_saved_message(session_file: str, text: str) -> None:
    api_id = os.environ.get("TG_API_ID")
    api_hash = os.environ.get("TG_API_HASH")
    if not api_id or not api_hash:
        raise RuntimeError("TG_API_ID and TG_API_HASH must be set.")

    phone = os.environ.get("TG_PHONE")
    client = TelegramClient(session_file, int(api_id), api_hash)
    await client.connect()
    try:
        if not await client.is_user_authorized():
            if not phone:
                raise RuntimeError(
                    "First login required: set TG_PHONE (e.g. +49...) and run again."
                )
            await client.send_code_request(phone)
            code = input("Enter Telegram login code: ").strip()
            try:
                await client.sign_in(phone=phone, code=code)
            except SessionPasswordNeededError:
                pw = input("Telegram 2FA password: ").strip()
                await client.sign_in(password=pw)
        await client.send_message("me", text)
    finally:
        await client.disconnect()


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    start = time.time()
    timeout = args.timeout_hours * 3600.0

    print(f"[watch] run={run_dir}")
    print(f"[watch] polling every {args.poll_seconds}s for up to {args.timeout_hours:.1f}h")

    while True:
        marker = completion_marker(run_dir)
        if marker is not None:
            msg = (
                f"[{args.tag}] finished\n"
                f"run: {run_dir}\n"
                f"marker: {marker}\n"
            )
            asyncio.run(send_saved_message(args.session_file, msg))
            print("[watch] notification sent to Saved Messages.")
            return 0

        if time.time() - start > timeout:
            print("[watch] timeout reached without completion marker.", file=sys.stderr)
            return 2

        time.sleep(max(5, args.poll_seconds))


if __name__ == "__main__":
    raise SystemExit(main())
