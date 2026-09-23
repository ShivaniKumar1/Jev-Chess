#!/usr/bin/env python3
"""
Jev Plays Chess

Has TypeSafe AI's Jev model (via DO's /v1/systemone endpoint) play chess
against itself in a local web UI -- one decision per move, not per candidate
line. Jev picks from a list of legal moves each turn, and separately rates
the position and its sharpness, all in a single request per move.

Setup:
    pip install requests python-chess
    export MODEL_ACCESS_KEY="sk-do-..."
    python3 jev_plays_chess.py --moves 60 --delay 1.0

Run `python3 jev_plays_chess.py --help` for all flags.
"""

import argparse
import json
import os
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

try:
    import requests
except ImportError:
    print("This script needs the 'requests' package: pip install requests", file=sys.stderr)
    sys.exit(1)

try:
    import chess
    import chess.pgn
except ImportError:
    print("This script needs the 'python-chess' package: pip install python-chess", file=sys.stderr)
    sys.exit(1)

DEFAULT_INFERENCE_PROXY_BASE_URL = "https://inference.do-ai.run"
DEFAULT_WEB_UI_PORT = 8766
UI_HTML_PATH = Path(__file__).resolve().parent / "ui.html"

PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0,
}


def material_balance(board):
    """White material minus Black material, in pawns."""
    total = 0
    for piece_type, value in PIECE_VALUES.items():
        total += value * len(board.pieces(piece_type, chess.WHITE))
        total -= value * len(board.pieces(piece_type, chess.BLACK))
    return total


def move_features(board, move):
    """Push `move` on a scratch copy of `board` and compute the exact same
    raw features that get described to Jev in plain language. Keeping this
    as the single source of truth means the naive greedy baseline (below)
    is judged on identical information to what Jev sees -- no unfair
    advantage either way."""
    mover_is_white = board.turn == chess.WHITE
    is_capture = board.is_capture(move)
    san = board.san(move)

    scratch = board.copy()
    scratch.push(move)

    gives_check = scratch.is_check()
    is_mate = scratch.is_checkmate()
    balance_after = material_balance(scratch)
    # Report material from the mover's perspective (positive = good for the mover).
    balance_for_mover = balance_after if mover_is_white else -balance_after
    reply_count = scratch.legal_moves.count()

    return {
        "san": san,
        "is_capture": is_capture,
        "gives_check": gives_check,
        "is_mate": is_mate,
        "material_for_mover": balance_for_mover,
        "opponent_replies": reply_count,
    }


def describe_features(features):
    """Plain-language description of `features`, exactly what gets sent to
    Jev as a candidate move's `criteria` text."""
    parts = [f"move {features['san']}"]
    parts.append("a capture" if features["is_capture"] else "not a capture")
    if features["is_mate"]:
        parts.append("delivers checkmate")
    elif features["gives_check"]:
        parts.append("gives check")
    parts.append(f"material balance after the move: {features['material_for_mover']:+.0f} pawns for the mover")
    parts.append(f"opponent has {features['opponent_replies']} legal replies afterward")
    return ", ".join(parts)


def greedy_baseline_score(features):
    """Naive baseline: argmax over the exact same features described to
    Jev, weighted in line with the instructions we give Jev ("prefer moves
    that win material, deliver check or mate, and avoid handing the
    opponent a much stronger reply position").

    This exists to answer one question: is Jev doing anything beyond
    sorting by the obvious numbers we handed it? See `baseline_divergent`
    / `baseline_total` in the run state and the README for how the
    divergence rate is used."""
    if features["is_mate"]:
        return 1_000_000.0
    score = float(features["material_for_mover"])
    if features["gives_check"]:
        score += 0.3
    score -= features["opponent_replies"] * 0.02
    return score


def board_metrics_for_narration(board):
    """Simple position-health signal for the score/noul narration questions."""
    balance = material_balance(board)
    balance_for_side_to_move = balance if board.turn == chess.WHITE else -balance
    return balance_for_side_to_move


def board_grid(board, last_move):
    """Flatten the board into a8..h1 order for the web UI, with piece codes
    like 'wP'/'bK' and a flag for squares touched by the last move."""
    grid = []
    for rank in range(7, -1, -1):
        for file in range(8):
            square = chess.square(file, rank)
            piece = board.piece_at(square)
            piece_code = None
            if piece is not None:
                color = "w" if piece.color == chess.WHITE else "b"
                piece_code = color + piece.symbol().upper()
            highlight = last_move is not None and square in (last_move.from_square, last_move.to_square)
            light = (file + rank) % 2 == 1
            grid.append(
                {
                    "square": chess.square_name(square),
                    "piece": piece_code,
                    "highlight": highlight,
                    "light": light,
                }
            )
    return grid


def result_reason(board):
    if board.is_checkmate():
        winner = "Black" if board.turn == chess.WHITE else "White"
        return f"Checkmate -- {winner} wins."
    if board.is_stalemate():
        return "Stalemate -- draw."
    if board.is_insufficient_material():
        return "Insufficient material -- draw."
    if board.can_claim_threefold_repetition():
        return "Threefold repetition -- draw."
    if board.can_claim_fifty_moves():
        return "Fifty-move rule -- draw."
    return None


class WebUI:
    def __init__(self, port=DEFAULT_WEB_UI_PORT):
        self._lock = threading.Lock()
        self.closed = False
        self._start_handler = None
        self._end_handler = None
        self._state = {
            "status": "Click Start to let Jev play.",
            "grid": board_grid(chess.Board(), None),
            "ply_no": 0,
            "side_to_move": "white",
            "last_move_san": None,
            "forced": None,
            "material_balance": 0,
            "sharp_noul": None,
            "candidate_count": None,
            "latency_ms": None,
            "last_call": None,
            "result": None,
            "baseline_agreement": None,
            "baseline_alt_san": None,
            "baseline_divergent": 0,
            "baseline_total": 0,
            "baseline_divergence_pct": None,
            "running": False,
        }

        try:
            self._server = ThreadingHTTPServer(("127.0.0.1", port), self._make_handler())
        except OSError as exc:
            raise RuntimeError(
                f"Could not start the web UI on http://127.0.0.1:{port} ({exc}).\n"
                f"This usually means another copy of this demo is still running there.\n"
                f"Either open http://127.0.0.1:{port} in your browser (it may already be live),\n"
                f"or stop the other process (Ctrl+C in its terminal) and try again."
            ) from exc
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}"
        webbrowser.open(self.url, new=1)

    def _make_handler(self):
        ui = self

        class Handler(BaseHTTPRequestHandler):
            def _respond_json(self, payload, status=200):
                body = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _respond_html(self, html, status=200):
                body = html.encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path == "/":
                    self._respond_html(ui._html())
                    return
                if self.path == "/state":
                    self._respond_json(ui.snapshot())
                    return
                if self.path == "/favicon.ico":
                    self.send_response(204)
                    self.end_headers()
                    return
                self.send_error(404)

            def do_POST(self):
                if self.path == "/start":
                    if ui._start_handler is not None:
                        ui._start_handler()
                    self.send_response(204)
                    self.end_headers()
                    return
                if self.path == "/end":
                    if ui._end_handler is not None:
                        ui._end_handler()
                    self.send_response(204)
                    self.end_headers()
                    return
                self.send_error(404)

            def log_message(self, fmt, *args):
                return

        return Handler

    def _html(self):
        return UI_HTML_PATH.read_text(encoding="utf-8")

    def snapshot(self):
        with self._lock:
            return dict(self._state)

    def update(self, **kwargs):
        if self.closed:
            return
        with self._lock:
            self._state.update(kwargs)

    def set_start_handler(self, handler):
        self._start_handler = handler

    def set_end_handler(self, handler):
        self._end_handler = handler

    def run_forever(self):
        print(f"Web UI available at {self.url}")
        try:
            while not self.closed:
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass
        finally:
            self.close()

    def close(self):
        if self.closed:
            return
        self.closed = True
        self._server.shutdown()
        self._server.server_close()


def call_jev(base_url, access_key, model, board, candidates, capture=None):
    headers = {
        "Authorization": f"Bearer {access_key}",
        "Content-Type": "application/json",
    }

    criteria = {move_id: desc for move_id, desc, _, _ in candidates}
    balance_for_mover = board_metrics_for_narration(board)

    body = {
        "state": {
            "fen": board.fen(),
            "side_to_move": "white" if board.turn == chess.WHITE else "black",
            "material_balance_for_side_to_move": balance_for_mover,
            "fullmove_number": board.fullmove_number,
        },
        "model": model,
        "questions": {
            "best_move": {
                "type": "choice",
                "instructions": (
                    "Chess position, given as FEN, plus a list of legal moves for the side "
                    "to move with their effects described. Choose the strongest move: prefer "
                    "moves that win material, deliver check or mate, and avoid moves that hand "
                    "the opponent a much stronger reply position."
                ),
                "criteria": criteria,
            },
            "is_sharp": {
                "type": "noul",
                "instructions": "Is this position tactically sharp -- does a single move likely swing the outcome significantly?",
            },
        },
    }

    endpoint = f"{base_url.rstrip('/')}/v1/systemone"
    if capture is not None:
        capture["endpoint"] = endpoint
        capture["request"] = body

    resp = requests.post(endpoint, headers=headers, json=body, timeout=30)
    resp.raise_for_status()
    return resp.json()


def main():
    parser = argparse.ArgumentParser(description="Jev plays chess (self-play) via /v1/systemone")
    parser.add_argument("--moves", type=int, default=60, help="Stop after N half-moves/plies.")
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--model", default="typesafe-jev-1.13.0")
    parser.add_argument("--port", type=int, default=DEFAULT_WEB_UI_PORT)
    parser.add_argument("--pgn", default=None, help="Write the game to this PGN file when it ends.")
    args = parser.parse_args()

    try:
        ui = WebUI(port=args.port)
    except RuntimeError as e:
        print(e, file=sys.stderr)
        sys.exit(1)

    os.environ["INFERENCE_PROXY_BASE_URL"] = DEFAULT_INFERENCE_PROXY_BASE_URL
    base_url = DEFAULT_INFERENCE_PROXY_BASE_URL
    state_lock = threading.Lock()
    state = {
        "run_id": 0,
        "running": False,
    }

    def finish_run(message):
        with state_lock:
            state["running"] = False
        ui.update(status=message, running=False)
        print(message)

    def end_game():
        with state_lock:
            if not state["running"]:
                ui.update(status="Idle. Click Start to let Jev play.")
                return
            state["run_id"] += 1
            state["running"] = False
        ui.update(status="Stopped. Click Start to play again.", running=False)

    def is_active(run_id):
        with state_lock:
            return state["running"] and run_id == state["run_id"]

    def play_loop(run_id):
        baseline_divergent = 0
        baseline_total = 0
        try:
            access_key = os.environ.get("MODEL_ACCESS_KEY")
            board = chess.Board()
            game = chess.pgn.Game()
            node = game

            for ply_no in range(1, args.moves + 1):
                if not is_active(run_id):
                    return
                if board.is_game_over():
                    reason = result_reason(board)
                    finish_run(reason or "Game over.")
                    ui.update(result=reason)
                    return

                legal_moves = list(board.legal_moves)
                side_label = "White" if board.turn == chess.WHITE else "Black"

                if len(legal_moves) == 1:
                    # TypeSafe's `choice` type requires at least 2 options -- a single
                    # forced move (only escape from check, forced recapture, etc.) has
                    # no real decision to make, so skip the API call entirely.
                    chosen_move = legal_moves[0]
                    forced = True
                    latency_ms = None
                    sharp_noul = None
                    candidate_count = 1
                    baseline_agreement = None
                    baseline_alt_san = None
                    last_call_update = {}
                else:
                    forced = False
                    candidates = []
                    for move in legal_moves:
                        features = move_features(board, move)
                        candidates.append((move.uci(), describe_features(features), move, features))
                    candidate_count = len(candidates)

                    # Naive greedy baseline: argmax over the exact same features
                    # handed to Jev. This is the cheapest way to check whether
                    # Jev is doing anything beyond sorting by the obvious number.
                    greedy_id, _, _, greedy_features = max(candidates, key=lambda c: greedy_baseline_score(c[3]))

                    ui.update(
                        status=f"Calling Jev for ply #{ply_no} ({side_label} to move, {candidate_count} legal moves)...",
                        grid=board_grid(board, None),
                        ply_no=ply_no,
                        side_to_move=side_label.lower(),
                        candidate_count=candidate_count,
                    )

                    capture = {}
                    call_start = time.monotonic()
                    try:
                        result = call_jev(base_url, access_key, args.model, board, candidates, capture=capture)
                    except requests.RequestException as e:
                        ui.update(
                            last_call={
                                "endpoint": capture.get("endpoint"),
                                "request": capture.get("request"),
                                "error": str(e),
                            }
                        )
                        finish_run(f"Jev call failed: {e}")
                        return
                    latency_ms = (time.monotonic() - call_start) * 1000

                    if not is_active(run_id):
                        return

                    answers = result.get("answers", {})
                    chosen_id = answers.get("best_move", {}).get("choice")
                    chosen_move = None
                    for move_id, _, move, _ in candidates:
                        if move_id == chosen_id:
                            chosen_move = move
                            break

                    # Only compare to the baseline when Jev actually returned a
                    # valid candidate -- not when we're about to fall back.
                    if chosen_move is not None:
                        baseline_total += 1
                        if chosen_id == greedy_id:
                            baseline_agreement = "agree"
                            baseline_alt_san = None
                        else:
                            baseline_agreement = "diverge"
                            baseline_alt_san = greedy_features["san"]
                            baseline_divergent += 1
                    else:
                        # Fall back defensively: prefer captures, then any legal move.
                        chosen_move = legal_moves[0]
                        for move in legal_moves:
                            if board.is_capture(move):
                                chosen_move = move
                                break
                        baseline_agreement = None
                        baseline_alt_san = None

                    sharp = answers.get("is_sharp", {}) or {}
                    sharp_noul = sharp.get("noul")
                    last_call_update = {
                        "last_call": {
                            "endpoint": capture.get("endpoint"),
                            "request": capture.get("request"),
                            "response": result,
                            "latency_ms": round(latency_ms),
                        }
                    }

                san = board.san(chosen_move)
                board.push(chosen_move)
                node = node.add_variation(chosen_move)

                baseline_divergence_pct = (
                    (baseline_divergent / baseline_total * 100) if baseline_total else None
                )

                ui.update(
                    status=f"Jev played {san} ({side_label})" + ("  -- forced move" if forced else ""),
                    grid=board_grid(board, chosen_move),
                    ply_no=ply_no,
                    side_to_move="white" if board.turn == chess.WHITE else "black",
                    last_move_san=san,
                    forced=forced,
                    material_balance=material_balance(board),
                    sharp_noul=sharp_noul,
                    candidate_count=candidate_count,
                    latency_ms=latency_ms,
                    baseline_agreement=baseline_agreement,
                    baseline_alt_san=baseline_alt_san,
                    baseline_divergent=baseline_divergent,
                    baseline_total=baseline_total,
                    baseline_divergence_pct=baseline_divergence_pct,
                    **last_call_update,
                )
                baseline_note = (
                    " [forced]" if forced
                    else f" [baseline diverges, would play {baseline_alt_san}]" if baseline_agreement == "diverge"
                    else " [matches baseline]" if baseline_agreement == "agree"
                    else ""
                )
                print(
                    f"ply #{ply_no} ({side_label}): played {san}"
                    + (f", {latency_ms:.0f} ms" if latency_ms is not None else " (forced, no Jev call)")
                    + baseline_note
                )

                reason = result_reason(board)
                if reason:
                    ui.update(status=reason, result=reason)
                    finish_run(reason)
                    break

                elapsed = 0.0
                while elapsed < args.delay:
                    if not is_active(run_id):
                        return
                    step = min(0.1, args.delay - elapsed)
                    time.sleep(step)
                    elapsed += step
            else:
                finish_run(f"Done after {args.moves} ply(s).")

            if args.pgn:
                with open(args.pgn, "w") as f:
                    print(game, file=f)
                print(f"PGN written to {args.pgn}")
        except Exception as e:
            finish_run(f"Runtime error: {type(e).__name__}: {e}")
        finally:
            if baseline_total:
                rate = baseline_divergent / baseline_total * 100
                print(
                    f"Baseline divergence this run: {baseline_divergent}/{baseline_total} decisions "
                    f"({rate:.1f}%) -- how often Jev picked something the naive greedy heuristic "
                    f"(argmax over the same features) wouldn't have."
                )

    def start_game():
        access_key = os.environ.get("MODEL_ACCESS_KEY")
        if not access_key:
            ui.update(status="Set MODEL_ACCESS_KEY env var first.")
            return
        with state_lock:
            if state["running"]:
                return
            state["run_id"] += 1
            state["running"] = True
            run_id = state["run_id"]
        ui.update(
            status="Preparing first move...",
            grid=board_grid(chess.Board(), None),
            ply_no=0,
            side_to_move="white",
            last_move_san=None,
            forced=None,
            material_balance=0,
            sharp_noul=None,
            candidate_count=None,
            latency_ms=None,
            last_call=None,
            result=None,
            baseline_agreement=None,
            baseline_alt_san=None,
            baseline_divergent=0,
            baseline_total=0,
            baseline_divergence_pct=None,
            running=True,
        )
        thread = threading.Thread(target=play_loop, args=(run_id,), daemon=True)
        thread.start()

    ui.set_start_handler(start_game)
    ui.set_end_handler(end_game)
    ui.run_forever()


if __name__ == "__main__":
    main()
