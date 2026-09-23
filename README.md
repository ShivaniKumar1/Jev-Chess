# Jev Plays Chess Using DigitalOcean /v1/systemone Endpoint

A demo that has TypeSafe AI's Jev model (via DigitalOcean's `/v1/systemone` endpoint) play chess against itself - one decision per move, not per candidate line.

Jev isn't a generative/chat model - it answers typed questions (`noul` / `choice` / `score`) about a `state` you hand it. So instead of asking it to search chess positions itself, this script does the engine legwork: for the side to move, it enumerates every legal move (never more than 218, comfortably under the `choice` type's 255-option cap), computes simple heuristics for each (material change, captures, checks/mate, opponent mobility), and asks Jev a single `choice` question to pick the best one - plus a `score` (position assessment) and `noul` (tactical sharpness) question in the same call, for narration. One API call per move.

## Setup

```bash
pip install requests python-chess
export MODEL_ACCESS_KEY="sk-do-..."
```

The script always uses DO's Inference URL:

`INFERENCE_PROXY_BASE_URL="https://inference.do-ai.run"`

## Usage

```bash
python3 jev_plays_chess.py --moves 60 --delay 1.0
```

When the script starts, it launches a local browser UI automatically. The UI shows the live board, whose turn it is, the last move played, a material/position/sharpness readout, request latency, and the raw `answers.best_move` from the most recent `/v1/systemone` call.

- Click `Start` to let Jev begin playing both sides.
- Click `End` to stop the current run.

## What the bubbles mean

- `Side to move`: Which side is about to play (`White` or `Black`).
- `Last move`: The move just played (chess notation, for example `Nf3`).
- `Candidates`: Number of legal moves available in the current position.
- `Material`: Piece-value balance (`White - Black`) using pawn=1, knight=3, bishop=3, rook=5, queen=9. Positive means White is ahead; negative means Black is ahead.
- `Latency`: Round-trip time for the most recent Jev API request.
- `Sharpness`: Jev's estimate (0-100%) of how tactical/volatile the position is (how much one move could swing the game).

## Flags

| Flag | Default | Description |
| --- | --- | --- |
| `--moves N` | 60 | Stop after N half-moves (plies). |
| `--delay S` | 1.0 | Minimum seconds to pause between moves, for watchability. |
| `--model NAME` | `typesafe-jev-1.13.0` | Override the model `internal_name`. |
| `--port N` | 8766 | Port for the local web UI. |
