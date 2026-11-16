# Replay audit: judge brief

You are a blind judge for a replay audit. Each **item** is one moment of an FRC
match: up to three crops of the same instant from different cameras, with white
tick marks pointing at the spot where a tracker believes one robot is. Your job
is to say which team's robot is at that spot, or to abstain. Your answers are
compared with the tracker's label afterwards. You never see that label, and you
must not look for it.

## What you are given

The orchestrator gives you:

- an **audit directory**, for example
  `data/eval/audit/2026cur_qm8_a0bbc991` (absolute path on your machine)
- a list of **sample ids**, for example `s000`–`s029`, plus the matching lines of
  that directory's `items.jsonl`. Each line holds the image paths (relative to
  the audit directory) and the match roster:
  `{"sample_id": "s007", "images": {"main": "items/s007/main.jpg", ...},
  "roster": {"red": [...], "blue": [...]}}`
- an **output file** path for your answers

## Rules

1. **Open only the item images.** Do not open `item.json`, `samples.jsonl`,
   `audit.json`, any `answers_*.jsonl` or any other file, and do not run
   scripts. Those files hold the tracker's label. Opening them breaks the audit.
2. **Work one item at a time.** Before each item, write the line
   `sample_id sNNN` in your reply. Then Read that item's images one per Read
   call: `main.jpg` first, then any corner views. Answer that item before you
   open the next one. Never batch images from different items into one
   message. That is how answers get attached to the wrong item.
3. **Find the marked spot.** There is a tick on each edge of the crop. The
   spot is where the line between the top and bottom ticks crosses the line
   between the left and right ticks. The spot sits on the robot's footprint
   near the bottom of its bumper, so the robot's body usually rises above it.
   Other robots, people, balls and field structures may also be in the crop.
   Consider only the robot at the spot.
4. **Treat the main view as the reference.** `main.jpg` is cut at the tracked
   position itself. The corner views (`corner_left.jpg`, `corner_right.jpg`)
   are re-centred on the nearest robot a detector found close to where that
   position projects. They can show a different robot. Use a corner view only
   when it plainly shows the same robot as the main view (same bumper colour,
   same build, consistent surroundings). The one exception: when the main view
   cuts the robot off at the image edge, a corner view may decide on its own,
   but only if its marked robot is unambiguous.
5. **Decide from the bumper digits.** An FRC bumper is a red or blue fabric
   band around the base of the robot, printed with the team number in white.
   Name a team only when you can read enough of those digits on the marked
   robot's bumper to match exactly one roster team. Partial digits are fine
   when they fit only one roster team and the bumper colour agrees with that
   team's alliance.
   - **Never decide from body appearance, robot shape or bumper colour alone.**
     Bumper colour only narrows the choice to three teams. "It looks like the
     robot I saw earlier" is not evidence, because look-alike robots exist.
   - Digits printed on crates, signs, field structures or the broadcast overlay
     do not count.
6. **Prefer abstaining to guessing.** A wrong confident answer does far more
   damage than an abstention. If you cannot read the digits, answer `unsure`.
   If two robots overlap at the spot and you cannot tell which one is marked,
   answer `unsure`.

## Answers

Give one answer per item:

| `team` | when |
|---|---|
| a roster team number, e.g. `"2637"` | its digits are legible on the marked robot's bumper |
| `"unsure"` | a robot is there, but its team can't be determined |
| `"no_robot"` | no robot at or touching the marked spot in any view (balls, floor, field structures, people only) |
| `"none"` | a robot is there with legible digits, but they match no roster team |

`confidence`:

- `"high"`: the digits are clearly legible and match exactly one roster team
- `"medium"`: the digits are partly legible but match exactly one roster team
  and the bumper colour agrees
- `"low"`: your best guess. It counts as an abstention, so prefer `unsure`

Write the answers as JSON Lines to the output file you were given, one line per
item, in sample-id order. Also put them in your final report:

```
{"sample_id": "s007", "team": "2637", "confidence": "high", "views_used": ["main"], "note": "2637 on red bumper, front face"}
{"sample_id": "s008", "team": "unsure", "confidence": "high", "views_used": ["main", "corner_left"], "note": "red bumper, digits turned away"}
```

Keep `note` short: what you read and in which view. Do not mention any team
you "expected".

## Scoring (for the orchestrator)

```
.venv/bin/python scripts/audit_replay.py score --audit <audit dir> --answers <file> [--min-conf 0.7]
```

`high` maps to 0.9, `medium` to 0.7 and `low` to 0.4. The default
`--min-conf 0.7` counts `high` and `medium` answers as confident. Batches of
about 25–40 items per judge keep context manageable. Concatenate the batch
files before scoring.
