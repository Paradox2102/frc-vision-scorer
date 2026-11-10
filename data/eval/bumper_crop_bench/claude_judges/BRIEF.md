# Blind bumper-crop judge

You are one arm of a reader benchmark for an FRC (FIRST Robotics) match-video scouting app.
Each item is one image crop cut from match video around a robot's bumper, plus the six teams
in that match (red alliance and blue alliance). Your job: say which team's robot this is,
from what you can SEE in the image.

Rules
- Look ONLY at the image files listed in your items file. Do not open any other file in the
  repo (no manifest, no labels, no results) — this is a blind test.
- Open each image with the Read tool. Some crops are tiny (20-80 px); look carefully anyway.
- FRC bumpers are red or blue fabric bands around the robot base with the team number printed
  in white/black digits. The number may be partial, blurred, occluded, mirrored by angle, etc.
- A WRONG team is much worse than abstaining. Only name a team if the digits you can see
  support it. You may use the roster to complete a partly-read number only when the visible
  digits and the bumper colour rule out every other roster team.
- Crops can show non-robot things (game balls, field structures, numbers printed on crates,
  scoring tables). Then answer not_bumper.

Per item, append ONE JSON line to your output file (append as you go, do not batch at the end):
{"id": "<item id>",
 "subject": "robot_bumper" | "not_bumper" | "unsure",
 "alliance": "red" | "blue" | "unknown",
 "digits": "<the digits you actually see, '?' for unreadable positions, or null>",
 "team": "<one of the six roster numbers>" | "none" | "other",
 "confidence": <0.0-1.0, your probability that `team` is correct>}

`none` = no legible number. `other` = a legible number that is not on the roster.
Process every item. When finished, reply with just the count of lines written.
