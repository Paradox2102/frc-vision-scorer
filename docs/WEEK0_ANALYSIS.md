# Week 0 Finals Analysis (2026-02-23)

## Videos Analyzed
- Finals 1: https://www.youtube.com/watch?v=JPk0U_NXfmo
- Finals 2: https://www.youtube.com/watch?v=tdLupXNtBD4
- Finals 3: https://www.youtube.com/watch?v=8r3-hp36KUc

## TBA Match Results
| Match | Red Teams | Blue Teams | Red Hub | Blue Hub |
|-------|-----------|------------|---------|----------|
| F1M1  | 2342, 6328, 2877 | 1768, 4909, 5962 | 263 (45+218) | 187 (33+154) |
| F1M2  | 2342, 6328, 2877 | 5962, 1768, 4909 | 77 (21+56) | 253 (46+207) |
| F1M3  | 2342, 6328, 2877 | 9644, 1768, 4909 | 208 (38+170) | 185 (36+149) |

## Video Layout
- 1280x720, 30fps, ~172 seconds
- Top ~65%: Overhead elevated field camera (full field)
- Bottom ~35%: Split into two field-level alliance cameras (pillar divider)
- Scoreboard overlay at top: live alliance scores, timer, team numbers

## Key Findings
1. Bottom alliance views: Close but low angle, hub openings obscured, too many ground balls
2. Top overhead view: Good for robot tracking, objects small
3. Scoreboard overlay: Most reliable signal for scoring events (1 ball = 1 point)
4. Field has 50-100+ scattered yellow balls creating visual noise

## Recommended Pipeline
1. **Stage 1**: Scoreboard OCR → exact frame-level scoring events
2. **Stage 2**: YOLO robot detection + ByteTrack on overhead view
3. **Stage 3**: Attribution (score event + robot position → per-robot count)
