Good context gathered. Here's the picture: you have full 6-DOF camera pose (not just a floor homography), full 3D CAD wireframes for hubs/bumps/towers/trenches, per-epoch calibration with drift detection, ray-casting to arbitrary z-planes, and multi-camera ray-gated fusion. Today that geometry is spent almost entirely on *defense* — filtering bad detections, blocking bad seeds, vetoing bad merges. The biggest untapped value is using it *offensively*: to generate metrics and to predict things before they go wrong.

Here are the ideas, grouped, with the ones I think are highest-value for your current pain points marked ★.

## Tracking accuracy (attacks your known failure modes)

**★ 1. Ray-consistency check on identity votes.** Your memory notes misattributed votes (track↔object switches) are the next layer down in identity binding. Before counting a bumper vote, cast the ray through the crop's bottom-center and check it passes within tolerance of the track's *fused* field position. A vote whose ray lands 2m from the track it's credited to is a misattribution — discard it. This is cheap (you already have `image_to_field()`) and directly attacks the current bottleneck.

**★ 2. Expected-size prior as a continuous hijack detector.** `_expected_robot_pixel_width()` already forward-projects max robot width at a field position — but only at detection-filter time. Run it per-frame against each SAM2 mask: a track whose mask area drifts to 3× (or 0.3×) the geometrically-expected size at its Kalman position has almost certainly latched onto a structure or merged with a pile. This catches equipment hijacks *geometrically*, independent of YOLO corroboration — which matters because your memory says equipment hijacks are detector-corroborated, so evidence-based culls can't win. Geometry is the one signal YOLO false-fires can't pollute.

**★ 3. Autonomous starting-zone seed priors.** Legal starting positions are tightly constrained by the game manual (touching alliance wall / specific tarmac zones). Project those zones into each camera and run aggressive, low-threshold seeding *only there, only at t=0*. This directly attacks "fast robot never seeded" — you don't need to catch a sprinting robot mid-field if you grab it while it's parked at the wall.

**4. Occlusion prediction and handoff portals.** You currently detect occlusion reactively (2D shadow test → coast). Instead, intersect the track's velocity vector with the hub's shadow polygon to predict *where and when* it will re-emerge, and pre-arm a re-acquisition window at the exit point. Also a hijack veto: a track that "passes through" a hub prism without the predicted occlusion gap is suspect.

**5. Kinematic plausibility envelope.** FRC robots top out around 5–6 m/s and ~4 m/s² sustained. A per-frame field-space accel/velocity gate on track updates is a hijack tripwire that needs no appearance model at all — a track jumping 3m in 2 frames didn't move, it got stolen.

## Robot performance metrics (new scouting product)

**★ 6. Climb detection and level scoring from tower rungs.** You already model rungs at 27"/45"/63" in `overlay_geometry.py`. Project the rung lines into the image; when a track enters the tower zone and its mask's top/bottom crosses rung height lines, you get climb attempt, achieved level, and climb duration — a scoring event scouts care about a lot, derived almost entirely from geometry you already have.

**★ 7. Cycle detection and cycle-time.** A small state machine over field-space zones (loading zone → transit → scoring zone → repeat) gives cycles/match, mean cycle time, and cycle-time trend within a match. This is *the* canonical scouting metric and it falls out of positions you already produce. Pairs naturally with Manta reconciliation — cycle counts are a great cross-check against human-scouted scoring counts.

**8. Defense metrics.** Time-within-1.5m of an opposing robot, sustained-contact events (converging tracks + mutual deceleration), pinning near known wall geometry, and time-in-opponent-half → classifies defense bots automatically and quantifies how much defense a team ate.

**9. Route choice and traffic analysis.** With trench/bump geometry: does a robot go over the bump or through the trench? How much time does each route cost per crossing? Which robots yield vs. contest in traffic? These are drivetrain/driver-skill signals nobody scouts by hand.

**10. Shot-position map.** At each scoring event (VLM-confirmed or Manta-reconciled), record distance and bearing from the hub. Over an event you get a shooting-range map per team: fixed-position shooter vs. anywhere-on-the-field shooter is a huge alliance-selection signal.

**11. Speed/accel profiles per team.** Max sustained speed, cross-field transit time, acceleration out of stops — a drivetrain capability fingerprint from the Kalman states you already compute.

**12. Quantitative auton path comparison.** You overlay Manta scouted paths; upgrade to a DTW/Fréchet distance between your tracked auton trajectory and the scouted one. Gives both a pipeline-accuracy metric and detection of teams running a *different* auton than scouted.

## Pipeline health & ops

**13. Continuous drift monitoring.** Drift is checked only at match start; a camera bump at t=60s silently poisons the rest of the match. Cheap version: re-run the AprilTag alignment check every N seconds on frames you're already decoding, and flag (or stage an autofix) mid-match.

**14. Camera coverage/blind-spot analysis.** Offline: rasterize the field into cells, and for each cell compute how many cameras see it unoccluded (hub prisms + frustums + your corner-weight falloff). Produces a coverage heatmap that explains failures like the 6800-never-seeded case and tells you where to point the next camera — turning camera placement from folklore into a computed answer.

`★ Insight ─────────────────────────────────────`
- The common thread in the tracking ideas: geometry is the only evidence channel in your stack that *can't be fooled by appearance*. YOLO false-fires on equipment, SAM2 drifts onto textures, VLM misreads bumpers — but a mask that's 3× too big for its field position, or a vote-ray that misses its track by 2m, is wrong no matter what it looks like. That's why geometric gates compose so well with your existing evidence-tier system: they're independent failure modes.
- The metrics ideas are nearly free: they consume the tracker's *output* (field-space positions you already trust enough to render on radar) rather than touching the fragile detection/tracking layers. Low risk, high scouting value.
`─────────────────────────────────────────────────`

If I were picking a next move: **#1 (vote ray-consistency)** and **#2 (size-prior hijack detector)** are small, attack your two named bottlenecks, and slot into existing code paths — while **#6/#7 (climb + cycles)** are the biggest new user-facing value. Want to dig into one of these and shape it into a design, or should I rough out feasibility/effort estimates across the shortlist first?