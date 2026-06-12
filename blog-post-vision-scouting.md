I really enjoy working on developing our scouting application, it's grown in scope of the past 2 seasons and has basically become a full competition app, not just scouting, and we have no intention of abandoning it.... However, if I'm honest, it's hard to motivate team members to input good data, and we just don't have the more disciplined culture of other teams that have armies of well trained and motivated scouters, especially for a game this difficult to scout. So what's the alternative?

## [color=#FFC107] Automated Vision Scouting [/color] [color=#D32F2F] - Very Much In Progress[/color]

I've been working on our own AI vision scouting app with the help of an alumni parent from 2016-2019 that is a machine learning data scientist at UCSD, and there's just so much to consider to make it usable. It's nowhere near done, but it IS at a place where I'd like to talk about it.

I haven't gotten to fuel *attribution* yet because robot identity had to come first — tracking fuel doesn't do you any good if you can't attribute it properly. Identity has gotten a lot better recently (on my test matches it now positively identifies 5 or 6 of the 6 robots), but it took a LOT of machinery to get there, which I'll get into. I'm developing on my own 3090 (24GB of vram) so I don't spend a fortune, but then at an actual competition I'd likely spend some $ on runpod and rent out a good GPU for a couple days to run it fast and set it to queue up matches to process as soon as the match video is available. Probably like an H100 for $2.50/hr.

I was heavily inspired by this project about tracking basketball players in a game:

https://www.reddit.com/r/LocalLLaMA/comments/1pes3pu/basketball_ai_with_rfdetr_sam2_and_smolvlm2/
https://www.youtube.com/watch?v=yGQb9KkvQ1Q
https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/basketball-ai-how-to-detect-track-and-identify-basketball-players.ipynb#scrollTo=HDS4Ug9gN4or

As well as [1148's Project Vector](https://www.chiefdelphi.com/t/1148-project-vector-ai-scouting-for-rebuilt/519559)

## [color=#1976d2] Our Basic Requirements [/color]
* no human scouters identifying robot images
* no special equipment at events (use event stream)
* universal to any event without custom training/dataset per event (like taking photos of each robot to train on)
* relatively adaptable to future seasons
* doesn't need to be realtime, but can't take longer than 15 minutes per match (so it can catch up during resets and lunch breaks), so it can be ready for the picklist meeting, and have mostly up to date info for match strategy
* Initial testing can be slower and running on my own hardware, but during an event can run on beefier rented GPUs.

## [color=#D32F2F] Calibration [/color]

First of all, I wanted to not require our own cameras at an event, requiring us to get equipment, get in consistent locations in the stands, etc... but use the TBA match video YouTube link. I set the event I'm scouting, and I calibrate for it, since the cameras are the same, but the viewing angles are NOT (and sometimes they get moved mid event). I'm actually using the april tags and the field geometry to map the lens distortion, position, and angle. Most april tags are too small to actually read the data, so I manually click on them and then on the tag location on a 2D map. Since the cameras DO get bumped or re-zoomed mid-event, every match now also runs an automatic drift check against the calibration — if the camera moved more than half a meter's worth, it refuses to run rather than produce garbage positions.

![UI for mapping April Tags on the Main Cam view for calibration|583x500](upload://fYeMoc8drGHePZ2scDeBxg0NxQH.jpeg)
![Main Cam after mapping april tags and solving for the position, rotation, and lens|690x274](upload://nOWd2AK68EeoMKtFOgIOCSgthUB.jpeg)
![Corner Cam after mapping april tags and solving for the position, rotation, and lens|690x288](upload://mWCdNOUp69h73zTwoTQs2y29XQJ.jpeg)

Once identity/tracking is as good as I want it to be, this lets me map locations to a 2D top down view. Let me do things like view a robot's auto path or a heatmap of where they like to drive. More importantly, though, is that it allows me to combine the main view data with the corner camera data and know that it's looking at the same field, just from three viewpoints.

![2D map of the field|428x246](upload://aKs1gDjs3P0xDKsTZ5YdAnANAlY.png)

Since it knows the physical bounds of the field, I can also tell it to ignore things that are in the view, but outside the field, like robots sitting in queue that are visible or weird things in the audience that somehow detect as a robot. I also draw zones on the map for static field structures (FMS radio stands, the hubs) so the tracker is never allowed to *start* tracking a "robot" there — that killed a whole category of false positives where a radio on a pole would get adopted as a robot for the entire match.

After that, I go to the list of matches for the set event, and pick one. It pulls TBA data about the robots in the match, and the final score, and critically, the youtube link for the matches. I pick a match out, and if I haven't done it for this match, it has me draw a box around the red score and blue score at the top, and the clock. The UI is consistent right now, but I did this in case Chezy Arena changes the UI, and I can adapt easily, and I can more easily adapt for next season. I do a quick pass on the timer box and segment out auto and teleop (plus 3 seconds after 0), and ignore anything outside of that so I'm not processing video frames I don't need to.

![Drawing boxes on the UI|690x299](upload://15NpolCkf6ZO8Wsc8ipXu0FzFNq.jpeg)

## [color=#FFC107] Robot Detection [/color]

For the actual video analysis, it's actually three Yolo (You Only Look Once) models now. One is trained specifically on red bumpers and blue bumpers, with hard-negative training for things that kept getting detected as robots that are on the field (or appear to be on the field): scoring tablets, FMS radios (the ones on the poles), as well as a generic human model (because refs/scorers on the side of the field obscure the field view). The other is a field model that detects fuel, robots in general, and the hub openings.

The bumper model boxes around bumpers (and avoids all the stuff I don't want tracked) and contributes to the alliance color. If the alliance can't be determined yet, it at least tags it as a robot and keeps tracking it until it can determine the alliance. On top of the model itself there's now a stack of runtime sanity checks I had to add one painful false-positive at a time: detections that overlap a person get rejected, boxes that are implausibly wide for their distance from the camera get rejected, and masks that are tall and top-heavy (the silhouette of a radio on a stand, basically) get rejected — while being careful not to reject a robot that's climbing.

## [color=#1976d2] Robot Tracking [/color]

The yolo boxes are used as a hint to a segmentation tracker that masks out the entire robot, which I find better than yolo at overlapping robots and partially obscured robots, and they augment each other's weak points. Yeah, it makes it much slower. The main thing I like about it is that it's temporally aware, as opposed to yolo that is purely frame by frame — if a robot gets occluded for a few seconds, the tracker carries the same identity through it instead of "discovering" a brand new robot on the other side. Yolo re-prompts the tracker every 2 seconds, so anything it missed initially gets picked up quickly.

I started on SAM2, but I've recently switched the default to SAM3. SAM2 wanted to hold the whole video's state in VRAM, which doesn't fit a full match on 24GB, so the SAM3 backend processes the match in chunks of a few hundred frames and re-seeds every live robot across each chunk boundary so identities survive. Compared to SAM2 it confirmed more robots on my test matches at about half the wall-clock time, which matters a lot for that 15-minutes-per-match budget.

There's also now a backward-confirmation pass: after the match finishes processing, it re-resolves identity over the whole timeline with hindsight, so a robot that didn't get positively identified until 30 seconds in still gets correctly attributed for the time *before* that.

![Yolo is the boxes, the SAM3 tracker mask is over the robot's entire silhouette |441x291](upload://xahIwqhZUubi4qmtDDpNPsYBZVU.jpeg)

## [color=#D32F2F] Robot Identity [/color]

Every so often, I send a cropped image of a robot's bumpers to a small-ish VLM (Qwen3-VL-4B — it can also run behind a vLLM server for better batching). Since it knows the 3 robots on each alliance, and the yolo model contributes to the alliance color, there's a whole voting/confidence system that tries to identify a robot. It also knows that two robots can't be positively identified as the same robot at the same time, so there's a whole assignment step about which one it's *more* confident about.

The locking logic took a while. A team "locks" after 3 consecutive agreeing reads, but the lock is never permanent — 2 consecutive contradicting reads will flip it. That sounds backwards (why make it easy to un-lock?) but I learned the hard way that a confidently *wrong* lock is way worse than an unconfident one. An early misread would freeze forever and a robot would spend the whole match labeled as a team that wasn't even in the match. For the same reason, even "confirmed" robots get periodically re-audited with fresh reads instead of being trusted forever.

Two other things that I added:
* **An appearance gallery.** Every positively-identified robot deposits a visual embedding (DINOv2) into a per-event gallery, so later in the event, a robot whose bumper is obscured can still get a vote based on what it *looks like*. It's a supplementary vote, never an override — appearance alone is not allowed to flip an identity. I do have to go in here and trim out bad images every so often, so it's not without human input.
* **Not wasting reads.** OCR is one of the most GPU/time intensive parts of the process, so it now skips reads when the bumper is occluded in every camera, and skips re-reading a robot that hasn't moved since the last read. That cut reads substantially with no identity loss.

![Preview of bumper crops that are being analyzed|341x500](upload://ukO5X4ZnVPlaFx5BToP449wIJ5B.png)

## [color=#FFC107] The Limitations [/color]

I had tried using bumper color values with HSV previously, but it was too finnicky, didn't like dark blue bumpers, and was too prone to seeing painted robot parts (like 254 when on red alliance) and get the alliance wrong. The yolo model trained specifically on red and blue bumpers does much better in that regard.

Identity is where most of the iteration has gone, and while it's much better than it was, it's still the fragile part. First of all, like a fifth of bumper fonts are just unreadable (including my own team's number) because they're too skinny or the kerning is too tight at this distance. The resolution isn't good enough to consistently read bumpers at this distance. Bumper numbers being so low means basically everything, including balls, obscure them easily. If an event has the camera up high, looking down at a steeper angle, you get better positional data, but it's nearly impossible to read the bumpers at that angle.

I also learned a deflating lesson about appearance matching, that visual embeddings are great at telling a red robot from a blue robot, and nearly useless at telling two robots on the *same* alliance apart (especially in a game where everyone is a box). From a wide camera, robots-with-red-bumpers just look alike. So the bumper number is, in the end, the only signal that actually discriminates individual robots, and everything else can only support it.

![image|690x118](upload://xBRLB5T63IRtl3uVVVtk3078ePd.png)


So it's good at detecting **a** robot and tracking its movement, and now usually right about *which* robot — on my test matches it confirms 5-6 of the 6, but there's run-to-run variance and I don't fully trust it at a new venue yet. For a seemingly open field, there's still a lot of things in the way. The corner cameras are calibrated and fused in now, but the massive event name label at the bottom still eats a big chunk of their view. There's also a human-correction path: if I spot a wrong identity, my correction gets stored as a verified example in the appearance gallery and outvotes the automatic ones — so a few minutes of human spot-checking per event improves every later match, without going back to full-time scouter tagging.

Full match export video:
https://drive.google.com/file/d/1wU0_bZa8YgWeSwRXxKaRGGDIMItkVwQS/view?usp=sharing

## [color=#FFC107] Fuel Tracking [/color]

Funny enough, SAM3 was originally my plan for fuel tracking (it's better at tracking hundreds of little objects) and instead it ended up becoming the robot tracker first. Fuel *detection* is actually already in — the field model picks up fuel balls every frame — but I haven't built the attribution part yet. The plan there is still pretty simple: I don't really need to track each fuel making it in, I just need to track if fuel is leaving a robot (and ideally what volume of fuel) and then count score going up and correlate. Yes, it won't be perfect scoring when two robots are shooting at the same time, and one is accurate and the other is missing, but it's better than nothing.

My thought is that fuel tracking will only really be useful for this off-season, but robot detection/tracking/identity is universal to all future seasons, so that is my focus.

What do you think? Any suggestions or criticisms? How are you trying to tackle this problem?
