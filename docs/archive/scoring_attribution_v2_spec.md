# Scoring Attribution v2 - Scoreboard-Driven Approach

## Overview
The current scoring attribution system tracks ball trajectories to determine which robot scored. This approach has limitations:
- Balls are small and fast (often visible for only 1-2 frames)
- Ball tracking is unreliable for attribution
- ~90% of ball detections are ground balls (not in flight)

**New approach:** Use scoreboard OCR to detect WHEN scores happen, and robot proximity to hubs to determine WHO scored.

## Implementation Plan

### 1. Scoreboard OCR System
**Current state:** Team numbers are extracted once from an early frame  
**Needed:** Frame-by-frame score extraction

#### Scoreboard Layout (FRC 2022 matches)
- Top ~33% of full frame contains scoreboard
- Alliance scores displayed as numbers (e.g., "24" for blue, "18" for red)
- Score positions are consistent across frames

#### Implementation Steps
```python
def extract_scores(self, frame: np.ndarray) -> Dict[str, int]:
    """Extract alliance scores from scoreboard region.
    
    Returns: {"red": int, "blue": int} or None if OCR fails
    """
    # Define scoreboard regions (y coordinates in full 720p frame)
    # Blue score: approximately x=XXX-XXX, y=YYY-YYY
    # Red score: approximately x=XXX-XXX, y=YYY-YYY
    
    # Extract score regions
    # Apply preprocessing (grayscale, threshold, resize 4x)
    # Run pytesseract with --psm 7 (single line)
    # Parse digits only
    pass
```

#### Score Change Detection
```python
def detect_score_change(self, current_scores: Dict[str, int], 
                       prev_scores: Dict[str, int]) -> Optional[str]:
    """Detect if an alliance scored.
    
    Returns: "red" or "blue" if score increased, None otherwise
    """
    if current_scores["red"] > prev_scores["red"]:
        return "red"
    elif current_scores["blue"] > prev_scores["blue"]:
        return "blue"
    return None
```

### 2. Robot Proximity Tracking
When a score change is detected, analyze robot positions from the last ~60 frames (2 seconds at 30fps).

```python
def find_scoring_robot(self, alliance: str, score_frame: int, 
                      window_frames: int = 60) -> Optional[int]:
    """Find which robot of the alliance was closest to hub around scoring time.
    
    Args:
        alliance: "red" or "blue"
        score_frame: Frame index when score was detected
        window_frames: Number of frames to look back
    
    Returns:
        Robot canonical ID, or None if no robot was near hub
    """
    # Get hub position for this alliance
    hub_pos = self.hub_positions[alliance]
    
    # Look back through frames
    start_frame = max(0, score_frame - window_frames)
    
    # Track minimum distance for each robot in this alliance
    robot_min_distances = {}  # robot_id -> (min_distance, frame)
    
    for frame_idx in range(start_frame, score_frame + 1):
        # Find all robots of this alliance active at this frame
        for robot_id in self.canonical_robots[alliance]:
            # Get robot position at this frame (if exists)
            # Calculate distance to hub
            # Track minimum
            pass
    
    # Return robot with minimum distance (if within threshold)
    # Threshold: ~150 pixels = reasonable shooting range
    SCORING_RANGE = 150
    
    if robot_min_distances:
        closest_robot_id = min(robot_min_distances, 
                              key=lambda r: robot_min_distances[r][0])
        min_dist, min_frame = robot_min_distances[closest_robot_id]
        
        if min_dist < SCORING_RANGE:
            return closest_robot_id
    
    return None
```

### 3. Attribution Confidence Levels
- **High confidence:** Robot was within 100px of hub within 30 frames (1 sec) of score
- **Medium confidence:** Robot was within 150px of hub within 60 frames (2 sec)
- **Low confidence:** Robot was within 200px, or only robot near hub
- **Unattributed:** No robot of scoring alliance was near hub

### 4. Integration Points

#### In `process_video()` method:
```python
# After detection and tracking update:
if frame_idx % 5 == 0:  # Check scores every 5 frames (reduce OCR overhead)
    current_scores = self.extract_scores(full_frame)  # Full frame, not cropped
    if current_scores:
        scoring_alliance = self.detect_score_change(current_scores, self.prev_scores)
        if scoring_alliance:
            robot_id = self.find_scoring_robot(scoring_alliance, frame_idx)
            confidence = self.determine_confidence(robot_id, scoring_alliance, frame_idx)
            
            # Create scoring event
            event = ScoringEvent(
                frame=frame_idx,
                timestamp=frame_idx / fps,
                ball_track_id=None,  # Not tracking individual balls anymore
                robot_track_id=robot_id,
                alliance=scoring_alliance,
                attribution_confidence=confidence,
                launch_frame=None
            )
            self.scoring_events.append(event)
            self.alliance_scores[scoring_alliance] = current_scores[scoring_alliance]
            
            if robot_id:
                canon_key = f"{scoring_alliance}_{robot_id}"
                self.robot_scores[canon_key].scores += 1
        
        self.prev_scores = current_scores
```

### 5. Advantages of Scoreboard-Driven Approach
1. **Accurate score counts:** Ground truth from official scoreboard
2. **Handles missed ball detections:** Don't rely on tracking small, fast balls
3. **Robust attribution:** Uses robot proximity, which is more reliable than ball trajectory
4. **Simpler logic:** No complex ball-to-hub trajectory analysis
5. **Matches human intuition:** "Which robot was shooting around that time?"

### 6. Potential Issues & Solutions

#### Issue: OCR failures
**Solution:** Use previous score if OCR fails, log warnings

#### Issue: Multiple robots near hub
**Solution:** Attribute to closest robot, mark as "low confidence"

#### Issue: Scoreboard layout changes between videos
**Solution:** Make scoreboard regions configurable via UI or config file

#### Issue: Simultaneous scoring (rare)
**Solution:** Current frame-skip approach (every 5 frames) reduces likelihood; if detected, attribute to closest robot

### 7. Testing Plan
1. Run on Week 0 test videos with known ground truth
2. Compare ball-trajectory vs scoreboard-driven attribution accuracy
3. Measure OCR reliability (success rate per frame)
4. Tune proximity thresholds and time windows
5. Validate confidence levels align with accuracy

## Status
- [x] Specification documented
- [ ] Implement scoreboard OCR extraction
- [ ] Implement score change detection
- [ ] Implement proximity-based attribution
- [ ] Replace ball-trajectory logic
- [ ] Test and validate
- [ ] Deploy to production

## Priority
**Medium** - Current ball-trajectory approach works but is unreliable. This is a quality improvement, not a critical bug fix.
