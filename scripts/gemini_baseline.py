#!/usr/bin/env python3
"""
Test Gemini's video understanding as a baseline for ball counting.
Feed short video clips and ask structured questions about scoring.
"""

import os
import json
import time
from pathlib import Path
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

def analyze_video_clip(video_path: str, prompt: str, model: str = "gemini-2.5-flash"):
    """Upload a video and analyze with Gemini."""
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    
    print(f"Uploading {video_path}...")
    video_file = client.files.upload(file=video_path)
    
    # Wait for processing
    while video_file.state.name == "PROCESSING":
        print("  Processing...")
        time.sleep(2)
        video_file = client.files.get(name=video_file.name)
    
    if video_file.state.name == "FAILED":
        print(f"  Upload failed: {video_file.state}")
        return None
    
    print(f"  File ready. Analyzing with {model}...")
    
    response = client.models.generate_content(
        model=model,
        contents=[
            types.Content(
                role="user",
                parts=[
                    types.Part.from_uri(
                        file_uri=video_file.uri,
                        mime_type=video_file.mime_type,
                    ),
                    types.Part.from_text(text=prompt),
                ]
            )
        ],
        config=types.GenerateContentConfig(
            temperature=0.1,
            max_output_tokens=2000,
        )
    )
    
    # Cleanup
    try:
        client.files.delete(name=video_file.name)
    except:
        pass
    
    return response.text


if __name__ == "__main__":
    base = Path(__file__).parent.parent
    
    # Test with the 58-ball auto video (short, high action)
    video = str(base / "data/videos/2017/58ball_auto.mp4")
    
    if not os.path.exists(video):
        print(f"Video not found: {video}")
        exit(1)
    
    # Prompt 1: General understanding
    print("\n=== Test 1: General Understanding ===")
    result = analyze_video_clip(video, """
    This is a video from the 2017 FIRST Robotics Competition (FRC) game called Steamworks.
    Robots shoot small yellow balls (called "fuel") into a boiler structure to score points.
    
    Please describe:
    1. What do you see happening in this video?
    2. How many robots are visible?
    3. Can you see yellow balls being shot or collected?
    4. Approximately how many yellow balls can you count in any given frame?
    5. Can you identify any team numbers on the robot bumpers?
    """)
    print(result)
    
    # Prompt 2: Scoring analysis
    print("\n=== Test 2: Scoring Analysis ===")
    result = analyze_video_clip(video, """
    This is a 2017 FRC Steamworks match clip. The yellow balls are called "fuel" and robots 
    shoot them into a scoring structure called a "boiler."
    
    Watch the entire video carefully and answer:
    1. How many times do you see a ball enter the boiler/scoring structure?
    2. Which robot(s) appear to be shooting balls?
    3. What is the approximate rate of balls being shot (balls per second)?
    4. Are there any balls that miss the target?
    
    Be as specific and quantitative as possible.
    """)
    print(result)
    
    # Test with full match
    match_video = str(base / "data/videos/2017/midwest_final1.mp4")
    if os.path.exists(match_video):
        print("\n=== Test 3: Full Match Analysis ===")
        result = analyze_video_clip(match_video, """
        This is a full 2017 FRC Steamworks match from the Midwest Regional Finals.
        
        Yellow balls ("fuel") are scored into "boilers" by robots. Each alliance (red and blue)
        has 3 robots and their own boiler.
        
        Please analyze:
        1. Which alliance appears to score more fuel overall?
        2. Can you identify which individual robots are the primary shooters?
        3. Approximately how many total balls were scored by each alliance?
        4. What team numbers can you read on the robot bumpers?
        5. Rate your confidence in these estimates (1-10).
        
        Focus on being quantitative. Give specific numbers where possible.
        """)
        print(result)
