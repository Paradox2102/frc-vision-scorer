#!/bin/bash
echo "Testing FRC Vision Scorer Server..."
echo ""

# Test status endpoint
echo "1. Testing /api/status endpoint:"
curl -s http://localhost:8004/api/status | python3 -m json.tool
echo ""

# Test HTML serving
echo "2. Testing / endpoint (HTML):"
curl -s http://localhost:8004/ | grep -o "<title>.*</title>"
echo ""

# Check if server is running
echo "3. Server status:"
if curl -s http://localhost:8004/api/status > /dev/null; then
    echo "✓ Server is running on port 8004"
else
    echo "✗ Server is not responding"
fi
echo ""

# Model file check
echo "4. Model file:"
if [ -f "runs/train/hub_zone_yolo26s/weights/best.pt" ]; then
    echo "✓ YOLO26s model exists"
    ls -lh runs/train/hub_zone_yolo26s/weights/best.pt
else
    echo "✗ Model file not found"
fi
echo ""

# Test video check
echo "5. Test videos:"
ls -lh data/videos/2026/week0_f1m*.mp4 2>/dev/null || echo "Test videos not found"
echo ""

echo "Setup complete! Access at: http://localhost:8004"
