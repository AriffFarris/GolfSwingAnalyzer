# ⛳ Golf Swing Analyzer

An AI-powered golf swing analysis tool that uses computer vision to provide real-time feedback on your swing mechanics.

## Overview

Golf Swing Analyzer uses MediaPipe's pose detection to track your body positions throughout your swing and provides actionable feedback to help improve your technique. Upload a video of your swing and receive detailed analysis of key metrics including arm angles, knee flex, spine angle, and shoulder rotation.

## Features

- **Pose Detection**: Real-time body tracking using MediaPipe
- **Video Annotation**: Visual overlay showing detected pose landmarks
- **Swing Metrics**: Quantitative measurements of key angles and positions
- **Actionable Feedback**: Severity-rated tips for improvement
- **Easy Interface**: Simple drag-and-drop video upload via Streamlit

## What Gets Analyzed

| Metric | Description | Ideal Range |
|--------|-------------|-------------|
| Lead Arm Angle | Extension of the lead arm through swing | >160° |
| Trail Arm Angle | Position of the trail arm | 150-170° |
| Knee Flex | Stability of lead knee | 140-165° |
| Spine Angle | Forward tilt from hips | 25-45° |
| Shoulder Rotation | Turn during backswing | >10° |

## Installation

### Prerequisites

- Python 3.8 or higher
- Webcam or pre-recorded swing videos

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/golf-swing-analyzer.git
   cd golf-swing-analyzer
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Start the application:
   ```bash
   streamlit run app.py
   ```

2. Open your browser to `http://localhost:8501`

3. Upload a video of your golf swing (MP4, MOV, AVI, or MKV)

4. Click **Analyze My Swing** and wait for the results

## Tips for Best Results

- **Full body visible**: Ensure your entire body is in frame from head to feet
- **Side view**: Film from a direct side angle (down the target line or facing you)
- **Good lighting**: Well-lit environments improve pose detection accuracy
- **Steady camera**: Use a tripod or stable surface
- **Short clips**: Keep videos under 30 seconds for faster processing

## Project Structure

```
golf-swing-analyzer/
├── app.py              # Streamlit web application
├── swing_analyzer.py   # Core analysis logic using MediaPipe
├── requirements.txt    # Python dependencies
└── README.md
```

## Requirements

```
streamlit>=1.28.0
mediapipe>=0.10.0
opencv-python>=4.8.0
numpy>=1.24.0
Pillow>=10.0.0
```

## How It Works

1. **Video Processing**: Frames are extracted from your uploaded video at regular intervals
2. **Pose Detection**: MediaPipe identifies 33 body landmarks in each frame
3. **Angle Calculation**: Key joint angles are computed using vector mathematics
4. **Feedback Generation**: Metrics are compared against ideal ranges to generate tips
5. **Visualization**: Annotated video with pose overlay is displayed alongside feedback

## Limitations

- Assumes right-handed golfer (left arm as lead arm)
- Works best with side-view recordings
- Requires full body visibility for accurate detection
- Analysis is frame-by-frame, not swing-phase aware

## Contributing

Contributions are welcome! Some ideas for improvement:

- Add swing phase detection (address, backswing, downswing, impact, follow-through)
- Support left-handed golfer analysis
- Add comparison with professional swing templates
- Implement swing tempo analysis
- Add club head tracking

## License

MIT License - feel free to use and modify for your own projects.

## Acknowledgments

- [MediaPipe](https://mediapipe.dev/) for pose detection
- [Streamlit](https://streamlit.io/) for the web interface
- [OpenCV](https://opencv.org/) for video processing