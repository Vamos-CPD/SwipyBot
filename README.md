
# SwipyBot

Advanced FaceSwap Telegram Bot for images and videos.

## Features
- Swap faces in images.
- Swap faces in videos.
- No limitations on file size (subject to Telegram's limits).

## Commands
- `/start`: Start the bot.
- `/swipFace`: Initiate the face-swapping process.

## Setup
1. Install dependencies: `pip install insightface onnxruntime opencv-python-headless python-telegram-bot tqdm requests moviepy`
2. Download the `inswapper_128.onnx` model and place it in the `models/` directory.
3. Run the bot: `python bot.py`
