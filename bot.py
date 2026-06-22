
import logging
import os
import asyncio
import cv2
import numpy as np
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from insightface.app import FaceAnalysis
from insightface.model_zoo import model_zoo
import subprocess

# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# Bot token from the user
BOT_TOKEN = "8955338935:AAGpPmF7HLOZCtiCUD-ZP96UpYPeTEt5cz0"

# Model paths
MODEL_DIR = "/home/ubuntu/SwipyBot/models"
INSWAFFER_MODEL_PATH = os.path.join(MODEL_DIR, "inswapper_128.onnx")

# Initialize FaceAnalysis and FaceSwapper
app = FaceAnalysis(name="buffalo_l", root=MODEL_DIR)
app.prepare(ctx_id=-1, det_size=(640, 640)) # Use CPU (ctx_id=-1)

swapper = model_zoo.get_model(INSWAFFER_MODEL_PATH, download=False, download_zip=False)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /start is issued."""
    user = update.effective_user
    context.user_data.clear() # Clear previous state
    await update.message.reply_html(
        f"Hi {user.mention_html()}!\nI am a FaceSwap bot. "
        "Please send me an **image** to use as the **source face**."
    )

async def handle_media(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle incoming media (photos and videos) for face swapping."""
    user_data = context.user_data
    chat_id = update.effective_chat.id

    if "source_face" not in user_data:
        # State 1: Waiting for source face
        if update.message.photo:
            file_id = update.message.photo[-1].file_id
            file = await context.bot.get_file(file_id)
            source_path = f"/tmp/{file_id}.jpg"
            await file.download_to_drive(source_path)
            user_data["source_face"] = source_path
            await update.message.reply_text("Source face received. Now send the **target image or video** to swap the face onto.")
        else:
            await update.message.reply_text("Please send an **image** for the source face.")
        return

    # State 2: Source face received, waiting for target media
    source_face_path = user_data["source_face"]
    source_face_img = cv2.imread(source_face_path)
    source_faces = app.get(source_face_img)

    if not source_faces:
        await update.message.reply_text("Could not detect a face in the source image. Please try again with a clearer image.")
        if os.path.exists(source_face_path):
            os.remove(source_face_path)
        del user_data["source_face"]
        return
    
    source_face = source_faces[0]

    if update.message.photo:
        file_id = update.message.photo[-1].file_id
        file = await context.bot.get_file(file_id)
        target_path = f"/tmp/{file_id}.jpg"
        await update.message.reply_text("Downloading target image...")
        await file.download_to_drive(target_path)
        await update.message.reply_text("Processing image face swap...")
        output_path = await process_image_swap(source_face, target_path)
        if output_path and os.path.exists(output_path):
            await update.message.reply_document(document=output_path)
            os.remove(output_path)
        else:
            await update.message.reply_text("Failed to process image face swap.")
        if os.path.exists(target_path):
            os.remove(target_path)

    elif update.message.video:
        file_id = update.message.video.file_id
        file = await context.bot.get_file(file_id)
        target_path = f"/tmp/{file_id}.mp4"
        await update.message.reply_text("Downloading target video...")
        await file.download_to_drive(target_path)
        progress_message = await update.message.reply_text("Processing video face swap... This may take a while.")
        output_path = await process_video_swap(source_face, target_path, progress_message, context.bot, chat_id)
        if output_path and os.path.exists(output_path):
            await context.bot.edit_message_text(chat_id=chat_id, message_id=progress_message.message_id, text="Video face swap complete! Sending result...")
            await update.message.reply_video(video=output_path)
            os.remove(output_path)
        else:
            await context.bot.edit_message_text(chat_id=chat_id, message_id=progress_message.message_id, text="Failed to process video face swap.")
        if os.path.exists(target_path):
            os.remove(target_path)
    else:
        await update.message.reply_text("Please send a valid **image or video** for the target.")
    
    # Clean up source face
    if os.path.exists(source_face_path):
        os.remove(source_face_path)
    del user_data["source_face"]

async def process_image_swap(source_face, target_image_path: str) -> str | None:
    """Performs face swap on an image."""
    target_img = cv2.imread(target_image_path)
    if target_img is None:
        logger.error(f"Could not read target image: {target_image_path}")
        return None

    target_faces = app.get(target_img)
    if not target_faces:
        logger.warning(f"No faces detected in target image: {target_image_path}")
        return None

    result_img = swapper.get(target_img, target_faces[0], source_face, paste_back=True)
    output_path = target_image_path.replace(".jpg", "_swapped.jpg")
    cv2.imwrite(output_path, result_img)
    return output_path

async def process_video_swap(source_face, target_video_path: str, progress_message, bot, chat_id) -> str | None:
    """Performs face swap on a video using a more memory-efficient frame-by-frame approach."""
    output_path = target_video_path.replace(".mp4", "_swapped.mp4")
    temp_video_no_audio = target_video_path.replace(".mp4", "_temp_no_audio.mp4")
    
    cap = cv2.VideoCapture(target_video_path)
    if not cap.isOpened():
        logger.error(f"Could not open target video: {target_video_path}")
        return None
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Use mp4v codec for broader compatibility
    out = cv2.VideoWriter(temp_video_no_audio, fourcc, fps, (width, height))
    
    if not out.isOpened():
        logger.error(f"Could not open video writer for {temp_video_no_audio}")
        cap.release()
        return None

    try:
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            target_faces = app.get(frame)
            if target_faces:
                # Sort faces by size (area) to swap the largest one
                target_faces = sorted(target_faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)
                frame = swapper.get(frame, target_faces[0], source_face, paste_back=True)
            
            out.write(frame)
            frame_count += 1
            if frame_count % 50 == 0:
                progress_percent = int((frame_count / total_frames) * 100)
                await bot.edit_message_text(chat_id=chat_id, message_id=progress_message.message_id, text=f"Processing video face swap... {progress_percent}% complete.")
    except Exception as e:
        logger.error(f"Error during frame processing: {e}")
        return None
    finally:
        cap.release()
        out.release()
    
    # Use FFmpeg to merge original audio with swapped video
    try:
        # Check if the original video has an audio track
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'a', '-show_entries', 'stream=codec_type', '-of', 'csv=p=0', target_video_path],
            capture_output=True, text=True, check=True
        )
        has_audio = bool(result.stdout.strip())

        if has_audio:
            logger.info("Merging video with original audio.")
            subprocess.run([
                'ffmpeg', '-y', '-i', temp_video_no_audio, '-i', target_video_path,
                '-map', '0:v', '-map', '1:a', '-c:v', 'libx264', '-c:a', 'aac',
                '-shortest', output_path
            ], check=True)
        else:
            logger.info("Original video has no audio, skipping audio merge.")
            os.rename(temp_video_no_audio, output_path)

    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg or FFprobe error: {e.stderr}")
        # If FFmpeg fails, just rename the video without audio
        os.rename(temp_video_no_audio, output_path)
    except Exception as e:
        logger.error(f"Error during FFmpeg audio merge: {e}")
        os.rename(temp_video_no_audio, output_path)
    finally:
        if os.path.exists(temp_video_no_audio):
            os.remove(temp_video_no_audio)
        
    return output_path

def main() -> None:
    """Start the bot."""
    application = Application.builder().token(BOT_TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.PHOTO | filters.VIDEO, handle_media))

    application.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()
