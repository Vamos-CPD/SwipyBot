
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
from decord import VideoReader, cpu

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

# Define conversation states
SELECTING_SOURCE_IMAGE = 0
SELECTING_TARGET_MEDIA = 1

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /start is issued."""
    user = update.effective_user
    context.user_data.clear() # Clear previous state
    context.user_data["status"] = SELECTING_SOURCE_IMAGE
    await update.message.reply_html(
        f"Hi {user.mention_html()}!\nI am a FaceSwap bot. "
        "Please send me an **image** to use as the **source face**."
    )

async def handle_media(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle incoming media (photos and videos) for face swapping."""
    user_data = context.user_data
    chat_id = update.effective_chat.id
    current_status = user_data.get("status", SELECTING_SOURCE_IMAGE)

    if current_status == SELECTING_SOURCE_IMAGE:
        if update.message.photo:
            file_id = update.message.photo[-1].file_id
            file = await context.bot.get_file(file_id)
            source_path = f"/tmp/{file_id}.jpg"
            await update.message.reply_text("Downloading source image...")
            await file.download_to_drive(source_path)
            user_data["source_face"] = source_path
            user_data["status"] = SELECTING_TARGET_MEDIA
            await update.message.reply_text("Source face received. Now send the **target image or video** to swap the face onto.")
        else:
            await update.message.reply_text("Please send an **image** for the source face.")
        return

    elif current_status == SELECTING_TARGET_MEDIA:
        source_face_path = user_data["source_face"]
        source_face_img = cv2.imread(source_face_path)
        source_faces = app.get(source_face_img)

        if not source_faces:
            await update.message.reply_text("Could not detect a face in the source image. Please try again with a clearer image.")
            if os.path.exists(source_face_path):
                os.remove(source_face_path)
            user_data.clear()
            user_data["status"] = SELECTING_SOURCE_IMAGE
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
            progress_message = await update.message.reply_text("Processing video face swap... 0% complete.")
            
            # Offload heavy processing to an asyncio task
            asyncio.create_task(process_video_swap_and_reply(
                source_face, target_path, progress_message, context.bot, chat_id, user_data
            ))
            user_data["status"] = SELECTING_SOURCE_IMAGE # Reset state while processing
            return # Exit to allow async task to run
        else:
            await update.message.reply_text("Please send a valid **image or video** for the target.")
        
        # Clean up source face and reset state after sync processing
        if os.path.exists(source_face_path):
            os.remove(source_face_path)
        user_data.clear()
        user_data["status"] = SELECTING_SOURCE_IMAGE

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

async def process_video_swap_and_reply(source_face, target_video_path: str, progress_message, bot, chat_id, user_data) -> None:
    """Performs face swap on a video and replies to the user."""
    output_path = None
    try:
        output_path = await _process_video_swap(source_face, target_video_path, progress_message, bot, chat_id)
        if output_path and os.path.exists(output_path):
            await bot.edit_message_text(chat_id=chat_id, message_id=progress_message.message_id, text="Video face swap complete! Sending result...")
            await bot.send_video(chat_id=chat_id, video=output_path)
            os.remove(output_path)
        else:
            await bot.edit_message_text(chat_id=chat_id, message_id=progress_message.message_id, text="Failed to process video face swap.")
    except Exception as e:
        logger.error(f"Error in process_video_swap_and_reply: {e}")
        await bot.edit_message_text(chat_id=chat_id, message_id=progress_message.message_id, text="An unexpected error occurred during video processing.")
    finally:
        if os.path.exists(target_video_path):
            os.remove(target_video_path)
        if "source_face" in user_data and os.path.exists(user_data["source_face"]):
            os.remove(user_data["source_face"])
        user_data.clear()
        user_data["status"] = SELECTING_SOURCE_IMAGE

async def _process_video_swap(source_face, target_video_path: str, progress_message, bot, chat_id) -> str | None:
    """Internal function to perform face swap on a video."""
    output_path = target_video_path.replace(".mp4", "_swapped.mp4")
    temp_video_no_audio = target_video_path.replace(".mp4", "_temp_no_audio.mp4")
    
    try:
        vr = VideoReader(target_video_path, ctx=cpu(0)) # Use Decord for efficient video reading
        fps = vr.get_avg_fps()
        # Corrected: Use shape to get dimensions in Decord
        frame_shape = vr[0].shape
        height, width, _ = frame_shape
        total_frames = len(vr)

        if total_frames <= 0:
            logger.warning(f"Could not determine total frames for {target_video_path}. Progress updates will be limited.")
            total_frames = -1 # Indicate unknown total frames

        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Use mp4v codec for broader compatibility
        out = cv2.VideoWriter(temp_video_no_audio, fourcc, fps, (width, height))
        
        if not out.isOpened():
            logger.error(f"Could not open video writer for {temp_video_no_audio}")
            return None

        last_reported_percent = -1
        for frame_idx in range(total_frames):
            frame = vr[frame_idx].asnumpy() # Get frame as numpy array
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            target_faces = app.get(frame_bgr)
            if target_faces:
                target_faces = sorted(target_faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)
                frame_bgr = swapper.get(frame_bgr, target_faces[0], source_face, paste_back=True)
            
            out.write(frame_bgr)

            if total_frames > 0:
                current_percent = int(((frame_idx + 1) / total_frames) * 100)
                if current_percent > last_reported_percent + 5 or current_percent == 100: # Update every 5% or at 100%
                    await bot.edit_message_text(chat_id=chat_id, message_id=progress_message.message_id, text=f"Processing video face swap... {current_percent}% complete.")
                    last_reported_percent = current_percent
            elif (frame_idx + 1) % 100 == 0: # Fallback for unknown total frames
                await bot.edit_message_text(chat_id=chat_id, message_id=progress_message.message_id, text=f"Processing video face swap... Processed {frame_idx + 1} frames.")

    except Exception as e:
        logger.error(f"Error during frame processing with Decord: {e}")
        return None
    finally:
        if 'out' in locals() and out.isOpened():
            out.release()
    
    # Use FFmpeg to merge original audio with swapped video
    try:
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
