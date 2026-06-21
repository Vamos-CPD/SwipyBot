
import logging
import os
import asyncio
from telegram import Update, Bot
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from insightface.app import FaceAnalysis
from insightface.model_zoo import model_zoo
import cv2
import numpy as np
from moviepy import VideoFileClip, ImageSequenceClip
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

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
app.prepare(ctx_id=0, det_size=(640, 640))

swapper = model_zoo.get_model(INSWAFFER_MODEL_PATH)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /start is issued."""
    user = update.effective_user
    await update.message.reply_html(
        f"Hi {user.mention_html()}!\nI am a FaceSwap bot. Send me /swipFace to start swapping faces."
    )

async def swip_face_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Inform user to send source and target media."""
    await update.message.reply_text(
        "Please send me two images or a video and an image.\n"\
        "The first media will be the source face, and the second will be the target."
    )

async def handle_media(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle incoming media (photos and videos) for face swapping."""
    chat_id = update.effective_chat.id
    user_data = context.user_data

    if "source_face" not in user_data:
        # First media is the source face
        if update.message.photo:
            file_id = update.message.photo[-1].file_id
            file = await context.bot.get_file(file_id)
            source_path = f"/tmp/{file_id}.jpg"
            await file.download_to_drive(source_path)
            user_data["source_face"] = source_path
            await update.message.reply_text("Source face received. Now send the target image or video.")
        elif update.message.video:
            await update.message.reply_text("Videos cannot be used as source faces. Please send an image for the source face.")
        else:
            await update.message.reply_text("Please send an image for the source face.")
        return

    # Second media is the target
    source_face_path = user_data["source_face"]
    source_face_img = cv2.imread(source_face_path)
    source_faces = app.get(source_face_img)

    if not source_faces:
        await update.message.reply_text("Could not detect a face in the source image. Please try again with a clearer image.")
        del user_data["source_face"]
        return
    
    source_face = source_faces[0]

    if update.message.photo:
        file_id = update.message.photo[-1].file_id
        file = await context.bot.get_file(file_id)
        target_path = f"/tmp/{file_id}.jpg"
        await file.download_to_drive(target_path)
        await update.message.reply_text("Processing image face swap...")
        output_path = await process_image_swap(source_face, target_path)
        if output_path:
            await update.message.reply_document(document=output_path)
        else:
            await update.message.reply_text("Failed to process image face swap. No faces detected in target or an error occurred.")

    elif update.message.video:
        file_id = update.message.video.file_id
        file = await context.bot.get_file(file_id)
        target_path = f"/tmp/{file_id}.mp4"
        await file.download_to_drive(target_path)
        await update.message.reply_text("Processing video face swap... This may take a while.")
        output_path = await process_video_swap(source_face, target_path)
        if output_path:
            await update.message.reply_video(video=output_path)
        else:
            await update.message.reply_text("Failed to process video face swap. No faces detected in target or an error occurred.")
    else:
        await update.message.reply_text("Please send a valid image or video for the target.")
    
    # Clean up user data for the next swap
    if "source_face" in user_data:
        os.remove(user_data["source_face"])
        del user_data["source_face"]

async def process_image_swap(source_face, target_image_path: str) -> str | None:
    """Performs face swap on an image."""
    target_img = cv2.imread(target_image_path)
    if target_img is None:
        logger.error(f"Could not read target image: {target_image_path}")
        return None

    target_faces = app.get(target_img)
    if not target_faces:
        return None

    # Assuming we swap the source face onto the first detected target face
    result_img = swapper.get(target_img, target_faces[0], source_face, paste_back=True)
    output_path = target_image_path.replace(".jpg", "_swapped.jpg")
    cv2.imwrite(output_path, result_img)
    return output_path

async def process_video_swap(source_face, target_video_path: str) -> str | None:
    """Performs face swap on a video."""
    output_path = target_video_path.replace(".mp4", "_swapped.mp4")
    temp_audio_path = target_video_path.replace(".mp4", "_audio.mp3")

    try:
        video_clip = VideoFileClip(target_video_path)
        video_clip.audio.write_audiofile(temp_audio_path)
    except Exception as e:
        logger.warning(f"Could not extract audio from video: {e}. Proceeding without audio.")
        temp_audio_path = None

    frames = []
    for i, frame in enumerate(video_clip.iter_frames(fps=video_clip.fps)):
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        target_faces = app.get(frame_bgr)
        if target_faces:
            # Swap face on the first detected face in the frame
            result_frame_bgr = swapper.get(frame_bgr, target_faces[0], source_face, paste_back=True)
            result_frame_rgb = cv2.cvtColor(result_frame_bgr, cv2.COLOR_BGR2RGB)
            frames.append(result_frame_rgb)
        else:
            frames.append(frame) # If no face detected, keep original frame

    if not frames:
        return None

    swapped_clip = ImageSequenceClip(frames, fps=video_clip.fps)
    if temp_audio_path and os.path.exists(temp_audio_path):
        swapped_clip = swapped_clip.set_audio(VideoFileClip(temp_audio_path).audio)
    
    swapped_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")
    
    # Clean up temporary audio file
    if temp_audio_path and os.path.exists(temp_audio_path):
        os.remove(temp_audio_path)

    return output_path

def main() -> None:
    """Start the bot."""
    from telegram.request import HTTPXRequest
    request = HTTPXRequest(connect_timeout=30.0, read_timeout=30.0)
    application = Application.builder().token(BOT_TOKEN).request(request).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("swipFace", swip_face_command))
    application.add_handler(MessageHandler(filters.PHOTO | filters.VIDEO, handle_media))

    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
