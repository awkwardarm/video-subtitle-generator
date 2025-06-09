import os
import subprocess
import textwrap
import whisper
from pysrt import SubRipFile, SubRipItem, SubRipTime
from datetime import timedelta
import torch


def main(video_path, max_duration, wrap_length):
    """Generate subtitles for a video file using the Whisper model.

    Args:
        video_path (str): Path to input video file.
        max_duration (float): Maximum duration of each subtitle segment (in seconds).
        wrap_length (int): Maximum number of characters per line in the subtitles.
    """

    # Set output paths
    base_name = os.path.splitext(video_path)[0]
    srt_output_path = base_name + ".srt"

    video_filebool = True  # Initialize video_filebool

    # Extract audio if a video file is provided
    if os.path.splitext(video_path)[1].lower() in [
        ".mp4",
        ".mov",
        ".avi",
        ".mkv",
        ".flv",
        ".wmv",
        ".webm",
    ]:
        # Set bool to indicate that the input is a video file
        print(f"Extracting audio from video: {video_path}")
        # Ensure the audio path does not already exist
        audio_path = base_name + "_audio.wav"
        extract_audio(video_path, audio_path)

    # If an audio file is provided, use it directly
    elif os.path.splitext(video_path)[1].lower() in [
        ".mp3",
        ".wav",
        ".aac",
        ".ogg",
        ".aif",
        ".aiff",
        ".m4a",
        ".flac",
    ]:
        video_filebool = False  # Set bool to indicate that the input is an audio file
        audio_path = video_path
        print(f"Using provided audio file: {audio_path}")

    # Generate subtitles
    generate_subtitles(audio_path, srt_output_path, max_duration, wrap_length)

    if video_filebool:
        # Clean up intermediate audio file if desired
        os.remove(audio_path)
        print("Audio file removed.")


def extract_audio(video_path, output_audio_path):
    """
    Extract audio from video file using FFmpeg.

    Args:
        video_path (str): Path to input video file.
        output_audio_path (str): Path to output audio file.
    """

    # Run FFmpeg command
    command = [
        "ffmpeg",
        "-i",
        video_path,  # Input video file
        "-vn",  # No video
        "-acodec",
        "pcm_s16le",  # Output codec
        "-ar",
        "16000",  # Set audio sampling rate
        output_audio_path,
    ]

    # if FileNotFoundError: likely ffmpeg not found in PATH
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError:
        print("FFmpeg not found. Please install FFmpeg and add it to your PATH.")
        raise

    print(f"Audio extracted to: {output_audio_path}")


def timedelta_to_srt_time(td):
    """
    Convert a timedelta object to a SubRipTime object.

    Args:
        td (timedelta): A timedelta object.
    Returns:
        SubRipTime: A SubRipTime object
    """
    total_seconds = int(td.total_seconds())
    milliseconds = int((td.total_seconds() - total_seconds) * 1000)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return SubRipTime(
        hours=hours, minutes=minutes, seconds=seconds, milliseconds=milliseconds
    )


def wrap_text(text, wrap_length):
    """
    Wraps text to ensure no line exceeds wrap_length characters.
    """
    return "\n".join(textwrap.wrap(text, width=wrap_length))


def generate_subtitles(audio_path, srt_output_path, max_duration, wrap_length):
    """Generate subtitles from an audio file using the Whisper model.

    Args:
        audio_path (str): Path to input audio file.
        srt_output_path (str): Path to output SRT file.
        max_duration (float): Maximum duration of each subtitle segment (in seconds).
        wrap_length (int): Maximum number of characters per line in the subtitles.
    """
    # Set device to GPU if available
    device = torch.device(
        "cuda:0"
        if torch.cuda.is_available()
        else "cpu"  #! MPS is not supported for this model
    )

    print(f"Using torch device: {device}")
    # Load model
    model = whisper.load_model(
        "small.en",
        device=device,
    )  # Use "base" or other models (e.g., "large") as needed

    # Transcribe audio
    print("Transcribing audio...")
    result = model.transcribe(audio_path, word_timestamps=True, verbose=True)
    print("Transcription complete. Generating subtitles...")

    # Create SRT file
    srt_file = SubRipFile()

    # Process each segment from the model output
    for i, segment in enumerate(result["segments"]):
        start = segment["start"]
        words = segment["words"]  # Word-level timestamps

        # Break segment into smaller chunks based on max_duration
        current_start = start
        current_text = []

        # Process each word in the segment
        for word in words:
            word_end = word["end"]
            word_text = word["word"]

            # Add word to current chunk
            current_text.append(word_text)

            # If max_duration is reached or end of segment, create a new subtitle
            if word_end - current_start >= max_duration or word is words[-1]:
                # Convert timedelta objects to a SubRipTime objects
                start_time = timedelta_to_srt_time(timedelta(seconds=current_start))
                end_time = timedelta_to_srt_time(timedelta(seconds=word_end))

                # Join and wrap text to new line that is over wrap_length
                text = wrap_text(" ".join(current_text), wrap_length)

                # Create SRT item
                item = SubRipItem(
                    index=len(srt_file) + 1, start=start_time, end=end_time, text=text
                )
                srt_file.append(item)

                # Set new start time from word_end
                current_start = word_end
                # Reset for the next chunk of words
                current_text = []

    # Save SRT file
    srt_file.save(srt_output_path, encoding="utf-8")
    print(f"Subtitles saved to {srt_output_path}")


if __name__ == "__main__":

    # Set working directory to the folder containing the video file

    # check if os is mac or windows
    if os.name == "nt":
        folder_path = r"Y:\Dropbox\C2C\subtitle-generator"
    else:
        folder_path = "/Users/matthewtryba/Dropbox/C2C/subtitle-generator"

    # list of valid video file extensions
    valid_extensions = [
        ".mp4",
        ".mov",
        ".avi",
        ".mkv",
        ".flv",
        ".wmv",
        ".webm",
        ".mp3",
        ".wav",
        ".aac",
        ".ogg",
        ".aif",
        ".aiff",
        ".m4a",
        ".flac",
    ]

    # get all video files in the folder
    video_files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if os.path.splitext(f)[1].lower() in valid_extensions
    ]

    if len(video_files) == 0:
        raise ValueError("No video files found in the folder.")
    elif len(video_files) > 1:
        raise ValueError(
            "Multiple video files found in the folder. Please choose only one."
        )
    else:
        video_path = video_files[0]
        print(f"Using video file: {video_path}")

    MAX_DURATION = 2.5
    WRAP_LENGTH = 40

    #! Loading model via huggingface transformers allows for mps but generates improper timestamps
    main(video_path, max_duration=MAX_DURATION, wrap_length=WRAP_LENGTH)
