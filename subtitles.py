import os
import subprocess
import textwrap
import whisper
from pysrt import SubRipFile, SubRipItem, SubRipTime
from datetime import timedelta
import torch

# Set device to GPU if available
device = torch.device(
    "cuda:0"
    if torch.cuda.is_available()
    else "cpu"  # "mps" if torch.backends.mps.is_available() else "cpu" #! MPS is not supported for this model
)

# * Use "tryba-automation" conda environment


# Main function
def main(video_path, max_duration, wrap_length):

    # Set output paths
    base_name = os.path.splitext(video_path)[0]
    audio_path = base_name + "_audio.wav"
    srt_output_path = base_name + ".srt"

    # Extract audio and generate subtitles
    extract_audio(video_path, audio_path)

    # Generate subtitles
    generate_subtitles(audio_path, srt_output_path, max_duration, wrap_length)

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

    subprocess.run(command, check=True)
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

    # Load model
    model = whisper.load_model(
        "large", device=device
    )  # Use "base" or other models (e.g., "large") as needed

    # Transcribe audio
    result = model.transcribe(audio_path, word_timestamps=True)

    # Create SRT file
    srt_file = SubRipFile()

    # Process each segment
    for i, segment in enumerate(result["segments"]):
        start = segment["start"]
        end = segment["end"]
        words = segment["words"]  # Word-level timestamps

        # Break segment into smaller chunks based on max_duration
        current_start = start
        current_text = []

        for word in words:
            # // word_start = word["start"]
            word_end = word["end"]
            word_text = word["word"]

            # Add word to current chunk
            current_text.append(word_text)

            # If max_duration is reached or end of segment, create a new subtitle
            if word_end - current_start >= max_duration or word is words[-1]:
                start_time = timedelta_to_srt_time(timedelta(seconds=current_start))
                end_time = timedelta_to_srt_time(timedelta(seconds=word_end))

                # Join and wrap text that is over wrap_length
                text = wrap_text(" ".join(current_text), wrap_length)

                # Create SRT item
                item = SubRipItem(
                    index=len(srt_file) + 1, start=start_time, end=end_time, text=text
                )
                srt_file.append(item)

                # Reset for the next chunk
                current_start = word_end
                current_text = []

    # Save SRT file
    srt_file.save(srt_output_path, encoding="utf-8")
    print(f"Subtitles saved to {srt_output_path}")


# Usage
if __name__ == "__main__":

    MAX_DURATION = 3
    WRAP_LENGTH = 40

    video_path = "/Users/matthewtryba/Desktop/music-data.mp4"

    main(video_path, max_duration=MAX_DURATION, wrap_length=WRAP_LENGTH)
