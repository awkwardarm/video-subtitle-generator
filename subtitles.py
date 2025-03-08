import os
import subprocess
import textwrap
import whisper
from pysrt import SubRipFile, SubRipItem, SubRipTime
from datetime import timedelta
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


def main(video_path, max_duration, wrap_length):
    """Generate subtitles for a video file using the Whisper model.

    Args:
        video_path (str): Path to input video file.
        max_duration (float): Maximum duration of each subtitle segment (in seconds).
        wrap_length (int): Maximum number of characters per line in the subtitles.
    """

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


def split_chunks(chunks, max_duration=2.5):
    new_chunks = []

    for chunk in chunks:
        start, end = chunk["timestamp"]
        text = chunk["text"]
        duration = end - start

        if duration <= max_duration:
            new_chunks.append(chunk)
            continue

        words = text.split()
        total_words = len(words)
        time_per_word = duration / total_words

        current_start = start
        temp_words = []

        for word in words:
            temp_words.append(word)
            current_duration = len(temp_words) * time_per_word

            if current_duration >= max_duration:
                new_chunks.append(
                    {
                        "timestamp": (current_start, current_start + current_duration),
                        "text": " ".join(temp_words),
                    }
                )
                current_start += current_duration
                temp_words = []

        if temp_words:
            new_chunks.append(
                {
                    "timestamp": (current_start, end),
                    "text": " ".join(temp_words),
                }
            )

        # remove empty chunks
        new_chunks = [chunk for chunk in new_chunks if chunk["text"]]

    return new_chunks


def generate_subtitles(audio_path, srt_output_path, max_duration, wrap_length):
    """Generate subtitles from an audio file using the Whisper model.

    Args:
        audio_path (str): Path to input audio file.
        srt_output_path (str): Path to output SRT file.
        max_duration (float): Maximum duration of each subtitle segment (in seconds).
        wrap_length (int): Maximum number of characters per line in the subtitles.
    """

    # # * Load model
    # model = whisper.load_model(
    #     "small.en", device=device
    # )  # Use "base" or other models (e.g., "large") as needed

    # # Transcribe audio
    # result = model.transcribe(audio_path, word_timestamps=True)

    # * Load Model via transformers
    model_id = "openai/whisper-large-v3-turbo"
    model_id = "openai/whisper-small.en"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
    )

    result = pipe(
        audio_path, return_timestamps=True
    )  # , generate_kwargs=generate_kwargs)
    print(result)

    # Resize chunks
    chunk_size = MAX_DURATION
    chunks = split_chunks(result["chunks"], chunk_size)

    for chunk in chunks:
        print(chunk)

    # Create SRT file
    srt_file = SubRipFile()

    # Process each segment from the model output
    for i, chunk in enumerate(result["chunks"]):
        chunk_start_timestamp = chunk["timestamp"][0]  # Segment start time
        chunk_end_timestamp = chunk["timestamp"][1]  # Segment end time
        words = chunk["text"]  # List of words in the segment

        # Break segment into smaller chunks based on max_duration
        current_start = chunk_start_timestamp
        current_text = []

        if chunk_end_timestamp - chunk_start_timestamp >= max_duration:
            # If the segment is too long, split it into smaller chunks
            words = chunk["text"]
            current_start = words[0]["start"]
            current_text = []

        # Process each word in the segment
        for word in words:
            word_end = word["end"]
            word_text = word["word"]

            # Add word to current chunk
            current_text.append(word)

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
    # video_path = r"C:\Users\matth\Dropbox\C2C\subtitle-generator\music-data.mp4"
    video_path = r"/Users/matthewtryba/Desktop/Stem Logic - Setup 2025-03-07.mov"

    MAX_DURATION = 2.5
    WRAP_LENGTH = 40

    # Set torch device
    device = torch.device(
        "cuda:0"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
        # else "cpu"
    )

    # Set torch dtype
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    main(video_path, max_duration=MAX_DURATION, wrap_length=WRAP_LENGTH)
