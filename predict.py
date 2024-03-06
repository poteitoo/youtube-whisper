from cog import BasePredictor, Input, Path, BaseModel
from pydub import AudioSegment
from whisperx.audio import N_SAMPLES, log_mel_spectrogram
import os
import whisperx
import json
import torch
from yt_dlp import YoutubeDL
from pprint import pprint

compute_type = "int8"
device = "cuda"


def checkIfIsAvailableVideo(info, *, incomplete):
    duration = info.get("duration")
    if duration is None:  # If the duration is unknown, e.g. live streaming
        raise SystemError("Video duration is unknown")


ydl_opts = {
    "extract_audio": True,
    "format": "bestaudio",
    "outtmpl": "audios/%(id)s.%(ext)s",
    "match_filter": checkIfIsAvailableVideo,
}


class Predictor(BasePredictor):
    def setup(self):
        self.model = whisperx.load_model("large-v2", device, compute_type=compute_type)

    def predict(
        self,
        url: str = Input(
            description="Youtube url containing video id",
            default=None,
        ),
        batch_size: int = Input(
            description="Parallelization of input audio transcription", default=64
        ),
        huggingface_access_token: str = Input(
            description="To enable diarization, please enter your HuggingFace token (read). You need to accept "
            "the user agreement for the models specified in the README.",
            default=None,
        ),
    ) -> str:
        video_info = self.download_youtube_video(url)
        ext = video_info.get("ext")
        video_id = video_info.get("id")
        audio_path = f"audios/{video_id}.{ext}"
        if os.path.exists(audio_path):
            raise SystemError("Audio file not found")

        with torch.inference_mode():
            # 1. Transcribe with original whisper (batched)
            audio = whisperx.load_audio(audio_path)
            result = self.model.transcribe(audio, batch_size=batch_size)

            # 2. Align whisper output
            model_a, metadata = whisperx.load_align_model(
                language_code=result["language"], device=device
            )
            print("Metadata")
            pprint(metadata)
            result = whisperx.align(
                result["segments"],
                model_a,
                metadata,
                audio,
                device,
                return_char_alignments=False,
            )
            print("Result")
            pprint(result)

            # 3. Assign speaker labels
            diarize_model = whisperx.DiarizationPipeline(
                use_auth_token=huggingface_access_token, device=device
            )
            diarize_segments = diarize_model(audio)
            result = whisperx.assign_word_speakers(diarize_segments, result)
        self.post_process(audio_path)
        return json.dumps(result["segments"])

    def download_youtube_video(self, url: str):
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            ydl.download([url])
            return ydl.sanitize_info(info)

    def post_process(self, file_path: str):
        if os.path.exists(file_path):
            os.remove(file_path)
