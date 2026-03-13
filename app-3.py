import os
import logging
import warnings

import gradio as gr
import torch
import torchaudio
import numpy as np
from transformers import pipeline

# ----------------------------------------------------------------------
# 1. Logging and warnings configuration
# ----------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", message=".*torch.utils.checkpoint.*")

# ----------------------------------------------------------------------
# 2. Import TigrinyaTransliterator from external package
# ----------------------------------------------------------------------
try:
    from ti_transliterator import TigrinyaTransliterator
    logger.info("Successfully imported TigrinyaTransliterator from external module.")
except ImportError as e:
    logger.error("Failed to import TigrinyaTransliterator. Make sure the package is installed correctly.")
    raise RuntimeError(
        "TigrinyaTransliterator module not found. Please ensure your GitHub repository "
        "contains a proper Python package (with setup.py or pyproject.toml) and is listed "
        "in requirements.txt as: git+https://github.com/Aregay01/TigrinyaTransliterator.git"
    ) from e

# ----------------------------------------------------------------------
# 3. Model loading (Using HF Pipeline for built-in chunking/accuracy)
# ----------------------------------------------------------------------
MODEL_NAME = "Aregay01/whisper-small-tigrinya-merged"

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

logger.info(f"Loading pipeline for {MODEL_NAME} on {device} with dtype {torch_dtype}")

try:
    asr_pipeline = pipeline(
        "automatic-speech-recognition",
        model=MODEL_NAME,
        torch_dtype=torch_dtype,
        device=device,
        generate_kwargs={
            "forced_decoder_ids": None,
            "max_new_tokens": 225,
            "num_beams": 5,
            "early_stopping": True,
            "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
            "condition_on_prev_tokens": False
        }
    )
    logger.info("Pipeline loaded successfully in full precision.")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise RuntimeError(f"Could not load model {MODEL_NAME}") from e

# Instantiate the transliterator
converter = TigrinyaTransliterator()

# ----------------------------------------------------------------------
# 4. Transcription function
# ----------------------------------------------------------------------
def transcribe(audio):
    """
    Takes an audio file path or numpy array (from microphone) and returns
    the Tigrinya transcription using the robust Hugging Face pipeline.
    """
    if audio is None:
        return "No audio provided."

    try:
        if isinstance(audio, str):
            audio_input = audio
        else:
            sr, y = audio
            if y.dtype == np.int16:
                y = y.astype(np.float32) / 32768.0
            if y.ndim > 1:
                y = y.mean(axis=1)
            audio_input = {"sampling_rate": sr, "raw": y}

        result = asr_pipeline(
            audio_input,
            chunk_length_s=30,
            batch_size=1 if device == "cpu" else 4
        )
        latin_text = result["text"]
        tigrinya = converter.to_tigrinya(latin_text)
        return tigrinya
    except Exception as e:
        logger.exception("Error during transcription")
        return f"Transcription failed: {str(e)}"

# ----------------------------------------------------------------------
# 5. Gradio interface
# ----------------------------------------------------------------------
title = "Tigrinya Speech-to-Text"
description = """
Upload an audio file or record your voice. The app will transcribe it into Tigrinya (Ge'ez) script using a fine-tuned Whisper model.
The model first produces a Latin transcription, then converts it to Tigrinya script.
"""

examples = []  # Add example file paths if available

iface = gr.Interface(
    fn=transcribe,
    inputs=gr.Audio(sources=["microphone", "upload"], type="numpy", label="Audio Input"),
    outputs=gr.Textbox(label="Tigrinya Transcription", lines=3),
    title=title,
    description=description,
    examples=examples,
    cache_examples=False,
)

# ----------------------------------------------------------------------
# 6. Launch
# ----------------------------------------------------------------------
if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0")