import argparse
import os
import sys
import tempfile
from pathlib import Path
import shutil
import glob
import subprocess
import math
import zipfile

import librosa
import numpy as np
import torch
import torchaudio
import traceback

from utils.formatter import format_audio_list, list_audios
from utils.gpt_train import train_gpt

from faster_whisper import WhisperModel

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

AUDIO_EXTENSIONS = [".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".opus"]

def check_yt_dlp():
    try:
        result = subprocess.run(["yt-dlp", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            raise Exception()
        return True
    except Exception:
        print("Error: 'yt-dlp' is not installed or not found in your PATH.")
        print("Please install it from https://github.com/yt-dlp/yt-dlp or via your OS package manager.")
        sys.exit(1)

def download_youtube_audio(url, output_dir):
    check_yt_dlp()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Will produce bestaudio in original format; will convert later if needed
    cmd = [
        "yt-dlp",
        "-f", "bestaudio",
        "--extract-audio",
        "--audio-format", "wav",
        "-o", str(output_dir / "%(title).200s.%(ext)s"),
        url,
    ]
    print(f"Downloading YouTube audio: {url}")
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        print("yt-dlp error output:\n", result.stderr)
        raise RuntimeError(f"yt-dlp failed to download audio from {url}")
    # Find new wav files in output_dir
    downloaded = list(output_dir.glob("*.wav"))
    return downloaded

def run_ffmpeg(cmd):
    try:
        print(f"Running FFmpeg command: {' '.join(cmd)}")
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, errors='ignore')
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            print(f"FFmpeg Error Output:\n{stderr}")
            raise RuntimeError(f"FFmpeg command failed with exit code {process.returncode}")
        return True
    except FileNotFoundError:
        print("Error: ffmpeg command not found. Please ensure FFmpeg is installed and in your PATH.")
        return False
    except Exception as e:
        print(f"An error occurred while running FFmpeg: {e}")
        traceback.print_exc()
        return False

def get_audio_duration(file_path):
    try:
        info = torchaudio.info(str(file_path))
        return info.num_frames / info.sample_rate
    except Exception as e:
        print(f"Error getting duration for {file_path}: {e}")
        try:
            cmd = ["ffmpeg", "-i", str(file_path), "-f", "null", "-"]
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, errors='ignore')
            stdout, stderr = process.communicate()
            for line in stderr.splitlines():
                if "Duration:" in line:
                    time_str = line.split("Duration:")[1].split(",")[0].strip()
                    h, m, s = map(float, time_str.split(':'))
                    duration = h * 3600 + m * 60 + s
                    return duration
            return None
        except Exception as ff_e:
            print(f"Error getting duration using FFmpeg for {file_path}: {ff_e}")
            return None

def find_audio_files(input_path):
    input_path = Path(input_path)
    if input_path.is_file():
        return [input_path]
    elif input_path.is_dir():
        found = []
        for ext in AUDIO_EXTENSIONS:
            found.extend(input_path.rglob(f"*{ext}"))
        # Remove duplicates and hidden files
        found = [f for f in set(found) if not f.name.startswith(".")]
        return sorted(found)
    else:
        return []

def prepare_audios(input_items, temp_dir, warning_limit_minutes=50):
    """
    input_items: list of Path objects (file or folder), or list of YouTube links as strings.
    Returns: list of Path objects to processed audio files (all mp3), and total duration in seconds.
    """
    prepared_files = []
    temp_dir = Path(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    total_duration = 0
    for idx, item in enumerate(input_items):
        # If it's a YouTube link:
        if isinstance(item, str) and (item.startswith("http://") or item.startswith("https://")):
            yt_files = download_youtube_audio(item, temp_dir)
            if not yt_files:
                print(f"No audio downloaded for {item}")
                continue
            for wav_in in yt_files:
                mp3_out = temp_dir / (wav_in.stem + ".mp3")
                cmd_convert = [
                    "ffmpeg", "-i", str(wav_in),
                    "-vn", "-acodec", "libmp3lame", "-q:a", "2",
                    "-ac", "1", "-ar", "44100",
                    str(mp3_out)
                ]
                if not run_ffmpeg(cmd_convert):
                    continue
                duration = get_audio_duration(mp3_out)
                if duration:
                    total_duration += duration
                prepared_files.append(mp3_out)
                wav_in.unlink()
        else:
            # Local file
            input_path = Path(item)
            if input_path.suffix.lower() == ".mp3":
                mp3_path = temp_dir / f"{input_path.stem}_copy.mp3"
                shutil.copy(str(input_path), str(mp3_path))
                duration = get_audio_duration(mp3_path)
                if duration:
                    total_duration += duration
                prepared_files.append(mp3_path)
            else:
                mp3_path = temp_dir / f"{input_path.stem}_converted.mp3"
                cmd_convert = [
                    "ffmpeg", "-i", str(input_path),
                    "-vn", "-acodec", "libmp3lame", "-q:a", "2",
                    "-ac", "1", "-ar", "44100",
                    str(mp3_path)
                ]
                if not run_ffmpeg(cmd_convert):
                    continue
                duration = get_audio_duration(mp3_path)
                if duration:
                    total_duration += duration
                prepared_files.append(mp3_path)
    # Warn if over limit
    warning_limit_seconds = warning_limit_minutes * 60
    if total_duration > warning_limit_seconds:
        print(f"WARNING: Total audio duration ({total_duration/60:.2f} min) exceeds {warning_limit_minutes} minutes.")
        print("This may result in overfitting, which can cause your model to hallucinate or behave unexpectedly.")
        response = input("Do you want to continue? [y/N]: ").strip().lower()
        if response not in ['y', 'yes']:
            print("Aborting as per user request.")
            for f in prepared_files:
                try: f.unlink()
                except Exception: pass
            sys.exit(1)
    return prepared_files, total_duration

def clear_gpu_cache():
    if torch.cuda.is_available():
        print("Clearing GPU cache...")
        torch.cuda.empty_cache()
        print("GPU cache cleared.")

XTTS_MODEL = None

def preprocess_dataset_headless(audio_files, language, whisper_model_name, dataset_out_path):
    clear_gpu_cache()
    print(f"\n--- Starting Step 1: Data Processing ---")
    print(f"Audio files: {audio_files}")
    print(f"Language: {language}")
    print(f"Whisper model: {whisper_model_name}")
    print(f"Dataset output path: {dataset_out_path}")

    os.makedirs(dataset_out_path, exist_ok=True)
    train_meta = ""
    eval_meta = ""
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float32"
        print(f"Loading Whisper model '{whisper_model_name}' on device '{device}' with compute type '{compute_type}'...")
        asr_model = WhisperModel(whisper_model_name, device=device, compute_type=compute_type)
        print("Whisper model loaded.")

        print("Formatting audio list...")
        train_meta, eval_meta, audio_total_size = format_audio_list(
            [str(x) for x in audio_files],
            asr_model=asr_model,
            target_language=language,
            out_path=dataset_out_path,
            gradio_progress=None
        )
        print("Audio list formatted.")

    except Exception as e:
        print(f"\n---!!! Data processing failed! !!!---")
        traceback.print_exc()
        return f"Data processing failed: {e}", "", ""

    del asr_model
    clear_gpu_cache()

    if not Path(train_meta).exists() or not Path(eval_meta).exists():
         message = "Data processing failed to create metadata files. The input audio might be silent or too noisy after processing."
         print(f"\n---!!! Data processing error: {message} !!!---")
         if audio_total_size < 1:
              print("Reported audio size from Whisper was less than 1 second.")
         return message, "", ""
    elif audio_total_size < 120:
        message = f"Warning: The total detected speech duration ({audio_total_size:.2f} seconds) is less than the recommended 120 seconds. Training quality might be affected."
        print(f"\n---!!! Data processing warning: {message} !!!---")
    print(f"Total detected speech size: {audio_total_size:.2f} seconds.")
    print(f"Training metadata file: {train_meta}")
    print(f"Evaluation metadata file: {eval_meta}")
    print(f"--- Step 1: Data Processing Completed ---")
    return "Dataset Processed Successfully!", str(train_meta), str(eval_meta)

def train_model_headless(language, train_csv_path, eval_csv_path, num_epochs, batch_size, grad_acumm, output_path_base, max_audio_length_sec, version="v2.0.2", custom_model=""):
    clear_gpu_cache()
    print(f"\n--- Starting Step 2: Fine-tuning XTTS ---")
    print(f"Language: {language}")
    print(f"Training CSV Path: {train_csv_path}")
    print(f"Evaluation CSV Path: {eval_csv_path}")
    print(f"Epochs: {num_epochs}")
    print(f"Batch Size: {batch_size}")
    print(f"Gradient Accumulation: {grad_acumm}")
    print(f"Max Audio Length: {max_audio_length_sec} seconds")
    print(f"Output Path Base: {output_path_base}")
    print(f"XTTS Base Version: {version}")
    print(f"Custom Base Model Path: {'Default' if not custom_model else custom_model}")

    output_path_base = Path(output_path_base)
    run_dir = output_path_base / "run"

    if run_dir.exists():
        print(f"Removing existing training run directory: {run_dir}")
        shutil.rmtree(run_dir)

    train_csv_path_obj = Path(train_csv_path).resolve()
    eval_csv_path_obj = Path(eval_csv_path).resolve()
    if not train_csv_path_obj.is_file() or not eval_csv_path_obj.is_file():
        print(f"Error: Training CSV ({train_csv_path_obj}) or Evaluation CSV ({eval_csv_path_obj}) file not found.")
        return "Training or Evaluation CSV file not found. Ensure Step 1 completed successfully.", "", "", "", "", ""

    train_csv_full_path = str(train_csv_path_obj)
    eval_csv_full_path = str(eval_csv_path_obj)
    try:
        max_audio_length_frames = int(max_audio_length_sec * 22050)
        print(f"Max audio length in frames: {max_audio_length_frames}")
        speaker_xtts_path, config_path, _, vocab_file, exp_path, speaker_wav = train_gpt(
            custom_model=custom_model,
            version=version,
            language=language,
            num_epochs=num_epochs,
            batch_size=batch_size,
            grad_acumm=grad_acumm,
            train_csv=train_csv_full_path,
            eval_csv=eval_csv_full_path,
            output_path=str(output_path_base.resolve()),
            max_audio_length=max_audio_length_frames
        )
        exp_path_obj = Path(exp_path)
        best_model_path = exp_path_obj / "best_model.pth"
        if not best_model_path.exists():
            pth_files = sorted(list(exp_path_obj.glob("*.pth")), key=os.path.getmtime, reverse=True)
            if pth_files:
                model_files = [p for p in pth_files if "optimizer" not in p.name.lower() and "dvae" not in p.name.lower()]
                if model_files:
                    best_model_path = model_files[0]
                    print(f"Warning: 'best_model.pth' not found. Using latest model checkpoint: {best_model_path}")
                elif pth_files:
                    best_model_path = pth_files[0]
                    print(f"Warning: 'best_model.pth' not found and no other model checkpoints. Using latest .pth file: {best_model_path}")
                else:
                    raise FileNotFoundError(f"No model checkpoints found in {exp_path_obj}")
            else:
                raise FileNotFoundError(f"No '.pth' model checkpoint found in {exp_path_obj}")

        # Move/copy only the needed files to output_path_base
        unoptimized_model_target_path = output_path_base / "model.pth"
        print(f"Copying best model checkpoint {best_model_path} to {unoptimized_model_target_path}")
        shutil.copy(str(best_model_path), str(unoptimized_model_target_path))

        # Config file
        source_config_path = Path(config_path).resolve()
        intended_config_path = (output_path_base / source_config_path.name).resolve()
        if source_config_path != intended_config_path:
            print(f"Copying config {source_config_path} to {intended_config_path}")
            shutil.copy(str(source_config_path), str(intended_config_path))
        final_config_path = intended_config_path

        # Vocab file
        source_vocab_path = Path(vocab_file).resolve()
        intended_vocab_path = (output_path_base / source_vocab_path.name).resolve()
        if source_vocab_path != intended_vocab_path:
            print(f"Copying vocab {source_vocab_path} to {intended_vocab_path}")
            shutil.copy(str(source_vocab_path), str(intended_vocab_path))
        final_vocab_path = intended_vocab_path

        # Speaker file
        final_speaker_xtts_path = None
        if speaker_xtts_path:
            source_speaker_xtts_path = Path(speaker_xtts_path).resolve()
            intended_speaker_xtts_path = (output_path_base / source_speaker_xtts_path.name).resolve()
            if source_speaker_xtts_path.exists():
                if source_speaker_xtts_path != intended_speaker_xtts_path:
                    print(f"Copying speaker profile {source_speaker_xtts_path} to {intended_speaker_xtts_path}")
                    shutil.copy(str(source_speaker_xtts_path), str(intended_speaker_xtts_path))
                final_speaker_xtts_path = intended_speaker_xtts_path
            else:
                exp_path_obj = Path(exp_path)
                alt_speaker_path = exp_path_obj / source_speaker_xtts_path.name
                if alt_speaker_path.exists():
                    print(f"Copying speaker profile from experiment dir {alt_speaker_path} to {intended_speaker_xtts_path}")
                    shutil.copy(str(alt_speaker_path), str(intended_speaker_xtts_path))
                    final_speaker_xtts_path = intended_speaker_xtts_path
                else:
                    print(f"Warning: Speaker profile {source_speaker_xtts_path.name} not found in expected location or experiment dir.")

        else:
            print("Warning: No speaker_xtts_path returned from train_gpt.")

        final_speaker_wav_path = str(Path(speaker_wav).resolve())
        # Remove run directory after successful copy
        if run_dir.exists():
            print(f"Removing temporary run directory: {run_dir}")
            shutil.rmtree(run_dir)

        print("--- Step 2: Fine-tuning Completed ---")
        return ("Model training done!",
                str(final_config_path),
                str(final_vocab_path),
                str(unoptimized_model_target_path.resolve()),
                str(final_speaker_xtts_path) if final_speaker_xtts_path else None,
                final_speaker_wav_path)
    except Exception as e:
        print(f"\n---!!! Model training failed! !!!---")
        traceback.print_exc()
        return f"Model training failed: {e}", "", "", "", "", ""

def create_reference_wavs(original_ref_wav_path, output_dir, output_basename):
    print("\n--- Creating Reference WAV Files ---")
    original_ref_wav_path = Path(original_ref_wav_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not original_ref_wav_path.exists():
        print(f"Error: Original reference WAV not found at {original_ref_wav_path}")
        wav_dir = original_ref_wav_path.parent
        found_wavs = list(wav_dir.glob("*.wav"))
        if found_wavs:
            original_ref_wav_path = found_wavs[0]
            print(f"Warning: Original path invalid. Using first found WAV as reference: {original_ref_wav_path}")
        else:
            print(f"Error: Cannot find any WAV file in {wav_dir} to use as reference.")
            return False

    final_ref_path = output_dir / f"{output_basename}.wav"
    print(f"Copying original reference {original_ref_wav_path} to {final_ref_path}")
    shutil.copy(str(original_ref_wav_path), str(final_ref_path))

    ref_16k_path = output_dir / f"{output_basename}_16000.wav"
    print(f"Creating 16kHz reference WAV: {ref_16k_path}")
    cmd_16k = ["ffmpeg", "-i", str(final_ref_path), "-ar", "16000", "-ac", "1", str(ref_16k_path), "-y"]
    if not run_ffmpeg(cmd_16k):
        print("Failed to create 16kHz reference WAV.")

    ref_24k_path = output_dir / f"{output_basename}_24000.wav"
    print(f"Creating 24kHz reference WAV: {ref_24k_path}")
    cmd_24k = ["ffmpeg", "-i", str(final_ref_path), "-ar", "24000", "-ac", "1", str(ref_24k_path), "-y"]
    if not run_ffmpeg(cmd_24k):
        print("Failed to create 24kHz reference WAV.")

    print("--- Reference WAV Creation Completed ---")
    return True

def load_model_headless(xtts_checkpoint, xtts_config, xtts_vocab, xtts_speaker):
    global XTTS_MODEL
    clear_gpu_cache()
    print(f"\n--- Starting Step 3: Loading Fine-tuned Model ---")
    print(f"Checkpoint: {xtts_checkpoint}")
    print(f"Config: {xtts_config}")
    print(f"Vocab: {xtts_vocab}")
    print(f"Speaker: {xtts_speaker}")

    if not Path(xtts_checkpoint).exists() or not Path(xtts_config).exists() or not Path(xtts_vocab).exists():
         missing = [p for p in [xtts_checkpoint, xtts_config, xtts_vocab] if not Path(p).exists()]
         print(f"Error: Model loading failed. Missing essential files: {missing}")
         return "Model loading failed: Essential files not found."
    if xtts_speaker and not Path(xtts_speaker).exists():
        print(f"Warning: Speaker file {xtts_speaker} not found. Model might load but speaker info will be missing.")

    try:
        print("Initializing XTTS model configuration...")
        config = XttsConfig()
        config.load_json(xtts_config)

        print("Initializing XTTS model from configuration...")
        XTTS_MODEL = Xtts.init_from_config(config)

        print("Loading checkpoint and speaker data...")
        XTTS_MODEL.load_checkpoint(
             config,
             checkpoint_path=xtts_checkpoint,
             vocab_path=xtts_vocab,
             speaker_file_path=xtts_speaker if xtts_speaker and Path(xtts_speaker).exists() else None,
             use_deepspeed=False
             )

        if torch.cuda.is_available():
            print("Moving model to GPU...")
            XTTS_MODEL.cuda()
        else:
            print("CUDA not available, using CPU.")

        print("--- Step 3: Model Loading Completed ---")
        return "Model Loaded Successfully!"
    except Exception as e:
        print(f"\n---!!! Model loading failed! !!!---")
        traceback.print_exc()
        XTTS_MODEL = None
        return f"Model loading failed: {e}"

def run_tts_headless(lang, tts_text, speaker_audio_file, output_wav_path, temperature=0.75, length_penalty=1.0, repetition_penalty=5.0, top_k=50, top_p=0.85, sentence_split=True):
    print(f"\n--- Starting Step 4: Generating Example TTS ---")
    print(f"Language: {lang}")
    print(f"Text: '{tts_text}'")
    print(f"Reference Speaker WAV: {speaker_audio_file}")
    print(f"Output Path: {output_wav_path}")
    print(f"Settings: Temp={temperature}, LenPenalty={length_penalty}, RepPenalty={repetition_penalty}, TopK={top_k}, TopP={top_p}, Split={sentence_split}")

    if XTTS_MODEL is None:
        print("Error: Model is not loaded. Cannot run TTS.")
        return "TTS failed: Model not loaded.", None
    if not Path(speaker_audio_file).exists():
        print(f"Error: Speaker reference audio not found at {speaker_audio_file}")
        return "TTS failed: Speaker reference audio not found.", None

    try:
        print("Getting conditioning latents...")
        speaker_audio_path = Path(speaker_audio_file)
        ref_24k_path = speaker_audio_path.parent / f"{speaker_audio_path.stem}_24000.wav"
        if ref_24k_path.exists():
             print(f"Using 24kHz reference: {ref_24k_path}")
             speaker_ref_to_use = str(ref_24k_path)
        else:
             print(f"Warning: 24kHz reference not found, using original: {speaker_audio_file}. Model will resample.")
             speaker_ref_to_use = str(speaker_audio_file)

        if hasattr(XTTS_MODEL, "get_conditioning_latents"):
            gpt_cond_latent, speaker_embedding = XTTS_MODEL.get_conditioning_latents(
                audio_path=speaker_ref_to_use,
                gpt_cond_len=getattr(XTTS_MODEL.config, 'gpt_cond_len', 30),
                max_ref_length=getattr(XTTS_MODEL.config, 'max_ref_len', 60),
                sound_norm_refs=getattr(XTTS_MODEL.config, 'sound_norm_refs', False)
            )
        elif hasattr(XTTS_MODEL, "extract_tts_latents"):
             latents = XTTS_MODEL.extract_tts_latents(
                 speaker_wav=speaker_ref_to_use,
                 language=lang,
             )
             gpt_cond_latent = latents.get("gpt_cond_latents")
             speaker_embedding = latents.get("speaker_embedding")
             if gpt_cond_latent is None or speaker_embedding is None:
                 raise RuntimeError("Failed to extract latents using 'extract_tts_latents'. Check XTTS version compatibility.")
        else:
             raise NotImplementedError("Could not find a method to get conditioning latents (get_conditioning_latents or extract_tts_latents) in the loaded XTTS model.")

        print("Conditioning latents obtained.")
        print("Running TTS inference...")
        out = XTTS_MODEL.inference(
            text=tts_text,
            language=lang,
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            temperature=temperature,
            length_penalty=length_penalty,
            repetition_penalty=float(repetition_penalty),
            top_k=top_k,
            top_p=top_p,
            enable_text_splitting=sentence_split
        )
        print("Inference completed.")

        print(f"Saving generated audio to {output_wav_path}...")
        if isinstance(out["wav"], (list, np.ndarray)):
            wav_tensor = torch.tensor(out["wav"]).unsqueeze(0)
        elif isinstance(out["wav"], torch.Tensor):
            wav_tensor = out["wav"] if out["wav"].dim() > 1 else out["wav"].unsqueeze(0)
        else:
            raise TypeError(f"Unexpected type for output waveform: {type(out['wav'])}")

        torchaudio.save(str(output_wav_path), wav_tensor.cpu(), 24000)

        print(f"--- Step 4: Example TTS Generation Completed ---")
        return "Speech generated successfully!", str(output_wav_path)

    except Exception as e:
        print(f"\n---!!! TTS inference failed! !!!---")
        traceback.print_exc()
        return f"TTS inference failed: {e}", None

def zip_dataset(dataset_dir, output_zip):
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(dataset_dir):
            for file in files:
                file_path = Path(root) / file
                arcname = str(file_path.relative_to(dataset_dir))
                zipf.write(str(file_path), arcname)
    print(f"Dataset zipped to {output_zip}")

def main():
    parser = argparse.ArgumentParser(description="Headless XTTS Fine-tuning and Inference Script")

    parser.add_argument("--input", type=str, required=True,
        help="Path to input audio file, folder, or YouTube link(s) (comma separated, in quotes).")
    parser.add_argument("--output_dir_base", type=str, default="./xtts_finetuned_models",
        help="Base directory where the output folder for this model will be created.")
    parser.add_argument("--model_name", type=str, default=None,
        help="Name for the output folder and reference files (defaults to input audio filename/folder/youtube id).")

    parser.add_argument("--lang", type=str, default="en", help="Language of the dataset (ISO 639-1 code).")
    parser.add_argument("--whisper_model", type=str, default="large-v3", help="Whisper model to use for transcription.")

    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for training.")
    parser.add_argument("--grad_acumm", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--max_audio_length", type=int, default=11, help="Maximum audio segment length in seconds for training.")
    parser.add_argument("--xtts_base_version", type=str, default="v2.0.2", help="Base XTTS model version to fine-tune from.")
    parser.add_argument("--custom_model", type=str, default="", help="(Optional) Path or URL to a custom .pth base model file.")

    parser.add_argument("--example_text", type=str, default="This is an example sentence generated by the fine tuned model.", help="Text to use for generating the example output WAV.")
    parser.add_argument("--temperature", type=float, default=0.75, help="TTS inference temperature.")
    parser.add_argument("--length_penalty", type=float, default=1.0, help="TTS inference length penalty.")
    parser.add_argument("--repetition_penalty", type=float, default=5.0, help="TTS inference repetition penalty.")
    parser.add_argument("--top_k", type=int, default=50, help="TTS inference top K.")
    parser.add_argument("--top_p", type=float, default=0.85, help="TTS inference top P.")
    parser.add_argument("--no_sentence_split", action="store_true", help="Disable sentence splitting during TTS inference.")

    args = parser.parse_args()

    input_arg = args.input
    # Support comma-separated YouTube links (in quotes)
    if "," in input_arg and (input_arg.startswith("http://") or input_arg.startswith("https://")):
        input_items = [x.strip() for x in input_arg.split(",") if x.strip()]
    else:
        input_path = Path(input_arg)
        if input_path.exists():
            if input_path.is_file():
                input_items = [input_path]
            elif input_path.is_dir():
                input_items = find_audio_files(input_path)
                if not input_items:
                    print(f"No audio files found in folder {input_path}")
                    sys.exit(1)
            else:
                print(f"Input path '{input_arg}' is neither a file nor a folder.")
                sys.exit(1)
        elif input_arg.startswith("http://") or input_arg.startswith("https://"):
            input_items = [input_arg]
        else:
            print(f"Input '{input_arg}' is not a valid file, folder, or YouTube link.")
            sys.exit(1)

    # Model name logic
    if args.model_name:
        output_name = args.model_name
    elif isinstance(input_items[0], Path):
        output_name = input_items[0].stem if input_items[0].is_file() else input_items[0].name
    elif isinstance(input_items[0], str):
        # For YouTube, use first link id/slug
        url = input_items[0]
        output_name = url.split("?")[0].split("/")[-1]
        if not output_name:
            output_name = "yt_model"
    else:
        output_name = "model"

    output_dir_base = Path(args.output_dir_base)
    output_dir = output_dir_base / output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Starting Headless XTTS Training Pipeline")
    print(f"Input: {input_arg}")
    print(f"Output Name: {output_name}")
    print(f"Output Directory: {output_dir}")
    print(f"-------------------------------------------")

    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Using temporary directory: {temp_dir}")
        prepared_files, total_dur = prepare_audios(input_items, temp_dir, warning_limit_minutes=50)
        if not prepared_files:
            print("Error: No valid audio files were prepared.")
            sys.exit(1)
        print(f"Prepared {len(prepared_files)} audio files; total duration: {total_dur/60:.2f} min.")

        # Step 1: Data Processing
        dataset_dir = output_dir / "dataset"
        status, train_csv_path, eval_csv_path = preprocess_dataset_headless(
            audio_files=prepared_files,
            language=args.lang,
            whisper_model_name=args.whisper_model,
            dataset_out_path=str(dataset_dir.resolve())
        )
        if "failed" in status.lower() or "error" in status.lower():
            print(f"Error during data processing: {status}")
            sys.exit(1)
        if not train_csv_path or not eval_csv_path:
            print("Error: Data processing did not return valid metadata paths. Exiting.")
            sys.exit(1)
        train_csv_path = Path(train_csv_path).resolve()
        eval_csv_path = Path(eval_csv_path).resolve()

        # Step 2: Training
        status, config_path, vocab_path, model_path, speaker_xtts_path, original_speaker_wav = train_model_headless(
             language=args.lang,
             train_csv_path=str(train_csv_path),
             eval_csv_path=str(eval_csv_path),
             num_epochs=args.epochs,
             batch_size=args.batch_size,
             grad_acumm=args.grad_acumm,
             output_path_base=str(output_dir.resolve()),
             max_audio_length_sec=args.max_audio_length,
             version=args.xtts_base_version,
             custom_model=args.custom_model
        )
        if "failed" in status.lower() or "error" in status.lower():
             print(f"Error during training: {status}")
             sys.exit(1)
        if not config_path or not vocab_path or not model_path or not original_speaker_wav:
             print("Error: Training step did not return all expected file paths. Exiting.")
             sys.exit(1)

        # Step 2.1: Reference WAVs
        success = create_reference_wavs(
             Path(original_speaker_wav).resolve(),
             output_dir.resolve(),
             output_name
        )
        if not success:
             print("Warning: Failed to create all reference WAVs.")

        final_reference_wav = output_dir / f"{output_name}.wav"

        # Step 3: Load Model
        final_config_path = Path(config_path).resolve()
        final_vocab_path = Path(vocab_path).resolve()
        final_speaker_xtts_path = Path(speaker_xtts_path).resolve() if speaker_xtts_path else None
        model_path = Path(model_path).resolve()
        status = load_model_headless(
            xtts_checkpoint=str(model_path),
            xtts_config=str(final_config_path),
            xtts_vocab=str(final_vocab_path),
            xtts_speaker=str(final_speaker_xtts_path) if final_speaker_xtts_path else None
        )
        if "failed" in status.lower() or "error" in status.lower():
            print(f"Error loading trained model: {status}")
            sys.exit(1)

        # Step 4: Generate Example TTS
        example_output_wav_path = output_dir / f"{output_name}_generated_example.wav"
        status, generated_wav = run_tts_headless(
            lang=args.lang,
            tts_text=args.example_text,
            speaker_audio_file=str(final_reference_wav.resolve()),
            output_wav_path=str(example_output_wav_path.resolve()),
            temperature=args.temperature,
            length_penalty=args.length_penalty,
            repetition_penalty=args.repetition_penalty,
            top_k=args.top_k,
            top_p=args.top_p,
            sentence_split=not args.no_sentence_split
        )
        if "failed" in status.lower() or "error" in status.lower():
            print(f"Error generating example TTS: {status}")

        # Step 5: Zip dataset and clean up intermediate dirs
        zip_path = output_dir / "dataset.zip"
        zip_dataset(dataset_dir, zip_path)
        shutil.rmtree(dataset_dir, ignore_errors=True)
        # Remove any 'run', 'ready', or temp dirs just in case
        shutil.rmtree(output_dir / "run", ignore_errors=True)
        shutil.rmtree(output_dir / "ready", ignore_errors=True)

    # End
    print(f"\n-------------------------------------------")
    print(f"Headless XTTS pipeline finished!")
    final_output_dir = output_dir.resolve()
    print(f"All output files are located in: {final_output_dir}")
    print(f"  - Final model: {final_output_dir / 'model.pth'}")
    print(f"  - Config: {final_output_dir / 'config.json'}")
    print(f"  - Vocab: {final_output_dir / 'vocab.json'}")
    print(f"  - Speaker: {final_output_dir / 'speaker.pth'} (if present)")
    print(f"  - Reference wav: {final_output_dir / (output_name + '.wav')}")
    print(f"  - 16kHz ref: {final_output_dir / (output_name + '_16000.wav')}")
    print(f"  - 24kHz ref: {final_output_dir / (output_name + '_24000.wav')}")
    print(f"  - Example wav: {final_output_dir / (output_name + '_generated_example.wav')}")
    print(f"  - Zipped dataset: {final_output_dir / 'dataset.zip'}")
    print(f"-------------------------------------------")

if __name__ == "__main__":
    main()
