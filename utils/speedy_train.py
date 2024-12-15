import argparse
import os
import sys
import tempfile
from pathlib import Path
import shutil
import glob

import gradio as gr
import librosa.display
import numpy as np
import torch
import torchaudio
import traceback

from utils.formatter import format_audio_list, find_latest_best_model, list_audios
from utils.gpt_train import train_gpt
from utils.vits_train import train_vits_model
from utils.speedy_train import train_speedy_model  # NEW IMPORT

from faster_whisper import WhisperModel
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import requests

def download_file(url, destination):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(destination, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded file to {destination}")
        return destination
    except Exception as e:
        print(f"Failed to download the file: {e}")
        return None

def remove_log_file(file_path):
    log_file = Path(file_path)
    if log_file.exists() and log_file.is_file():
        log_file.unlink()

def clear_gpu_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

XTTS_MODEL = None

def create_zip(folder_path, zip_name):
    zip_path = os.path.join(tempfile.gettempdir(), f"{zip_name}.zip")
    if os.path.exists(zip_path):
        os.remove(zip_path)
    shutil.make_archive(zip_path.replace('.zip', ''), 'zip', folder_path)
    return zip_path

def get_model_zip(out_path):
    ready_folder = os.path.join(out_path, "ready")
    if os.path.exists(ready_folder):
        return create_zip(ready_folder, "optimized_model")
    return None

def get_dataset_zip(out_path):
    dataset_folder = os.path.join(out_path, "dataset")
    if os.path.exists(dataset_folder):
        return create_zip(dataset_folder, "dataset")
    return None

def load_model(xtts_checkpoint, xtts_config, xtts_vocab, xtts_speaker):
    global XTTS_MODEL
    clear_gpu_cache()
    if not xtts_checkpoint or not xtts_config or not xtts_vocab:
        return "You need to set the XTTS checkpoint, config, and vocab paths!"
    config = XttsConfig()
    config.load_json(xtts_config)
    XTTS_MODEL = Xtts.init_from_config(config)
    print("Loading XTTS model!")
    XTTS_MODEL.load_checkpoint(config, checkpoint_path=xtts_checkpoint, vocab_path=xtts_vocab, speaker_file_path=xtts_speaker, use_deepspeed=False)
    if torch.cuda.is_available():
        XTTS_MODEL.cuda()
    print("Model Loaded!")
    return "Model Loaded!"

def run_tts(lang, tts_text, speaker_audio_file, temperature, length_penalty, repetition_penalty, top_k, top_p, sentence_split, use_config):
    if XTTS_MODEL is None or not speaker_audio_file:
        return "You need to load the model and provide a speaker audio file!", None, None

    gpt_cond_latent, speaker_embedding = XTTS_MODEL.get_conditioning_latents(
        audio_path=speaker_audio_file,
        gpt_cond_len=XTTS_MODEL.config.gpt_cond_len,
        max_ref_length=XTTS_MODEL.config.max_ref_len,
        sound_norm_refs=XTTS_MODEL.config.sound_norm_refs
    )

    if use_config:
        out = XTTS_MODEL.inference(
            text=tts_text,
            language=lang,
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            temperature=XTTS_MODEL.config.temperature,
            length_penalty=XTTS_MODEL.config.length_penalty,
            repetition_penalty=XTTS_MODEL.config.repetition_penalty,
            top_k=XTTS_MODEL.config.top_k,
            top_p=XTTS_MODEL.config.top_p,
            enable_text_splitting=True
        )
    else:
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

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fp:
        out["wav"] = torch.tensor(out["wav"]).unsqueeze(0)
        out_path = fp.name
        torchaudio.save(out_path, out["wav"], 24000)
    return "Speech generated!", out_path, speaker_audio_file

def load_params_tts(out_path, version):
    out_path = Path(out_path)
    ready_model_path = out_path / "ready"

    vocab_path = ready_model_path / "vocab.json"
    config_path = ready_model_path / "config.json"
    speaker_path = ready_model_path / "speakers_xtts.pth"
    reference_path = ready_model_path / "reference.wav"

    model_path = ready_model_path / "model.pth"
    if not model_path.exists():
        model_path = ready_model_path / "unoptimize_model.pth"
        if not model_path.exists():
            return "Params for TTS not found", "", "", "", "", ""

    return "Params for TTS loaded", str(model_path), str(config_path), str(vocab_path), str(speaker_path), str(reference_path)

def preprocess_dataset(audio_path, audio_folder_path, language, whisper_model, out_path, train_csv, eval_csv, progress=gr.Progress(track_tqdm=True)):
    clear_gpu_cache()
    train_csv = ""
    eval_csv = ""
    out_dataset = os.path.join(out_path, "dataset")
    os.makedirs(out_dataset, exist_ok=True)

    if audio_folder_path:
        audio_files = list(list_audios(audio_folder_path))
    else:
        audio_files = audio_path

    if not audio_files:
        return "No audio files found!", "", ""
    else:
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            compute_type = "float16" if torch.cuda.is_available() else "float32"
            asr_model = WhisperModel(whisper_model, device=device, compute_type=compute_type)
            train_meta, eval_meta, audio_total_size = format_audio_list(audio_files, asr_model=asr_model, target_language=language, out_path=out_dataset, gradio_progress=progress)
        except:
            traceback.print_exc()
            error = traceback.format_exc()
            return f"Data processing interrupted due to an error:\n{error}", "", ""
    
    if audio_total_size < 120:
        message = "The sum of the duration of the audios is less than 2 minutes!"
        return message, "", ""

    print("Dataset Processed!")
    return "Dataset Processed!", train_meta, eval_meta

def optimize_model(out_path, clear_train_data):
    out_path = Path(out_path)
    ready_dir = out_path / "ready"
    run_dir = out_path / "run"
    dataset_dir = out_path / "dataset"

    if clear_train_data in {"run", "all"} and run_dir.exists():
        try:
            shutil.rmtree(run_dir)
        except PermissionError as e:
            print(f"Error deleting {run_dir}: {e}")

    if clear_train_data in {"dataset", "all"} and dataset_dir.exists():
        try:
            shutil.rmtree(dataset_dir)
        except PermissionError as e:
            print(f"Error deleting {dataset_dir}: {e}")

    model_path = ready_dir / "unoptimize_model.pth"
    if not model_path.is_file():
        alt_model_path = ready_dir / "model.pth"
        if alt_model_path.is_file():
            model_path = alt_model_path
        else:
            return "Unoptimized model not found in ready folder", ""

    checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
    if "optimizer" in checkpoint:
        del checkpoint["optimizer"]

    # remove dvae key if present (XTTS)
    if "model" in checkpoint:
        keys_to_remove = [k for k in checkpoint["model"].keys() if "dvae" in k]
        for k in keys_to_remove:
            del checkpoint["model"][k]

    if model_path.exists():
        os.remove(model_path)
    optimized_model = ready_dir / "model.pth"
    torch.save(checkpoint, optimized_model)
    clear_gpu_cache()

    return f"Model optimized and saved at {optimized_model}!", str(optimized_model)

def load_params(out_path):
    path_output = Path(out_path)
    dataset_path = path_output / "dataset"
    if not dataset_path.exists():
        return "The output folder does not exist!", "", "", None

    eval_train = dataset_path / "metadata_train.csv"
    eval_csv = dataset_path / "metadata_eval.csv"
    lang_file_path = dataset_path / "lang.txt"
    current_language = None
    if lang_file_path.exists():
        with open(lang_file_path, 'r', encoding='utf-8') as f:
            current_language = f.read().strip()
    clear_gpu_cache()
    return "The data has been updated", str(eval_train), str(eval_csv), current_language

def handle_training(custom_model, model_selection, version, lang, train_csv, eval_csv, num_epochs, batch_size, grad_acumm, out_path, max_audio_length):
    clear_gpu_cache()
    if model_selection == "XTTS":
        max_audio_length = int(max_audio_length * 22050)
        return train_gpt(custom_model, version, lang, num_epochs, batch_size, grad_acumm, train_csv, eval_csv, output_path=out_path, max_audio_length=max_audio_length)
    elif model_selection == "VITS":
        return train_vits_model(custom_model, lang, train_csv, eval_csv, num_epochs, batch_size, grad_acumm, out_path, max_audio_length)
    elif model_selection == "SpeedySpeech":
        return train_speedy_model(custom_model, lang, train_csv, eval_csv, num_epochs, batch_size, grad_acumm, out_path, max_audio_length)
    else:
        return "Invalid model selection!", "", "", "", "", ""

def toggle_inference_visibility(model_type):
    # 20 outputs total. XTTS visible, others hidden.
    if model_type == "XTTS":
        return tuple([gr.update(visible=True) for _ in range(20)])
    else:
        # VITS or SpeedySpeech
        return tuple([gr.update(visible=False) for _ in range(20)])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="XTTS, VITS, SpeedySpeech fine-tuning demo", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--whisper_model", type=str, default="large-v3")
    parser.add_argument("--audio_folder_path", type=str, default="")
    parser.add_argument("--share", action="store_true", default=False)
    parser.add_argument("--port", type=int, default=5003)
    parser.add_argument("--out_path", type=str, default=str(Path.cwd() / "finetune_models"))
    parser.add_argument("--num_epochs", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_acumm", type=int, default=1)
    parser.add_argument("--max_audio_length", type=int, default=11)
    args = parser.parse_args()

    with gr.Blocks(title=os.environ.get("APP_NAME", "Gradio")) as demo:
        with gr.Tab("1 - Data processing"):
            out_path = gr.Textbox(value=args.out_path, label="Output path:")
            upload_file = gr.File(file_count="multiple", label="Audio files for TTS training")
            audio_folder_path = gr.Textbox(label="Path to the folder with audio files (optional):", value=args.audio_folder_path)
            whisper_model = gr.Dropdown(value=args.whisper_model, choices=["large-v3","large-v2","large","medium","small"], label="Whisper Model")
            lang = gr.Dropdown(value="en", choices=["en","es","fr","de","it","pt","pl","tr","ru","nl","cs","ar","zh","hu","ko","ja"], label="Dataset Language")
            progress_data = gr.Label(label="Progress:")
            prompt_compute_btn = gr.Button("Step 1 - Create dataset")

        with gr.Tab("2 - Fine-tuning TTS Model"):
            model_selection = gr.Dropdown(label="Select TTS model to fine-tune", value="XTTS", choices=["XTTS","VITS","SpeedySpeech"])
            load_params_btn = gr.Button("Load Params from output folder")
            version = gr.Dropdown(label="XTTS base version (ignored if not XTTS)", value="v2.0.2", choices=["v2.0.3","v2.0.2","v2.0.1","v2.0.0","main"])
            train_csv = gr.Textbox(label="Train CSV:")
            eval_csv = gr.Textbox(label="Eval CSV:")
            custom_model = gr.Textbox(label="(Optional) Custom model.pth URL or path", value="")
            num_epochs = gr.Slider(label="Number of epochs:", minimum=1, maximum=100, step=1, value=args.num_epochs)
            batch_size = gr.Slider(label="Batch size:", minimum=2, maximum=512, step=1, value=args.batch_size)
            grad_acumm = gr.Slider(label="Grad accumulation steps:", minimum=1, maximum=128, step=1, value=args.grad_acumm)
            max_audio_length = gr.Slider(label="Max permitted audio size (sec):", minimum=2, maximum=20, step=1, value=args.max_audio_length)
            clear_train_data = gr.Dropdown(label="Clear train data after optimizing", value="none", choices=["none","run","dataset","all"])
            progress_train = gr.Label(label="Progress:")
            train_btn = gr.Button("Step 2 - Run the training")
            optimize_model_btn = gr.Button("Step 2.5 - Optimize the model")

        with gr.Tab("3 - Inference"):
            with gr.Row():
                with gr.Column() as col1:
                    load_params_tts_btn = gr.Button("Load params for TTS from output folder")
                    xtts_checkpoint = gr.Textbox(label="XTTS checkpoint path:")
                    xtts_config = gr.Textbox(label="XTTS config path:")
                    xtts_vocab = gr.Textbox(label="XTTS vocab path:")
                    xtts_speaker = gr.Textbox(label="XTTS speaker path:")
                    progress_load = gr.Label(label="Progress:")
                    load_btn = gr.Button("Step 3 - Load Fine-tuned XTTS model")

                with gr.Column() as col2:
                    speaker_reference_audio = gr.Textbox(label="Speaker reference audio:")
                    tts_language = gr.Dropdown(label="Language", value="en", choices=["en","es","fr","de","it","pt","pl","tr","ru","nl","cs","ar","zh","hu","ko","ja"])
                    tts_text = gr.Textbox(label="Input Text.", value="This model sounds really good and above all, it's reasonably fast.")
                    with gr.Accordion("Advanced settings", open=False):
                        temperature = gr.Slider(label="temperature", minimum=0, maximum=1, step=0.05, value=0.75)
                        length_penalty = gr.Slider(label="length_penalty", minimum=-10.0, maximum=10.0, step=0.5, value=1)
                        repetition_penalty = gr.Slider(label="repetition penalty", minimum=1, maximum=10, step=0.5, value=5)
                        top_k = gr.Slider(label="top_k", minimum=1, maximum=100, step=1, value=50)
                        top_p = gr.Slider(label="top_p", minimum=0, maximum=1, step=0.05, value=0.85)
                        sentence_split = gr.Checkbox(label="Enable text splitting", value=True)
                        use_config = gr.Checkbox(label="Use Inference settings from config", value=False)
                    tts_btn = gr.Button("Step 4 - Inference")

                    model_download_btn = gr.Button("Step 5 - Download Optimized Model ZIP")
                    dataset_download_btn = gr.Button("Step 5 - Download Dataset ZIP")
                    model_zip_file = gr.File(label="Download Optimized Model", interactive=False)
                    dataset_zip_file = gr.File(label="Download Dataset", interactive=False)

                with gr.Column() as col3:
                    progress_gen = gr.Label(label="Progress:")
                    tts_output_audio = gr.Audio(label="Generated Audio.")
                    reference_audio = gr.Audio(label="Reference audio used:")

            model_selection.change(
                fn=toggle_inference_visibility,
                inputs=[model_selection],
                outputs=[
                    load_params_tts_btn,
                    xtts_checkpoint,
                    xtts_config,
                    xtts_vocab,
                    xtts_speaker,
                    load_btn,
                    speaker_reference_audio,
                    tts_language,
                    tts_text,
                    temperature,
                    length_penalty,
                    repetition_penalty,
                    top_k,
                    top_p,
                    sentence_split,
                    use_config,
                    tts_btn,
                    progress_gen,
                    tts_output_audio,
                    reference_audio
                ]
            )

            prompt_compute_btn.click(
                fn=preprocess_dataset,
                inputs=[upload_file, audio_folder_path, lang, whisper_model, out_path, train_csv, eval_csv],
                outputs=[progress_data, train_csv, eval_csv],
            )

            load_params_btn.click(
                fn=load_params,
                inputs=[out_path],
                outputs=[progress_train, train_csv, eval_csv, lang]
            )

            train_btn.click(
                fn=handle_training,
                inputs=[custom_model, model_selection, version, lang, train_csv, eval_csv, num_epochs, batch_size, grad_acumm, out_path, max_audio_length],
                outputs=[progress_train, xtts_config, xtts_vocab, xtts_checkpoint, xtts_speaker, speaker_reference_audio],
            )

            optimize_model_btn.click(
                fn=optimize_model,
                inputs=[out_path, clear_train_data],
                outputs=[progress_train, xtts_checkpoint],
            )

            load_btn.click(
                fn=load_model,
                inputs=[xtts_checkpoint, xtts_config, xtts_vocab, xtts_speaker],
                outputs=[progress_load],
            )

            tts_btn.click(
                fn=run_tts,
                inputs=[tts_language, tts_text, speaker_reference_audio, temperature, length_penalty, repetition_penalty, top_k, top_p, sentence_split, use_config],
                outputs=[progress_gen, tts_output_audio, reference_audio],
            )

            load_params_tts_btn.click(
                fn=load_params_tts,
                inputs=[out_path, version],
                outputs=[progress_load, xtts_checkpoint, xtts_config, xtts_vocab, xtts_speaker, speaker_reference_audio],
            )

            model_download_btn.click(
                fn=get_model_zip,
                inputs=[out_path],
                outputs=[model_zip_file]
            )

            dataset_download_btn.click(
                fn=get_dataset_zip,
                inputs=[out_path],
                outputs=[dataset_zip_file]
            )

    demo.launch(share=args.share, debug=False, server_port=args.port)
