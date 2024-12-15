# utils/vits_train.py
import logging
import os
import gc
from pathlib import Path
import shutil
import traceback
import torch
import requests

from trainer import Trainer, TrainerArgs
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.configs.vits_config import VitsConfig, VitsAudioConfig
from TTS.tts.models.vits import Vits
from TTS.tts.datasets import load_tts_samples
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor

from utils.formatter import find_latest_best_model
import torchaudio

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

def train_vits_model(custom_model, language, train_csv, eval_csv, num_epochs, batch_size, grad_acumm, output_path, max_audio_length):
    """
    Train a VITS model using the given parameters and dataset.
    """

    if not train_csv or not eval_csv:
        return "You need to run the data processing step or manually set Train CSV and Eval CSV fields!", "", "", "", "", ""

    output_path = Path(output_path)
    run_dir = output_path / "run"
    ready_dir = output_path / "ready"
    dataset_dir = output_path / "dataset"

    if not dataset_dir.exists():
        return "Dataset folder not found! Run data processing step first.", "", "", "", "", ""

    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    train_csv_path = Path(train_csv)
    eval_csv_path = Path(eval_csv)

    if not train_csv_path.exists() or not eval_csv_path.exists():
        return "Train/Eval CSV files not found!", "", "", "", "", ""

    lang_file_path = dataset_dir / "lang.txt"
    if lang_file_path.exists():
        with open(lang_file_path, 'r', encoding='utf-8') as f:
            current_language = f.read().strip()
            if current_language != language:
                language = current_language

    # Set up VITS config (customize as needed)
    audio_config = VitsAudioConfig(
        sample_rate=22050,
        win_length=1024,
        hop_length=256,
        num_mels=80,
        mel_fmin=0,
        mel_fmax=None
    )

    dataset_config = BaseDatasetConfig(
        formatter="coqui", 
        dataset_name="ft_dataset", 
        path=str(dataset_dir), 
        meta_file_train=train_csv_path.name,
        meta_file_val=eval_csv_path.name,
        language=language
    )

    config = VitsConfig(
        audio=audio_config,
        run_name="vits_ft",
        batch_size=batch_size,
        eval_batch_size=batch_size,
        batch_group_size=5,
        num_loader_workers=8,
        num_eval_loader_workers=4,
        run_eval=True,
        test_delay_epochs=-1,
        epochs=num_epochs,
        text_cleaner="multilingual_cleaners",
        use_phonemes=False,
        compute_input_seq_cache=True,
        print_step=25,
        print_eval=True,
        mixed_precision=False,
        output_path=str(run_dir),
        datasets=[dataset_config],
        cudnn_benchmark=False,
    )

    ap = AudioProcessor.init_from_config(config)
    tokenizer, config = TTSTokenizer.init_from_config(config)

    train_samples, eval_samples = load_tts_samples(
        dataset_config,
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size,
    )

    model = Vits(config, ap, tokenizer, speaker_manager=None)

    # If custom_model is a URL, download it
    if custom_model and custom_model.startswith("http"):
        custom_model_path = download_file(custom_model, "custom_model_vits.pth")
        if not custom_model_path:
            return "Failed to download custom VITS model!", "", "", "", "", ""
        custom_model = custom_model_path

    # Load custom checkpoint if provided
    if custom_model and os.path.isfile(custom_model):
        print(f"Loading custom VITS model from: {custom_model}")
        model.load_checkpoint(config, custom_model, eval=False, strict=False)

    trainer = Trainer(
        TrainerArgs(
            restore_path=None,
            skip_train_epoch=False,
            start_with_eval=False,
            grad_accum_steps=grad_acumm,
        ),
        config,
        run_dir,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )

    try:
        trainer.fit()
    except:
        traceback.print_exc()
        error = traceback.format_exc()
        return f"VITS training was interrupted due to an error:\n{error}", "", "", "", "", ""

    best_ckpt = find_latest_best_model(str(run_dir))
    if best_ckpt is None:
        candidates = list((run_dir / "checkpoints").glob("*.pth"))
        if candidates:
            best_ckpt = str(candidates[-1])
        else:
            return "No checkpoints found after training!", "", "", "", "", ""

    ready_dir.mkdir(exist_ok=True, parents=True)
    shutil.copy(best_ckpt, ready_dir / "unoptimize_model.pth")

    vocab_file = ready_dir / "vocab.json"
    if not vocab_file.exists():
        with open(vocab_file, "w", encoding="utf-8") as f:
            f.write("{}")

    config_file = ready_dir / "config.json"
    config.save_json(config_file)

    speaker_xtts_path = ready_dir / "speakers_xtts.pth"
    if not speaker_xtts_path.exists():
        torch.save({}, speaker_xtts_path)

    samples_len = [len(item["text"].split(" ")) for item in train_samples]
    longest_text_idx = samples_len.index(max(samples_len))
    speaker_ref = train_samples[longest_text_idx]["audio_file"]
    speaker_ref_path = dataset_dir / speaker_ref
    ref_audio_ready = ready_dir / "reference.wav"
    if speaker_ref_path.exists():
        shutil.copy(speaker_ref_path, ref_audio_ready)
    else:
        sr = 22050
        silence = torch.zeros(int(sr * 0.5))
        torchaudio.save(str(ref_audio_ready), silence.unsqueeze(0), sr)

    del model, trainer, train_samples, eval_samples
    gc.collect()

    return "Model training done!", str(config_file), str(vocab_file), str(ready_dir / "unoptimize_model.pth"), str(speaker_xtts_path), str(ref_audio_ready)
