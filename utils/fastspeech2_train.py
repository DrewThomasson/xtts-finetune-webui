# utils/fastspeech2_train.py
import logging
import os
import gc
from pathlib import Path
import shutil
import traceback
import torch
import torchaudio
import requests
import subprocess

from trainer import Trainer, TrainerArgs
from TTS.config.shared_configs import BaseAudioConfig, BaseDatasetConfig
from TTS.tts.configs.fastspeech2_config import Fastspeech2Config
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.forward_tts import ForwardTTS
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor
from TTS.utils.manage import ModelManager
from utils.formatter import find_latest_best_model

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

def train_fastspeech2_model(custom_model, language, train_csv, eval_csv, num_epochs, batch_size, grad_acumm, output_path, max_audio_length):
    """
    Train a FastSpeech2 model using given parameters and dataset.
    Similar approach to vits/speedy_train/gpt_train.
    """

    if not train_csv or not eval_csv:
        return "You need to run data processing or set Train/Eval CSV fields!", "", "", "", "", ""

    output_path = Path(output_path)
    run_dir = output_path / "run"
    ready_dir = output_path / "ready"
    dataset_dir = output_path / "dataset"

    if not dataset_dir.exists():
        return "Dataset folder not found! Run data processing first.", "", "", "", "", ""

    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    train_csv_path = Path(train_csv)
    eval_csv_path = Path(eval_csv)

    if not train_csv_path.exists() or not eval_csv_path.exists():
        return "Train/Eval CSV files not found!", "", "", "", "", ""

    # Check for language consistency
    lang_file_path = dataset_dir / "lang.txt"
    if lang_file_path.exists():
        with open(lang_file_path, 'r', encoding='utf-8') as f:
            current_language = f.read().strip()
            if current_language != language:
                language = current_language

    # Setup FastSpeech2 config
    audio_config = BaseAudioConfig(
        sample_rate=22050,
        do_trim_silence=True,
        trim_db=60.0,
        signal_norm=False,
        mel_fmin=0.0,
        mel_fmax=8000,
        spec_gain=1.0,
        log_func="np.log",
        ref_level_db=20,
        preemphasis=0.0,
    )

    dataset_config = BaseDatasetConfig(
        formatter="coqui",
        dataset_name="ft_dataset",
        path=str(dataset_dir),
        meta_file_train=train_csv_path.name,
        meta_file_val=eval_csv_path.name,
        language=language
    )

    config = Fastspeech2Config(
        run_name="fastspeech2_finetune",
        audio=audio_config,
        batch_size=batch_size,
        eval_batch_size=batch_size,
        num_loader_workers=8,
        num_eval_loader_workers=4,
        compute_input_seq_cache=True,
        compute_f0=True,
        f0_cache_path=os.path.join(str(output_path), "f0_cache"),
        compute_energy=True,
        energy_cache_path=os.path.join(str(output_path), "energy_cache"),
        run_eval=True,
        test_delay_epochs=-1,
        epochs=num_epochs,
        text_cleaner="multilingual_cleaners",
        use_phonemes=False,
        phoneme_cache_path=os.path.join(str(output_path), "phoneme_cache"),
        precompute_num_workers=4,
        print_step=50,
        print_eval=False,
        mixed_precision=False,
        max_seq_len=500000,
        output_path=str(run_dir),
        datasets=[dataset_config],
    )

    # compute alignments if needed (if config.model_args.use_aligner=False)
    # If aligner not used, we may need attention masks from a pretrained model.
    if not config.model_args.use_aligner:
        manager = ModelManager()
        model_path, config_path, _ = manager.download_model("tts_models/en/ljspeech/tacotron2-DCA")
        # Adjust paths if needed
        cmd = f"python TTS/bin/compute_attention_masks.py --model_path {model_path} --config_path {config_path} --dataset coqui --dataset_metafile {train_csv_path.name} --data_path {dataset_dir} --use_cuda"
        print("Computing attention masks (if needed)...")
        subprocess.run(cmd, shell=True, check=False)

    ap = AudioProcessor.init_from_config(config)
    tokenizer, config = TTSTokenizer.init_from_config(config)

    train_samples, eval_samples = load_tts_samples(
        dataset_config,
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size,
    )

    model = ForwardTTS(config, ap, tokenizer, speaker_manager=None)

    # If custom_model is a URL, download it
    if custom_model and custom_model.startswith("http"):
        custom_model_path = download_file(custom_model, "custom_model_fastspeech2.pth")
        if not custom_model_path:
            return "Failed to download custom FastSpeech2 model!", "", "", "", "", ""
        custom_model = custom_model_path

    # Load custom checkpoint if provided
    if custom_model and os.path.isfile(custom_model):
        print(f"Loading custom FastSpeech2 model from: {custom_model}")
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
        return f"FastSpeech2 training was interrupted due to an error:\n{error}", "", "", "", "", ""

    best_ckpt = find_latest_best_model(str(run_dir))
    if best_ckpt is None:
        candidates = list((run_dir / "checkpoints").glob("*.pth"))
        if candidates:
            best_ckpt = str(candidates[-1])
        else:
            return "No checkpoints found after training!", "", "", "", "", ""

    ready_dir.mkdir(exist_ok=True, parents=True)
    shutil.copy(best_ckpt, ready_dir / "unoptimize_model.pth")

    # Create placeholders for consistency
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
