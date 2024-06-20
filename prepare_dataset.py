import os
import sys
import traceback

now_dir = os.getcwd()
sys.path.append(now_dir)
sys.path.append("%s/GPT-SoVITS" % (now_dir))
sys.path.append("%s/GPT-SoVITS/GPT_SoVITS" % (now_dir))

import argparse
import glob
import json
import time
from datetime import datetime

import torch
import torchaudio
from demucs import pretrained
from demucs.apply import apply_model
from df import enhance, init_df
from GPT_SoVITS import utils
from GPT_SoVITS.feature_extractor import cnhubert
from GPT_SoVITS.module.models import SynthesizerTrn
from GPT_SoVITS.text import cleaned_text_to_sequence
from GPT_SoVITS.text.cleaner import clean_text
from safetensors.torch import save_file
from transformers import (
    AutoModelForMaskedLM,
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    AutoTokenizer,
    pipeline,
)

cnhubert_path = "GPT-SoVITS/GPT_SoVITS/pretrained_models/chinese-hubert-base"
bert_path = "GPT-SoVITS/GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"
s2config_path = "GPT-SoVITS/GPT_SoVITS/configs/s2.json"
pretrained_s2G = "GPT-SoVITS/GPT_SoVITS/pretrained_models/s2G488k.pth"

gpt_weight_root = "GPT-SoVITS/GPT_weights"
sovits_weight_root = "GPT-SoVITS/SoVITS_weights"
gpt_weight_suffix = "-e15.ckpt"
sovits_weight_suffix = "_e8_*.pth"


def process_text(inp_text, opt_dir, bert_pretrained_dir, is_half):
    device = "cuda"
    path_text = "%s/2-name2text.txt" % opt_dir
    bert_dir = "%s/3-bert" % (opt_dir)
    os.makedirs(bert_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(bert_pretrained_dir)
    bert_model = AutoModelForMaskedLM.from_pretrained(bert_pretrained_dir)
    if is_half == True:
        bert_model = bert_model.half().to(device)
    else:
        bert_model = bert_model.to(device)

    def get_bert_feature(text, word2ph):
        with torch.no_grad():
            inputs = tokenizer(text, return_tensors="pt")
            for i in inputs:
                inputs[i] = inputs[i].to(device)
            res = bert_model(**inputs, output_hidden_states=True)
            res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]

        assert len(word2ph) == len(text)
        phone_level_feature = []
        for i in range(len(word2ph)):
            repeat_feature = res[i].repeat(word2ph[i], 1)
            phone_level_feature.append(repeat_feature)

        phone_level_feature = torch.cat(phone_level_feature, dim=0)

        return phone_level_feature.T

    def process(name, text, lang):
        name = os.path.basename(name)
        phones, word2ph, norm_text = clean_text(
            text.replace("%", "-").replace("￥", ","), lang
        )
        path_bert = "%s/%s.pt" % (bert_dir, name)
        if os.path.exists(path_bert) == False and lang == "zh":
            bert_feature = get_bert_feature(norm_text, word2ph)
            assert bert_feature.shape[-1] == len(phones)
            torch.save(bert_feature, path_bert)
        phones = " ".join(phones)
        return (name, phones, word2ph, norm_text)

    with open(inp_text, "r", encoding="utf8") as f:
        lines = f.read().strip("\n").split("\n")

    language_v1_to_language_v2 = {
        "ZH": "zh",
        "zh": "zh",
        "JP": "ja",
        "jp": "ja",
        "JA": "ja",
        "ja": "ja",
        "EN": "en",
        "en": "en",
        "En": "en",
    }
    with open(path_text, "w", encoding="utf8") as f:
        for line in lines:
            try:
                wav_name, spk_name, language, text = line.split("|")
                name, phones, word2ph, norm_text = process(
                    wav_name, text, language_v1_to_language_v2.get(language, language)
                )
                f.write("%s\t%s\t%s\t%s\n" % (name, phones, word2ph, norm_text))
            except:
                print(line, traceback.format_exc())


def process_audio(
    inp_text,
    opt_dir,
    hubert_pretrained_dir,
    s2config_path,
    pretrained_s2G_path,
    is_half,
):
    hubert_dir = "%s/4-cnhubert" % (opt_dir)
    wav32dir = "%s/5-wav32k" % (opt_dir)
    semantic_path = "%s/6-name2semantic.tsv" % (opt_dir)

    os.makedirs(hubert_dir, exist_ok=True)
    os.makedirs(wav32dir, exist_ok=True)
    maxx = 0.95
    alpha = 0.5
    device = "cuda"
    cnhubert.cnhubert_base_path = hubert_pretrained_dir
    model = cnhubert.get_model()
    if is_half:
        model = model.half()
    model = model.to(device)

    hps = utils.get_hparams_from_file(s2config_path)
    vq_model = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model,
    )
    if is_half:
        vq_model = vq_model.half()
    vq_model.eval()
    vq_model.load_state_dict(
        torch.load(pretrained_s2G_path, map_location="cpu")["weight"], strict=False
    )
    vq_model.to(device)

    def name2go(wav_name, wav_path):
        hubert_path = "%s/%s.pt" % (hubert_dir, wav_name)
        audio, sr = torchaudio.load(wav_path)
        # reduce the channel to 1
        audio = audio.mean(dim=0).unsqueeze(0)

        audio32k = torchaudio.functional.resample(audio, sr, 32000)
        audio32k = audio32k / audio32k.abs().max() * (maxx * alpha + 1 - alpha)

        audio16k = torchaudio.functional.resample(audio, sr, 16000)
        audio16k = audio16k / audio16k.abs().max() * (maxx * alpha + 1 - alpha)
        audio16k *= 1145.14

        if is_half:
            audio16k = audio16k.half()
        audio16k = audio16k.to(device)

        ssl = model.model(audio16k)["last_hidden_state"].transpose(
            1, 2
        )  # torch.Size([1, 768, 215])

        audio16k.cpu()

        out_path = "%s/%s" % (wav32dir, wav_name)
        torchaudio.save(out_path, audio32k, 32000)
        torch.save(ssl, hubert_path)

        codes = vq_model.extract_latent(ssl)
        semantic = " ".join([str(i) for i in codes[0, 0, :].tolist()])

        audio16k = None
        ssl = None
        torch.cuda.empty_cache()

        return semantic

    with open(semantic_path, "w", encoding="utf8") as semantic_f:
        with open(inp_text, "r", encoding="utf8") as inp_f:
            lines = inp_f.read().strip("\n").split("\n")

            for line in lines:
                try:
                    wav_path, spk_name, language, text = line.split("|")
                    wav_name = os.path.basename(wav_path)
                    semantic = name2go(wav_name, wav_path)
                    semantic_f.write("%s\t%s\n" % (wav_name, semantic))

                except:
                    print(line, traceback.format_exc())


def prepare_training_dataset(
    inp_text,
    output_dir,
    if_half=True,
):
    os.makedirs(output_dir, exist_ok=True)
    process_text(inp_text, output_dir, bert_path, if_half)
    torch.cuda.empty_cache()

    process_audio(
        inp_text,
        output_dir,
        cnhubert_path,
        s2config_path,
        pretrained_s2G,
        if_half,
    )


def clean(input_dir, output_dir, max_segs_per_speaker=100):
    # grep all mp3, wav and flac files
    input_files = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith((".mp3", ".wav", ".flac", ".mp4", ".webm", ".m4a")):
                input_files.append(os.path.join(root, file))

    print(f"Found {len(input_files)} audio files")

    if not os.path.exists(f"{output_dir}/audio"):
        os.makedirs(f"{output_dir}/audio")

    # speaker_name = input_path.split("/")[-1].split(".")[0]
    # print(f"Processing {speaker_name}")

    since = time.time()

    sp_model = pretrained.get_model(pretrained.DEFAULT_MODEL)
    df_model, df_state, _ = init_df(post_filter=True)

    def adjust_volume(waveform, target_rms):
        current_rms = torch.sqrt(torch.mean(waveform**2))
        if current_rms == 0:  # Avoid division by zero
            return waveform
        adjustment_factor = target_rms / current_rms
        adjusted_waveform = waveform * adjustment_factor

        # Ensure no clipping occurs
        peak_amplitude = adjusted_waveform.abs().max()
        if peak_amplitude > 1:
            adjusted_waveform /= peak_amplitude  # Normalize to the max peak

        return adjusted_waveform

    def enhance_df(df_model, df_state, audio, samplerate):
        audio_list = []

        # segment vocals to 1 minutes clips to enhance
        for i in range(0, audio.shape[1], 60 * samplerate):
            vocal = vocals[:, i : i + 60 * samplerate]
            enhanced_audio = enhance(df_model, df_state, vocal)
            audio_list.append(enhanced_audio)

        return torch.cat(audio_list, dim=1)

    def transcribe_audio(asr_model, processor, audio_path):
        asr_transcribe = pipeline(
            "automatic-speech-recognition",
            model=asr_model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            max_new_tokens=128,
            chunk_length_s=30,
            batch_size=16,
            return_timestamps=True,
            torch_dtype=torch.float16,
            device="cuda",
        )

        result = asr_transcribe(
            audio_path, return_timestamps=True, generate_kwargs={"language": "english"}
        )

        output_segs = []
        text = ""
        start = None
        end = None
        for segment in result["chunks"]:
            text += segment["text"] + " "
            if start is None:
                start = segment["timestamp"][0]
                if end is not None and start - end > 1:
                    # Since we already separated the vocals, and removed silence more than 1s,
                    # the gap between segments is too large
                    start = end
            end = segment["timestamp"][1]
            if end is None:
                end = audio_len

            if end - start < 5:
                continue

            if (
                segment["text"][-1] not in [".", "!", "?", "。", "！", "？"]
                and end - start < 30
            ):
                continue

            output_segs.append({"start": start, "end": end, "text": text})
            start = None
            text = ""

        if text:
            output_segs.append({"start": start, "end": end, "text": text})

        return output_segs

    def apply_fade(audio, sr, fade_in_duration, fade_out_duration):
        num_channels, num_samples = audio.shape
        fade_in_samples = int(sr * fade_in_duration)
        fade_out_samples = int(sr * fade_out_duration)

        # Create fade in and fade out tensors
        fade_in = (
            torch.linspace(0.0, 1.0, fade_in_samples)
            .unsqueeze(0)
            .repeat(num_channels, 1)
        )
        fade_out = (
            torch.linspace(1.0, 0.0, fade_out_samples)
            .unsqueeze(0)
            .repeat(num_channels, 1)
        )

        # Apply fade in to the beginning of the audio
        audio[:, :fade_in_samples] *= fade_in

        # Apply fade out to the end of the audio
        audio[:, -fade_out_samples:] *= fade_out

        # Similar to inference, add 300ms silence after each sentence.
        audio = torch.cat(
            [
                audio,
                torch.zeros_like(audio[:, : int(sr * 0.3)]),
            ],
            dim=1,
        )

        return audio

    results = {}

    for input_path in input_files:
        speaker_name = input_path.split("/")[-1].split(".")[0]
        print(f"Processing {speaker_name}")
        enhanced_audio_path = f"{output_dir}/{speaker_name}_enhanced.flac"
        silence_removed_audio_path = f"{output_dir}/{speaker_name}_silence_removed.flac"

        if not os.path.exists(enhanced_audio_path):
            audio, sr = torchaudio.load(input_path)
            audio = torchaudio.functional.resample(audio, sr, sp_model.samplerate)
            # if audio is mono, convert it to stereo
            if audio.shape[0] == 1:
                audio = torch.cat([audio, audio], dim=0)

            out = apply_model(sp_model, audio.unsqueeze(0), device="cuda")[0]
            vocals = out[3]
            # vocal_audio_path = f"{output_dir}/{speaker_name}_vocal.flac"
            # torchaudio.save(vocal_audio_path, vocals, sp_model.samplerate)

            # adjust volume to 0.1, (-20dB)
            vocals = adjust_volume(vocals, 0.1)
            vocals = torchaudio.functional.resample(
                vocals, sp_model.samplerate, df_state.sr()
            )

            enhanced_audio = enhance_df(df_model, df_state, vocals, sp_model.samplerate)

            # TODO: may harm male voice
            # cutoff_freq = 120
            # enhanced_audio = torchaudio.functional.highpass_biquad(
            #     enhanced_audio, df_state.sr(), cutoff_freq
            # )

            torchaudio.save(enhanced_audio_path, enhanced_audio, df_state.sr())

        if not os.path.exists(silence_removed_audio_path):
            ffmpeg_cmd = f'ffmpeg -y -hide_banner -loglevel error -i {enhanced_audio_path} -af "silenceremove=start_periods=1:stop_periods=-1:stop_duration=1:stop_threshold=-50dB" {silence_removed_audio_path}'
            os.system(ffmpeg_cmd)

        results[speaker_name] = {
            "input": input_path,
            "enhanced": enhanced_audio_path,
            "silence_removed": silence_removed_audio_path,
        }

    # unload the enhanced model
    sp_model = None
    df_model = None
    torch.cuda.empty_cache()

    print(f"Separated and enhance in {time.time() - since:.2f}s")

    language = "en"

    model_id = "openai/whisper-large-v3"

    asr_model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        use_safetensors=True,
        attn_implementation="flash_attention_2",
    )
    asr_model.to("cuda")

    processor = AutoProcessor.from_pretrained(model_id)

    ref_dict = []

    # The TTS annotation .list file format:
    # vocal_path|speaker_name|language|text
    with open(f"{output_dir}/tts_annotation.list", "w") as f:
        for speaker_name, result in results.items():
            print(f"Processing {speaker_name}")
            audio_full, sr = torchaudio.load(result["silence_removed"])
            audio_len = audio_full.shape[1] / sr
            segs = transcribe_audio(asr_model, processor, result["silence_removed"])
            num_choosen = 0

            for i, seg in enumerate(segs):
                start = seg["start"]
                end = seg["end"]
                text = seg["text"]
                print(f"{speaker_name}_{i:03d} [{start:.2f}s -> {end:.2f}s] {text}")
                if end - start < 0.3:
                    print(f"Audio too short, skip part {i}")
                    continue

                out_path = f"{output_dir}/audio/{speaker_name}_{i:03}.flac"
                cropped_audio = audio_full[:, int(start * sr) : int(end * sr)]
                cropped_audio = apply_fade(
                    cropped_audio, sr, fade_in_duration=0.1, fade_out_duration=0.1
                )

                torchaudio.save(out_path, cropped_audio, sr)
                seg["audio_path"] = out_path

                cropped_audio_len = cropped_audio.shape[1] / sr
                if cropped_audio_len > 5 and cropped_audio_len < 10:
                    f.write(f"{out_path}|{speaker_name}|{language}|{text}\n")
                    # ref_dict.append(
                    #     {
                    #         "audio_path": out_path,
                    #         "text": text,
                    #         "speaker_name": speaker_name,
                    #         "idx": i,
                    #     }
                    # )
                    num_choosen += 1
                    if num_choosen >= max_segs_per_speaker:
                        break

    # unload the asr model
    asr_model = None
    processor = None
    torch.cuda.empty_cache()

    print(f"ASR in {time.time() - since:.2f}s")


def serialize_binary(binary_data):
    return torch.frombuffer(binary_data, dtype=torch.uint8)


def serialize_object(obj):
    data = json.dumps(obj).encode("utf-8")
    # obj is a python object
    return serialize_binary(data)


def pack_model(training_root, exp_name, output_path):
    data_dir = "%s/%s" % (training_root, exp_name)

    phones_text_path = "%s/2-name2text.txt" % data_dir
    bert_dir = "%s/3-bert" % data_dir
    wav32dir = "%s/5-wav32k" % data_dir

    semantic_path = "%s/6-name2semantic.tsv" % data_dir
    semantic_dict = {}

    gpt_weight_path = "%s/%s%s" % (gpt_weight_root, exp_name, gpt_weight_suffix)
    paths = glob.glob(gpt_weight_path)
    if len(paths) == 0:
        print(f"no gpt weight found for {exp_name}")
        return
    gpt_weight_path = paths[-1]

    sovits_weight_path = "%s/%s%s" % (
        sovits_weight_root,
        exp_name,
        sovits_weight_suffix,
    )
    paths = glob.glob(sovits_weight_path)
    if len(paths) == 0:
        print(f"no sovits weight found for {exp_name}")
        return
    sovits_weight_path = paths[-1]

    with open(semantic_path, "r", encoding="utf8") as semantic_f:
        lines = semantic_f.readlines()
        for line in lines:
            audio_name, semantic = line.strip().split("\t")
            semantic = [int(i) for i in semantic.split(" ")]
            semantic_dict[audio_name] = semantic

    output_data = {}

    with open(phones_text_path, "r", encoding="utf8") as f:
        lines = f.readlines()
        for line in lines:
            audio_name, phones, _, norm_text = line.strip().split("\t")
            audio, sr = torchaudio.load("%s/%s" % (wav32dir, audio_name))
            audio_len = audio.shape[-1] / sr
            if audio_len < 5 or audio_len > 10:
                continue
            with open("%s/%s" % (wav32dir, audio_name), "rb") as f:
                audio_data = f.read()
            phones = cleaned_text_to_sequence(phones.split(" "))
            speaker_name = "_".join(audio_name.split("_")[:-1])
            idx = audio_name.split("_")[-1].split(".")[0]
            bert_path = "%s/%s.pt" % (bert_dir, audio_name)
            if os.path.exists(bert_path):
                bert_features = torch.load(bert_path)
            else:
                bert_features = torch.zeros((1024, len(phones)))

            if audio_name in semantic_dict:
                semantic = semantic_dict[audio_name]
            else:
                print(f"semantic not found for {audio_name}")
                semantic = None

            item = {
                "speaker": serialize_object(speaker_name),
                "idx": serialize_object(idx),
                "audio_data": serialize_binary(audio_data),
                "phones": torch.tensor(phones),
                "norm_text": serialize_object(norm_text),
                "semantic": torch.tensor(semantic),
                "bert_features": bert_features,
            }
            for k, v in item.items():
                output_data["%s.%s" % (audio_name, k)] = v

    with open(gpt_weight_path, "rb") as f:
        gpt_weight = f.read()

    with open(sovits_weight_path, "rb") as f:
        sovits_weight = f.read()

    output_data["gpt_weight"] = serialize_binary(gpt_weight)
    output_data["sovits_weight"] = serialize_binary(sovits_weight)

    save_file(
        output_data,
        output_path,
        {
            "exp_name": exp_name,
            "version": "2.0",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
    )


def main():
    parser = argparse.ArgumentParser(
        description="WebRTC audio / video / data-channels demo"
    )
    parser.add_argument("--exp_name", type=str, required=True, help="exp name")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing input audio files",
    )
    args = parser.parse_args()

    exp_name = args.exp_name
    input_dir = args.input_dir

    dataset_root = "/root/dataset"
    training_root = "GPT-SoVITS/logs"
    ref_data_root = "/root"

    clean(input_dir, f"{dataset_root}/{exp_name}")
    prepare_training_dataset(
        f"{dataset_root}/{exp_name}/tts_annotation.list", f"{training_root}/{exp_name}"
    )
    # train the model

    # pack_model(training_root, exp_name, f"{ref_data_root}/{exp_name}.ref.safetensors")


if __name__ == "__main__":
    main()
