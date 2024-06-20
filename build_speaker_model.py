import os
import sys

now_dir = os.getcwd()
sys.path.append(now_dir)
sys.path.append("%s/GPT-SoVITS" % (now_dir))
sys.path.append("%s/GPT-SoVITS/GPT_SoVITS" % (now_dir))

import argparse
import glob
import json
from datetime import datetime

import torch
import torchaudio
from GPT_SoVITS.text import cleaned_text_to_sequence
from safetensors.torch import save_file

exp_root = "GPT-SoVITS/logs"
gpt_weight_root = "GPT-SoVITS/GPT_weights"
sovits_weight_root = "GPT-SoVITS/SoVITS_weights"
gpt_weight_suffix = "-e15.ckpt"
sovits_weight_suffix = "_e8_*.pth"


def serialize_binary(binary_data):
    return torch.frombuffer(binary_data, dtype=torch.uint8)


def serialize_object(obj):
    data = json.dumps(obj).encode("utf-8")
    # obj is a python object
    return serialize_binary(data)


def build_model(exp_name, output_path, speaker_names):
    data_dir = "%s/%s" % (exp_root, exp_name)

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
    print(f"Speaker names: {speaker_names}")
    with open(phones_text_path, "r", encoding="utf8") as f:
        lines = f.readlines()
        for line in lines:
            audio_name, phones, _, norm_text = line.strip().split("\t")
            audio, sr = torchaudio.load("%s/%s" % (wav32dir, audio_name))
            audio_len = audio.shape[-1] / sr
            if audio_len < 5 or audio_len > 10:
                continue

            speaker_name = "_".join(audio_name.split("_")[:-1])
            speaker_name_wo_ext = audio_name.split(".")[0]
            if speaker_names is not None and speaker_name_wo_ext not in speaker_names:
                continue
            print(f"Embedding {speaker_name_wo_ext}")

            with open("%s/%s" % (wav32dir, audio_name), "rb") as f:
                audio_data = f.read()
            phones = cleaned_text_to_sequence(phones.split(" "))
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
        description="Pack speaker model data into safetensors file."
    )
    parser.add_argument("--exp_name", type=str, required=True, help="exp name")
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Directory containing input audio files",
    )
    parser.add_argument(
        "--speaker_filter_file",
        type=str,
        default=None,
        help="File containing speaker names to filter",
    )
    args = parser.parse_args()

    if args.speaker_filter_file is not None:
        with open(args.speaker_filter_file, "r") as f:
            speaker_names = [line.strip() for line in f.readlines()]
    else:
        speaker_names = None

    if args.output_path is not None:
        output_path = args.output_path
    else:
        output_path = f"{args.exp_name}.safetensors"

    build_model(args.exp_name, output_path, speaker_names)


if __name__ == "__main__":
    main()
