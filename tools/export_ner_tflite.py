#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import json
import os
from pathlib import Path

import hanlp
import tensorflow as tf


def load_labels(model_dir: Path):
    vocab_path = model_dir / "vocabs.json"
    with vocab_path.open("r", encoding="utf-8") as f:
        vocabs = json.load(f)
    return vocabs["tag_vocab"]["idx_to_token"]


def save_labels(labels, export_dir: Path):
    labels_path = export_dir / "labels.txt"
    with labels_path.open("w", encoding="utf-8") as f:
        for label in labels:
            f.write(label + "\n")
    return labels_path


def save_meta(component, labels, export_dir: Path):
    meta = {
        "max_seq_length": int(component.config.get("max_seq_length", 128)),
        "num_labels": int(len(labels)),
        "input_names": ["input_ids", "mask_ids", "token_type_ids"],
    }
    meta_path = export_dir / "meta.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta_path


def build_rep_dataset(component, rep_texts):
    inputs = []
    for text in rep_texts:
        text = text.strip()
        if not text:
            continue
        inputs.append(list(text))
    dataset = component.transform.inputs_to_dataset(
        inputs, gold=False, batch_size=1, shuffle=False, prefetch=0, cache=False
    )

    def gen():
        for batch in dataset:
            x = batch[0]
            yield [x[0], x[1], x[2]]

    return gen


def convert_tflite(saved_model_dir: Path, tflite_path: Path, quant, full_int8,
                   rep_texts, allow_select_tf_ops, lower_tensor_list_ops):
    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))

    if allow_select_tf_ops:
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS,
        ]
        converter.allow_custom_ops = True
        if lower_tensor_list_ops is not None:
            converter._experimental_lower_tensor_list_ops = lower_tensor_list_ops

    if quant == "float16":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    elif quant == "int8":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        if full_int8:
            if rep_texts is None:
                raise ValueError("--full-int8 requires --rep-data or --rep-text")
            converter.representative_dataset = rep_texts
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
    elif quant != "none":
        raise ValueError(f"Unsupported quant mode: {quant}")

    tflite_model = converter.convert()
    tflite_path.parent.mkdir(parents=True, exist_ok=True)
    with tflite_path.open("wb") as f:
        f.write(tflite_model)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        default="ner_albert_base_zh_msra_20200111_202919",
        help="HanLP model directory with meta.json/config.json",
    )
    parser.add_argument(
        "--export-dir",
        default="export/ner_albert_msra",
        help="Directory to save SavedModel and labels",
    )
    parser.add_argument(
        "--tflite-path",
        default="export/ner_albert_msra/model.tflite",
        help="Output .tflite file path",
    )
    parser.add_argument(
        "--quant",
        default="float16",
        choices=["none", "float16", "int8"],
        help="Quantization mode",
    )
    parser.add_argument(
        "--full-int8",
        action="store_true",
        help="Use full int8 (requires representative data).",
    )
    parser.add_argument(
        "--rep-data",
        default=None,
        help="Path to representative text file, one sentence per line.",
    )
    parser.add_argument(
        "--rep-text",
        default=None,
        help="Inline representative text, can be used multiple times.",
        action="append",
    )
    parser.add_argument(
        "--allow-select-tf-ops",
        action="store_true",
        help="Enable SELECT_TF_OPS when TFLite builtin ops are insufficient.",
    )
    parser.add_argument(
        "--lower-tensor-list-ops",
        action="store_true",
        help="Force lower TensorList ops (may fail for some transformer models).",
    )
    parser.add_argument(
        "--no-lower-tensor-list-ops",
        action="store_true",
        help="Disable lowering TensorList ops (often needed with SELECT_TF_OPS).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model_dir = Path(args.model_dir).resolve()
    export_dir = Path(args.export_dir).resolve()
    tflite_path = Path(args.tflite_path).resolve()

    component = hanlp.load(str(model_dir))
    export_dir.mkdir(parents=True, exist_ok=True)
    component.export_model_for_serving(str(export_dir), version=1, overwrite=True, show_hint=False)
    saved_model_dir = export_dir / "1"

    labels = load_labels(model_dir)
    save_labels(labels, export_dir)
    save_meta(component, labels, export_dir)

    rep_texts = None
    if args.rep_data:
        with open(args.rep_data, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        rep_texts = build_rep_dataset(component, lines)
    elif args.rep_text:
        rep_texts = build_rep_dataset(component, args.rep_text)

    lower_tensor_list_ops = None
    if args.lower_tensor_list_ops:
        lower_tensor_list_ops = True
    if args.no_lower_tensor_list_ops:
        lower_tensor_list_ops = False

    convert_tflite(
        saved_model_dir=saved_model_dir,
        tflite_path=tflite_path,
        quant=args.quant,
        full_int8=args.full_int8,
        rep_texts=rep_texts,
        allow_select_tf_ops=args.allow_select_tf_ops,
        lower_tensor_list_ops=lower_tensor_list_ops,
    )


if __name__ == "__main__":
    main()
