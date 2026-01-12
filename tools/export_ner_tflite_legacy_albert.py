#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModel


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


def save_meta(max_seq_length: int, num_labels: int, export_dir: Path):
    meta = {
        "max_seq_length": int(max_seq_length),
        "num_labels": int(num_labels),
        "input_names": ["input_ids", "mask_ids", "token_type_ids"],
    }
    meta_path = export_dir / "meta.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta_path


def assign_weights(base_model, h5_bert):
    hf_weights = {w.name: w for w in base_model.weights}

    mapping = {
        "tf_albert_model/albert/embeddings/word_embeddings/weight:0":
            "bert/embeddings/word_embeddings/embeddings:0",
        "tf_albert_model/albert/embeddings/token_type_embeddings/embeddings:0":
            "bert/embeddings/token_type_embeddings/embeddings:0",
        "tf_albert_model/albert/embeddings/position_embeddings/embeddings:0":
            "bert/embeddings/position_embeddings/embeddings:0",
        "tf_albert_model/albert/embeddings/LayerNorm/gamma:0":
            "bert/embeddings/LayerNorm/gamma:0",
        "tf_albert_model/albert/embeddings/LayerNorm/beta:0":
            "bert/embeddings/LayerNorm/beta:0",
        "tf_albert_model/albert/encoder/embedding_hidden_mapping_in/kernel:0":
            "bert/embeddings/word_embeddings_projector/projector:0",
        "tf_albert_model/albert/encoder/embedding_hidden_mapping_in/bias:0":
            "bert/embeddings/word_embeddings_projector/bias:0",
        "tf_albert_model/albert/encoder/albert_layer_groups_._0/albert_layers_._0/attention/query/kernel:0":
            "bert/encoder/layer_shared/attention/self/query/kernel:0",
        "tf_albert_model/albert/encoder/albert_layer_groups_._0/albert_layers_._0/attention/query/bias:0":
            "bert/encoder/layer_shared/attention/self/query/bias:0",
        "tf_albert_model/albert/encoder/albert_layer_groups_._0/albert_layers_._0/attention/key/kernel:0":
            "bert/encoder/layer_shared/attention/self/key/kernel:0",
        "tf_albert_model/albert/encoder/albert_layer_groups_._0/albert_layers_._0/attention/key/bias:0":
            "bert/encoder/layer_shared/attention/self/key/bias:0",
        "tf_albert_model/albert/encoder/albert_layer_groups_._0/albert_layers_._0/attention/value/kernel:0":
            "bert/encoder/layer_shared/attention/self/value/kernel:0",
        "tf_albert_model/albert/encoder/albert_layer_groups_._0/albert_layers_._0/attention/value/bias:0":
            "bert/encoder/layer_shared/attention/self/value/bias:0",
        "tf_albert_model/albert/encoder/albert_layer_groups_._0/albert_layers_._0/attention/dense/kernel:0":
            "bert/encoder/layer_shared/attention/output/dense/kernel:0",
        "tf_albert_model/albert/encoder/albert_layer_groups_._0/albert_layers_._0/attention/dense/bias:0":
            "bert/encoder/layer_shared/attention/output/dense/bias:0",
        "tf_albert_model/albert/encoder/albert_layer_groups_._0/albert_layers_._0/attention/LayerNorm/gamma:0":
            "bert/encoder/layer_shared/attention/output/LayerNorm/gamma:0",
        "tf_albert_model/albert/encoder/albert_layer_groups_._0/albert_layers_._0/attention/LayerNorm/beta:0":
            "bert/encoder/layer_shared/attention/output/LayerNorm/beta:0",
        "tf_albert_model/albert/encoder/albert_layer_groups_._0/albert_layers_._0/ffn/kernel:0":
            "bert/encoder/layer_shared/intermediate/kernel:0",
        "tf_albert_model/albert/encoder/albert_layer_groups_._0/albert_layers_._0/ffn/bias:0":
            "bert/encoder/layer_shared/intermediate/bias:0",
        "tf_albert_model/albert/encoder/albert_layer_groups_._0/albert_layers_._0/ffn_output/kernel:0":
            "bert/encoder/layer_shared/output/dense/kernel:0",
        "tf_albert_model/albert/encoder/albert_layer_groups_._0/albert_layers_._0/ffn_output/bias:0":
            "bert/encoder/layer_shared/output/dense/bias:0",
        "tf_albert_model/albert/encoder/albert_layer_groups_._0/albert_layers_._0/full_layer_layer_norm/gamma:0":
            "bert/encoder/layer_shared/output/LayerNorm/gamma:0",
        "tf_albert_model/albert/encoder/albert_layer_groups_._0/albert_layers_._0/full_layer_layer_norm/beta:0":
            "bert/encoder/layer_shared/output/LayerNorm/beta:0",
    }

    for hf_name, h5_name in mapping.items():
        if hf_name not in hf_weights:
            raise KeyError(f"HF weight not found: {hf_name}")
        if h5_name not in h5_bert:
            raise KeyError(f"H5 weight not found: {h5_name}")
        val = h5_bert[h5_name][()]
        if hf_weights[hf_name].shape != val.shape:
            raise ValueError(f"Shape mismatch for {hf_name}: {hf_weights[hf_name].shape} vs {val.shape}")
        hf_weights[hf_name].assign(val)


def build_rep_dataset(tokenizer, max_seq_length: int, rep_texts):
    inputs = []
    for text in rep_texts:
        text = text.strip()
        if text:
            inputs.append(text)

    def encode(text: str):
        words = [c for c in text if not c.isspace()]
        tokens = []
        for w in words:
            sub = tokenizer.tokenize(w)
            if not sub:
                sub = [tokenizer.unk_token]
            tokens.extend(sub)

        limit = max_seq_length - 2
        if len(tokens) > limit:
            tokens = tokens[:limit]

        tokens = [tokenizer.cls_token] + tokens + [tokenizer.sep_token]
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_ids)
        token_type_ids = [0] * len(input_ids)

        pad_len = max_seq_length - len(input_ids)
        if pad_len:
            input_ids += [tokenizer.pad_token_id] * pad_len
            attention_mask += [0] * pad_len
            token_type_ids += [0] * pad_len

        return (
            np.array([input_ids], dtype=np.int32),
            np.array([attention_mask], dtype=np.int32),
            np.array([token_type_ids], dtype=np.int32),
        )

    def gen():
        for text in inputs:
            yield list(encode(text))

    return gen


def convert_tflite(saved_model_dir: Path, tflite_path: Path, quant: str, full_int8: bool,
                   rep_texts, allow_select_tf_ops: bool, lower_tensor_list_ops):
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
            # Keep int32 token id inputs; forcing int8 inputs can saturate ids and hurt accuracy.
    elif quant != "none":
        raise ValueError(f"Unsupported quant mode: {quant}")

    tflite_model = converter.convert()
    tflite_path.parent.mkdir(parents=True, exist_ok=True)
    with tflite_path.open("wb") as f:
        f.write(tflite_model)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, help="NER model directory with model.h5/vocabs.json")
    parser.add_argument("--base-model", required=True, help="HF ALBERT directory for config/tokenizer")
    parser.add_argument("--export-dir", required=True, help="Output directory for SavedModel and labels")
    parser.add_argument("--tflite-path", required=True, help="Output .tflite path")
    parser.add_argument("--quant", default="float16", choices=["none", "float16", "int8"], help="Quantization mode")
    parser.add_argument("--full-int8", action="store_true", help="Use full int8 (requires representative data).")
    parser.add_argument("--rep-data", default=None, help="Path to representative text file, one sentence per line.")
    parser.add_argument("--rep-text", default=None, help="Inline representative text, can be used multiple times.",
                        action="append")
    parser.add_argument("--allow-select-tf-ops", action="store_true",
                        help="Enable SELECT_TF_OPS when TFLite builtin ops are insufficient.")
    parser.add_argument("--lower-tensor-list-ops", action="store_true",
                        help="Force lower TensorList ops (may fail for some transformer models).")
    parser.add_argument("--no-lower-tensor-list-ops", action="store_true",
                        help="Disable lowering TensorList ops (often needed with SELECT_TF_OPS).")
    parser.add_argument("--max-seq-length", default=128, type=int, help="Max sequence length")
    return parser.parse_args()


def main():
    args = parse_args()
    model_dir = Path(args.model_dir).resolve()
    export_dir = Path(args.export_dir).resolve()
    tflite_path = Path(args.tflite_path).resolve()
    export_dir.mkdir(parents=True, exist_ok=True)

    labels = load_labels(model_dir)
    save_labels(labels, export_dir)
    save_meta(args.max_seq_length, len(labels), export_dir)

    h5_path = model_dir / "model.h5"
    with h5py.File(h5_path, "r") as h5f:
        h5_bert = h5f["bert"]
        h5_dense = h5f["dense"]

        base = TFAutoModel.from_pretrained(str(Path(args.base_model).resolve()))
        input_ids = tf.keras.layers.Input(shape=(args.max_seq_length,), dtype="int32", name="input_ids")
        mask_ids = tf.keras.layers.Input(shape=(args.max_seq_length,), dtype="int32", name="mask_ids")
        token_type_ids = tf.keras.layers.Input(shape=(args.max_seq_length,), dtype="int32", name="token_type_ids")
        sequence_output = base(input_ids, attention_mask=mask_ids, token_type_ids=token_type_ids).last_hidden_state
        logits = tf.keras.layers.Dense(len(labels), name="dense")(sequence_output)
        model = tf.keras.Model(inputs=[input_ids, mask_ids, token_type_ids], outputs=logits)

        assign_weights(base, h5_bert)

        dense_kernel = h5_dense["dense/kernel:0"][()]
        dense_bias = h5_dense["dense/bias:0"][()]
        model.get_layer("dense").set_weights([dense_kernel, dense_bias])

    saved_model_dir = export_dir / "1"
    tf.saved_model.save(model, str(saved_model_dir))

    rep_texts = None
    if args.rep_data:
        lines = Path(args.rep_data).read_text(encoding="utf-8").splitlines()
        rep_texts = build_rep_dataset(
            AutoTokenizer.from_pretrained(str(Path(args.base_model).resolve()), use_fast=False),
            args.max_seq_length,
            lines,
        )
    elif args.rep_text:
        rep_texts = build_rep_dataset(
            AutoTokenizer.from_pretrained(str(Path(args.base_model).resolve()), use_fast=False),
            args.max_seq_length,
            args.rep_text,
        )

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
