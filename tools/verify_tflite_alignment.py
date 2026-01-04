#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import sys
from pathlib import Path
from typing import List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import h5py
import numpy as np
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModel


def build_inputs(text: str, tokenizer, max_len: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    words = [c for c in text if not c.isspace()]
    tokens: List[str] = []
    label_mask: List[int] = []
    for w in words:
        sub = tokenizer.tokenize(w)
        if not sub:
            sub = [tokenizer.unk_token]
        for i, s in enumerate(sub):
            tokens.append(s)
            label_mask.append(1 if i == 0 else 0)

    limit = max_len - 2
    if len(tokens) > limit:
        tokens = tokens[:limit]
        label_mask = label_mask[:limit]

    tokens = [tokenizer.cls_token] + tokens + [tokenizer.sep_token]
    label_mask = [0] + label_mask + [0]

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    attention_mask = [1] * len(input_ids)
    token_type_ids = [0] * len(input_ids)

    pad_len = max_len - len(input_ids)
    input_ids += [tokenizer.pad_token_id] * pad_len
    attention_mask += [0] * pad_len
    token_type_ids += [0] * pad_len
    label_mask += [0] * pad_len

    return (np.array([input_ids], dtype=np.int32),
            np.array([attention_mask], dtype=np.int32),
            np.array([token_type_ids], dtype=np.int32),
            np.array(label_mask, dtype=np.int32))


def load_labels(labels_path: Path) -> List[str]:
    return [line.strip() for line in labels_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def argmax_tags(logits: np.ndarray, label_mask: np.ndarray) -> List[int]:
    pred = logits.argmax(-1)[0]
    return [int(pred[i]) for i in range(len(pred)) if label_mask[i] == 1]


def get_entities(tags: List[str]):
    spans = []
    prev_tag = "O"
    prev_type = ""
    begin = 0
    for i in range(len(tags) + 1):
        tag = "O" if i == len(tags) else tags[i]
        tag_prefix = "O"
        tag_type = ""
        if tag != "O":
            tag_prefix = tag[0]
            tag_type = tag[2:]
        if end_of_chunk(prev_tag, tag_prefix, prev_type, tag_type):
            spans.append((prev_type, begin, i))
        if start_of_chunk(prev_tag, tag_prefix, prev_type, tag_type):
            begin = i
        prev_tag = tag_prefix
        prev_type = tag_type
    return spans


def end_of_chunk(prev_tag, tag, prev_type, type_):
    if prev_tag in ("E", "S"):
        return True
    if prev_tag == "B" and tag in ("B", "S", "O"):
        return True
    if prev_tag in ("I", "M") and tag in ("B", "S", "O"):
        return True
    if prev_tag not in ("O", ".") and prev_type != type_:
        return True
    return False


def start_of_chunk(prev_tag, tag, prev_type, type_):
    if tag in ("B", "S"):
        return True
    if prev_tag == "E" and tag in ("E", "I", "M"):
        return True
    if prev_tag == "S" and tag in ("E", "I", "M"):
        return True
    if prev_tag == "O" and tag in ("E", "I", "M"):
        return True
    if tag not in ("O", ".") and prev_type != type_:
        return True
    return False


def decode_spans(text: str, tags: List[str]):
    chars = [c for c in text if not c.isspace()]
    spans = get_entities(tags)
    out = []
    for t, s, e in spans:
        out.append((t, "".join(chars[s:e]), s, e))
    return out


def build_keras_model(model_dir: Path, base_model_dir: Path, max_len: int, num_labels: int):
    with h5py.File(model_dir / "model.h5", "r") as h5f:
        h5_bert = h5f["bert"]
        h5_dense = h5f["dense"]

        base = TFAutoModel.from_pretrained(str(base_model_dir))
        input_ids = tf.keras.layers.Input(shape=(max_len,), dtype="int32", name="input_ids")
        mask_ids = tf.keras.layers.Input(shape=(max_len,), dtype="int32", name="mask_ids")
        token_type_ids = tf.keras.layers.Input(shape=(max_len,), dtype="int32", name="token_type_ids")
        sequence_output = base(input_ids, attention_mask=mask_ids, token_type_ids=token_type_ids).last_hidden_state
        logits = tf.keras.layers.Dense(num_labels, name="dense")(sequence_output)
        model = tf.keras.Model(inputs=[input_ids, mask_ids, token_type_ids], outputs=logits)

        # weight mapping is same as export script
        from tools.export_ner_tflite_legacy_albert import assign_weights
        assign_weights(base, h5_bert)

        dense_kernel = h5_dense["dense/kernel:0"][()]
        dense_bias = h5_dense["dense/bias:0"][()]
        model.get_layer("dense").set_weights([dense_kernel, dense_bias])
    return model


def run_tflite(tflite_path: Path, inputs):
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    name_to_arr = {
        "input_ids": inputs[0],
        "mask_ids": inputs[1],
        "token_type_ids": inputs[2],
    }
    for detail in input_details:
        name = detail["name"]
        matched = False
        for key, arr in name_to_arr.items():
            if key in name:
                interpreter.set_tensor(detail["index"], arr)
                matched = True
                break
        if not matched:
            raise RuntimeError(f"Unknown input name: {name}")
    interpreter.invoke()
    output_details = interpreter.get_output_details()
    return interpreter.get_tensor(output_details[0]["index"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--tflite", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--max-len", type=int, default=128)
    args = parser.parse_args()

    model_dir = Path(args.model_dir).resolve()
    base_model_dir = Path(args.base_model).resolve()
    labels = load_labels(Path(args.labels))
    tokenizer = AutoTokenizer.from_pretrained(str(base_model_dir))

    input_ids, mask_ids, token_type_ids, label_mask = build_inputs(args.text, tokenizer, args.max_len)

    keras_model = build_keras_model(model_dir, base_model_dir, args.max_len, len(labels))
    keras_logits = keras_model([input_ids, mask_ids, token_type_ids], training=False).numpy()
    keras_tag_ids = argmax_tags(keras_logits, label_mask)
    keras_tags = [labels[i] for i in keras_tag_ids]
    keras_spans = decode_spans(args.text, keras_tags)

    tflite_logits = run_tflite(Path(args.tflite), (input_ids, mask_ids, token_type_ids))
    tflite_tag_ids = argmax_tags(tflite_logits, label_mask)
    tflite_tags = [labels[i] for i in tflite_tag_ids]
    tflite_spans = decode_spans(args.text, tflite_tags)

    print("Keras tags == TFLite tags:", keras_tag_ids == tflite_tag_ids)
    print("Keras spans:", keras_spans)
    print("TFLite spans:", tflite_spans)


if __name__ == "__main__":
    main()
