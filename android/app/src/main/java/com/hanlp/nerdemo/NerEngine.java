package com.hanlp.nerdemo;

import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.Tensor;
import org.tensorflow.lite.Tensor.QuantizationParams;

import java.io.BufferedReader;
import java.io.Closeable;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class NerEngine implements Closeable {
    private static final String DEFAULT_MODEL_FILE = "ner_float16.tflite";
    private static final String VOCAB_FILE = "vocab.txt";
    private static final String LABELS_FILE = "labels.txt";
    private static final String CLS = "[CLS]";
    private static final String SEP = "[SEP]";
    private static final String PAD = "[PAD]";
    private static final String UNK = "[UNK]";

    private final Interpreter interpreter;
    private final Vocab vocab;
    private final List<String> labels;
    private final int maxLen;
    private final String modelFile;

    private NerEngine(Interpreter interpreter, Vocab vocab, List<String> labels, int maxLen, String modelFile) {
        this.interpreter = interpreter;
        this.vocab = vocab;
        this.labels = labels;
        this.maxLen = maxLen;
        this.modelFile = modelFile;
    }

    public static NerEngine fromAssets(AssetManager assets) throws IOException {
        return fromAssets(assets, DEFAULT_MODEL_FILE);
    }

    public static NerEngine fromAssets(AssetManager assets, String modelFile) throws IOException {
        String resolved = (modelFile == null || modelFile.trim().isEmpty())
                ? DEFAULT_MODEL_FILE
                : modelFile.trim();
        MappedByteBuffer modelBuffer = loadModelFile(assets, resolved);
        Interpreter.Options options = new Interpreter.Options();
        options.setNumThreads(4);
        Interpreter interpreter = new Interpreter(modelBuffer, options);
        int seqLen = interpreter.getInputTensor(0).shape()[1];
        Vocab vocab = Vocab.fromAsset(assets, VOCAB_FILE);
        List<String> labels = readLines(assets, LABELS_FILE);
        return new NerEngine(interpreter, vocab, labels, seqLen, resolved);
    }

    public List<Span> predict(String text) {
        List<String> chars = splitToChars(text);
        return predictWithChunks(chars);
    }

    @Override
    public void close() {
        interpreter.close();
    }

    public String debugSummary() {
        StringBuilder sb = new StringBuilder();
        sb.append("Model: ").append(modelFile).append("\n");
        int inputCount = interpreter.getInputTensorCount();
        for (int i = 0; i < inputCount; i++) {
            Tensor t = interpreter.getInputTensor(i);
            QuantizationParams q = t.quantizationParams();
            sb.append("Input ").append(i)
                    .append(": ").append(t.name())
                    .append(" ").append(t.dataType())
                    .append(" shape=").append(Arrays.toString(t.shape()))
                    .append(" scale=").append(q.getScale())
                    .append(" zero=").append(q.getZeroPoint())
                    .append("\n");
        }
        Tensor out = interpreter.getOutputTensor(0);
        QuantizationParams q = out.quantizationParams();
        sb.append("Output: ")
                .append(out.name())
                .append(" ").append(out.dataType())
                .append(" shape=").append(Arrays.toString(out.shape()))
                .append(" scale=").append(q.getScale())
                .append(" zero=").append(q.getZeroPoint());
        return sb.toString();
    }

    private Output run(Encoding encoding) {
        Object[] inputs = buildInputs(encoding);
        Tensor outputTensor = interpreter.getOutputTensor(0);
        int[] outShape = outputTensor.shape();
        int seqLen = outShape[1];
        int numLabels = outShape[2];
        DataType outType = outputTensor.dataType();
        Map<Integer, Object> outputs = new HashMap<>();
        if (outType == DataType.FLOAT32) {
            float[][][] logits = new float[1][seqLen][numLabels];
            outputs.put(0, logits);
            interpreter.runForMultipleInputsOutputs(inputs, outputs);
            return Output.fromFloat(logits);
        }
        if (outType == DataType.INT8) {
            byte[][][] logits = new byte[1][seqLen][numLabels];
            outputs.put(0, logits);
            interpreter.runForMultipleInputsOutputs(inputs, outputs);
            return Output.fromInt8(logits, outputTensor.quantizationParams());
        }
        throw new IllegalArgumentException("Unsupported output dtype: " + outType);
    }

    private Object[] buildInputs(Encoding encoding) {
        int inputCount = interpreter.getInputTensorCount();
        Object[] inputs = new Object[inputCount];
        for (int i = 0; i < inputCount; i++) {
            Tensor tensor = interpreter.getInputTensor(i);
            String name = tensor.name();
            DataType dtype = tensor.dataType();
            if (dtype == DataType.INT32) {
                if (name.contains("input_ids")) {
                    inputs[i] = new int[][]{encoding.inputIds};
                } else if (name.contains("mask_ids")) {
                    inputs[i] = new int[][]{encoding.attentionMask};
                } else if (name.contains("token_type_ids")) {
                    inputs[i] = new int[][]{encoding.tokenTypeIds};
                } else {
                    throw new IllegalArgumentException("Unknown input tensor: " + name);
                }
            } else if (dtype == DataType.INT8) {
                QuantizationParams q = tensor.quantizationParams();
                if (name.contains("input_ids")) {
                    inputs[i] = new byte[][]{quantize(encoding.inputIds, q)};
                } else if (name.contains("mask_ids")) {
                    inputs[i] = new byte[][]{quantize(encoding.attentionMask, q)};
                } else if (name.contains("token_type_ids")) {
                    inputs[i] = new byte[][]{quantize(encoding.tokenTypeIds, q)};
                } else {
                    throw new IllegalArgumentException("Unknown input tensor: " + name);
                }
            } else {
                throw new IllegalArgumentException("Unsupported input dtype: " + dtype);
            }
        }
        return inputs;
    }

    private static MappedByteBuffer loadModelFile(AssetManager assets, String fileName) throws IOException {
        AssetFileDescriptor fileDescriptor = assets.openFd(fileName);
        try (FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor())) {
            FileChannel fileChannel = inputStream.getChannel();
            long startOffset = fileDescriptor.getStartOffset();
            long declaredLength = fileDescriptor.getDeclaredLength();
            return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
        }
    }

    private static List<String> readLines(AssetManager assets, String fileName) throws IOException {
        List<String> lines = new ArrayList<>();
        try (BufferedReader reader = new BufferedReader(
                new InputStreamReader(assets.open(fileName), StandardCharsets.UTF_8))) {
            String line;
            while ((line = reader.readLine()) != null) {
                String trimmed = line.trim();
                if (!trimmed.isEmpty()) {
                    lines.add(trimmed);
                }
            }
        }
        return lines;
    }

    private static List<String> splitToChars(String text) {
        List<String> out = new ArrayList<>();
        text.codePoints().forEach(cp -> {
            if (!Character.isWhitespace(cp)) {
                out.add(new String(Character.toChars(cp)));
            }
        });
        return out;
    }

    private static Encoding encode(List<String> words, Vocab vocab, int maxLen) {
        List<String> tokens = new ArrayList<>();
        List<Integer> labelMask = new ArrayList<>();
        for (String word : words) {
            List<String> subTokens = wordpieceTokenize(word, vocab);
            for (int i = 0; i < subTokens.size(); i++) {
                tokens.add(subTokens.get(i));
                labelMask.add(i == 0 ? 1 : 0);
            }
        }
        int limit = maxLen - 2;
        if (tokens.size() > limit) {
            tokens = tokens.subList(0, limit);
            labelMask = labelMask.subList(0, limit);
        }

        List<String> finalTokens = new ArrayList<>();
        List<Integer> finalMask = new ArrayList<>();
        finalTokens.add(CLS);
        finalMask.add(0);
        finalTokens.addAll(tokens);
        finalMask.addAll(labelMask);
        finalTokens.add(SEP);
        finalMask.add(0);

        int[] inputIds = new int[maxLen];
        int[] attentionMask = new int[maxLen];
        int[] tokenTypeIds = new int[maxLen];
        int[] mask = new int[maxLen];
        for (int i = 0; i < finalTokens.size(); i++) {
            inputIds[i] = vocab.id(finalTokens.get(i));
            attentionMask[i] = 1;
            tokenTypeIds[i] = 0;
            mask[i] = finalMask.get(i);
        }
        int padId = vocab.id(PAD);
        for (int i = finalTokens.size(); i < maxLen; i++) {
            inputIds[i] = padId;
            attentionMask[i] = 0;
            tokenTypeIds[i] = 0;
            mask[i] = 0;
        }
        return new Encoding(inputIds, attentionMask, tokenTypeIds, mask);
    }

    private List<Span> predictWithChunks(List<String> chars) {
        int limit = maxLen - 2;
        List<Integer> tokenLens = new ArrayList<>(chars.size());
        int totalTokens = 0;
        for (String ch : chars) {
            int len = wordpieceTokenize(ch, vocab).size();
            tokenLens.add(len);
            totalTokens += len;
        }
        if (totalTokens <= limit) {
            return predictChunk(chars, 0);
        }
        List<Span> spans = new ArrayList<>();
        int start = 0;
        while (start < chars.size()) {
            int tokenSum = 0;
            int lastPunct = -1;
            int i = start;
            for (; i < chars.size(); i++) {
                int len = tokenLens.get(i);
                if (tokenSum + len > limit) {
                    break;
                }
                tokenSum += len;
                if (isSplitPunct(chars.get(i))) {
                    lastPunct = i + 1;
                }
            }
            int end;
            if (i >= chars.size()) {
                end = chars.size();
            } else if (lastPunct > start) {
                end = lastPunct;
            } else {
                end = i;
            }
            if (end <= start) {
                end = Math.min(start + 1, chars.size());
            }
            List<String> chunk = chars.subList(start, end);
            spans.addAll(predictChunk(chunk, start));
            start = end;
        }
        return spans;
    }

    private List<Span> predictChunk(List<String> chars, int offset) {
        Encoding encoding = encode(chars, vocab, maxLen);
        Output output = run(encoding);
        List<String> tags = outputToTags(output, labels, encoding.labelMask);
        List<Span> chunkSpans = decodeSpans(chars, tags);
        if (offset == 0) {
            return chunkSpans;
        }
        List<Span> shifted = new ArrayList<>(chunkSpans.size());
        for (Span span : chunkSpans) {
            shifted.add(new Span(span.type, span.text, span.start + offset, span.end + offset));
        }
        return shifted;
    }

    private static List<String> outputToTags(Output output, List<String> labels, int[] labelMask) {
        List<String> tags = new ArrayList<>();
        int seqLen = output.seqLen();
        for (int i = 0; i < seqLen; i++) {
            if (labelMask[i] != 1) {
                continue;
            }
            int labelId = output.argmax(i);
            tags.add(labels.get(labelId));
        }
        return tags;
    }

    private static List<Span> decodeSpans(List<String> chars, List<String> tags) {
        List<Span> spans = new ArrayList<>();
        String prevTag = "O";
        String prevType = "";
        int begin = 0;
        for (int i = 0; i <= tags.size(); i++) {
            String tag = (i == tags.size()) ? "O" : tags.get(i);
            String type = "";
            String tagPrefix = "O";
            if (!"O".equals(tag)) {
                tagPrefix = tag.substring(0, 1);
                type = tag.substring(2);
            }
            if (endOfChunk(prevTag, tagPrefix, prevType, type)) {
                spans.add(new Span(prevType, join(chars, begin, i), begin, i));
            }
            if (startOfChunk(prevTag, tagPrefix, prevType, type)) {
                begin = i;
            }
            prevTag = tagPrefix;
            prevType = type;
        }
        return spans;
    }

    private static boolean endOfChunk(String prevTag, String tag, String prevType, String type) {
        if ("E".equals(prevTag) || "S".equals(prevTag)) {
            return true;
        }
        if ("B".equals(prevTag) && ("B".equals(tag) || "S".equals(tag) || "O".equals(tag))) {
            return true;
        }
        if (("I".equals(prevTag) || "M".equals(prevTag))
                && ("B".equals(tag) || "S".equals(tag) || "O".equals(tag))) {
            return true;
        }
        return !"O".equals(prevTag) && !".".equals(prevTag) && !prevType.equals(type);
    }

    private static boolean startOfChunk(String prevTag, String tag, String prevType, String type) {
        if ("B".equals(tag) || "S".equals(tag)) {
            return true;
        }
        if ("E".equals(prevTag) && ("E".equals(tag) || "I".equals(tag) || "M".equals(tag))) {
            return true;
        }
        if ("S".equals(prevTag) && ("E".equals(tag) || "I".equals(tag) || "M".equals(tag))) {
            return true;
        }
        if ("O".equals(prevTag) && ("E".equals(tag) || "I".equals(tag) || "M".equals(tag))) {
            return true;
        }
        return !"O".equals(tag) && !".".equals(tag) && !prevType.equals(type);
    }

    private static String join(List<String> chars, int start, int end) {
        StringBuilder sb = new StringBuilder();
        for (int i = start; i < end && i < chars.size(); i++) {
            sb.append(chars.get(i));
        }
        return sb.toString();
    }

    private static boolean isSplitPunct(String token) {
        if (token == null || token.isEmpty()) {
            return false;
        }
        char c = token.charAt(0);
        return c == '，' || c == '。' || c == '！' || c == '？' || c == '；' || c == '：'
                || c == '、' || c == '…' || c == '—'
                || c == ',' || c == '.' || c == '!' || c == '?' || c == ';' || c == ':';
    }

    private static List<String> wordpieceTokenize(String token, Vocab vocab) {
        if (vocab.contains(token)) {
            List<String> out = new ArrayList<>();
            out.add(token);
            return out;
        }
        List<String> subTokens = new ArrayList<>();
        int len = token.length();
        int start = 0;
        while (start < len) {
            int end = len;
            String curSub = null;
            while (start < end) {
                String substr = token.substring(start, end);
                if (start > 0) {
                    substr = "##" + substr;
                }
                if (vocab.contains(substr)) {
                    curSub = substr;
                    break;
                }
                end -= 1;
            }
            if (curSub == null) {
                subTokens.clear();
                subTokens.add(UNK);
                return subTokens;
            }
            subTokens.add(curSub);
            start = end;
        }
        return subTokens;
    }

    private static byte[] quantize(int[] input, QuantizationParams q) {
        byte[] out = new byte[input.length];
        float scale = q.getScale();
        int zeroPoint = q.getZeroPoint();
        for (int i = 0; i < input.length; i++) {
            int v = Math.round(input[i] / scale) + zeroPoint;
            if (v > Byte.MAX_VALUE) {
                v = Byte.MAX_VALUE;
            } else if (v < Byte.MIN_VALUE) {
                v = Byte.MIN_VALUE;
            }
            out[i] = (byte) v;
        }
        return out;
    }

    private static class Encoding {
        final int[] inputIds;
        final int[] attentionMask;
        final int[] tokenTypeIds;
        final int[] labelMask;

        Encoding(int[] inputIds, int[] attentionMask, int[] tokenTypeIds, int[] labelMask) {
            this.inputIds = inputIds;
            this.attentionMask = attentionMask;
            this.tokenTypeIds = tokenTypeIds;
            this.labelMask = labelMask;
        }
    }

    private static class Output {
        final float[][][] floatLogits;
        final byte[][][] int8Logits;
        final QuantizationParams q;

        static Output fromFloat(float[][][] logits) {
            return new Output(logits, null, null);
        }

        static Output fromInt8(byte[][][] logits, QuantizationParams q) {
            return new Output(null, logits, q);
        }

        Output(float[][][] floatLogits, byte[][][] int8Logits, QuantizationParams q) {
            this.floatLogits = floatLogits;
            this.int8Logits = int8Logits;
            this.q = q;
        }

        int seqLen() {
            return floatLogits != null ? floatLogits[0].length : int8Logits[0].length;
        }

        int argmax(int pos) {
            if (floatLogits != null) {
                float[] row = floatLogits[0][pos];
                int best = 0;
                float bestVal = row[0];
                for (int i = 1; i < row.length; i++) {
                    if (row[i] > bestVal) {
                        bestVal = row[i];
                        best = i;
                    }
                }
                return best;
            }
            byte[] row = int8Logits[0][pos];
            int best = 0;
            float bestVal = dequantize(row[0]);
            for (int i = 1; i < row.length; i++) {
                float v = dequantize(row[i]);
                if (v > bestVal) {
                    bestVal = v;
                    best = i;
                }
            }
            return best;
        }

        private float dequantize(byte v) {
            if (q == null) {
                return v;
            }
            return (v - q.getZeroPoint()) * q.getScale();
        }
    }

    public static class Span {
        public final String type;
        public final String text;
        public final int start;
        public final int end;

        Span(String type, String text, int start, int end) {
            this.type = type;
            this.text = text;
            this.start = start;
            this.end = end;
        }
    }

    private static class Vocab {
        private final Map<String, Integer> tokenToId;

        private Vocab(Map<String, Integer> tokenToId) {
            this.tokenToId = tokenToId;
        }

        static Vocab fromAsset(AssetManager assets, String fileName) throws IOException {
            Map<String, Integer> map = new HashMap<>();
            try (BufferedReader reader = new BufferedReader(
                    new InputStreamReader(assets.open(fileName), StandardCharsets.UTF_8))) {
                String line;
                int idx = 0;
                while ((line = reader.readLine()) != null) {
                    String token = line.trim();
                    if (!token.isEmpty()) {
                        map.put(token, idx++);
                    }
                }
            }
            return new Vocab(map);
        }

        boolean contains(String token) {
            return tokenToId.containsKey(token);
        }

        int id(String token) {
            Integer id = tokenToId.get(token);
            if (id == null) {
                return tokenToId.getOrDefault(UNK, 0);
            }
            return id;
        }
    }
}
