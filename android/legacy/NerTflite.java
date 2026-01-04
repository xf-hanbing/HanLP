import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.Tensor;
import org.tensorflow.lite.Tensor.QuantizationParams;

public class NerTflite {
    private static final String CLS = "[CLS]";
    private static final String SEP = "[SEP]";
    private static final String PAD = "[PAD]";
    private static final String UNK = "[UNK]";

    public static void main(String[] args) throws Exception {
        Map<String, String> argMap = parseArgs(args);
        String modelPath = requiredArg(argMap, "--model");
        String vocabPath = requiredArg(argMap, "--vocab");
        String labelsPath = requiredArg(argMap, "--labels");
        String text = requiredArg(argMap, "--text");
        int maxLen = Integer.parseInt(argMap.getOrDefault("--max-len", "128"));

        Vocab vocab = Vocab.fromFile(vocabPath);
        List<String> labels = readLines(labelsPath);

        List<String> words = splitToChars(text);
        Encoding encoding = encode(words, vocab, maxLen);

        try (Interpreter interpreter = new Interpreter(new File(modelPath))) {
            Object[] inputs = buildInputs(interpreter, encoding);
            Output output = run(interpreter, inputs);
            List<String> predTags = outputToTags(output, labels, encoding.labelMask);
            List<Span> spans = getEntities(predTags);
            for (Span span : spans) {
                String entity = join(words, span.start, span.end);
                System.out.println(entity + "\t" + span.type + "\t" + span.start + "\t" + span.end);
            }
        }
    }

    private static Object[] buildInputs(Interpreter interpreter, Encoding encoding) {
        Tensor input0 = interpreter.getInputTensor(0);
        DataType dtype = input0.dataType();
        int[] shape = input0.shape();
        int seqLen = shape[1];
        if (seqLen != encoding.inputIds.length) {
            throw new IllegalArgumentException("Model seq_len " + seqLen + " != input " + encoding.inputIds.length);
        }
        int inputCount = interpreter.getInputTensorCount();
        Object[] inputs = new Object[inputCount];
        if (dtype == DataType.INT32) {
            for (int i = 0; i < inputCount; i++) {
                Tensor t = interpreter.getInputTensor(i);
                String name = t.name();
                if (name.contains("input_ids")) {
                    inputs[i] = new int[][]{encoding.inputIds};
                } else if (name.contains("mask_ids")) {
                    inputs[i] = new int[][]{encoding.attentionMask};
                } else if (name.contains("token_type_ids")) {
                    inputs[i] = new int[][]{encoding.tokenTypeIds};
                } else {
                    throw new IllegalArgumentException("Unknown input tensor: " + name);
                }
            }
        } else if (dtype == DataType.INT8) {
            QuantizationParams q = input0.quantizationParams();
            for (int i = 0; i < inputCount; i++) {
                Tensor t = interpreter.getInputTensor(i);
                String name = t.name();
                if (name.contains("input_ids")) {
                    inputs[i] = new byte[][]{quantize(encoding.inputIds, q)};
                } else if (name.contains("mask_ids")) {
                    inputs[i] = new byte[][]{quantize(encoding.attentionMask, q)};
                } else if (name.contains("token_type_ids")) {
                    inputs[i] = new byte[][]{quantize(encoding.tokenTypeIds, q)};
                } else {
                    throw new IllegalArgumentException("Unknown input tensor: " + name);
                }
            }
        } else {
            throw new IllegalArgumentException("Unsupported input dtype: " + dtype);
        }
        return inputs;
    }

    private static Output run(Interpreter interpreter, Object[] inputs) {
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
        } else if (outType == DataType.INT8) {
            byte[][][] logits = new byte[1][seqLen][numLabels];
            outputs.put(0, logits);
            interpreter.runForMultipleInputsOutputs(inputs, outputs);
            return Output.fromInt8(logits, outputTensor.quantizationParams());
        } else {
            throw new IllegalArgumentException("Unsupported output dtype: " + outType);
        }
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

    private static List<String> splitToChars(String text) {
        List<String> out = new ArrayList<>();
        text.codePoints().forEach(cp -> {
            if (!Character.isWhitespace(cp)) {
                out.add(new String(Character.toChars(cp)));
            }
        });
        return out;
    }

    private static String join(List<String> words, int start, int end) {
        StringBuilder sb = new StringBuilder();
        for (int i = start; i < end && i < words.size(); i++) {
            sb.append(words.get(i));
        }
        return sb.toString();
    }

    private static Map<String, String> parseArgs(String[] args) {
        Map<String, String> map = new HashMap<>();
        for (int i = 0; i < args.length; i++) {
            String key = args[i];
            if (!key.startsWith("--")) {
                continue;
            }
            if (i + 1 < args.length && !args[i + 1].startsWith("--")) {
                map.put(key, args[i + 1]);
                i++;
            } else {
                map.put(key, "true");
            }
        }
        return map;
    }

    private static String requiredArg(Map<String, String> map, String key) {
        if (!map.containsKey(key)) {
            throw new IllegalArgumentException("Missing arg: " + key);
        }
        return map.get(key);
    }

    private static List<String> readLines(String path) throws Exception {
        return Files.readAllLines(Paths.get(path), StandardCharsets.UTF_8);
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

    private static List<Span> getEntities(List<String> tags) {
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
                spans.add(new Span(prevType, begin, i));
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
        boolean chunkEnd = false;
        if ("E".equals(prevTag) || "S".equals(prevTag)) {
            chunkEnd = true;
        }
        if ("B".equals(prevTag) && ("B".equals(tag) || "S".equals(tag) || "O".equals(tag))) {
            chunkEnd = true;
        }
        if (("I".equals(prevTag) || "M".equals(prevTag)) && ("B".equals(tag) || "S".equals(tag) || "O".equals(tag))) {
            chunkEnd = true;
        }
        if (!"O".equals(prevTag) && !".".equals(prevTag) && !prevType.equals(type)) {
            chunkEnd = true;
        }
        return chunkEnd;
    }

    private static boolean startOfChunk(String prevTag, String tag, String prevType, String type) {
        boolean chunkStart = false;
        if ("B".equals(tag) || "S".equals(tag)) {
            chunkStart = true;
        }
        if ("E".equals(prevTag) && ("E".equals(tag) || "I".equals(tag) || "M".equals(tag))) {
            chunkStart = true;
        }
        if ("S".equals(prevTag) && ("E".equals(tag) || "I".equals(tag) || "M".equals(tag))) {
            chunkStart = true;
        }
        if ("O".equals(prevTag) && ("E".equals(tag) || "I".equals(tag) || "M".equals(tag))) {
            chunkStart = true;
        }
        if (!"O".equals(tag) && !".".equals(tag) && !prevType.equals(type)) {
            chunkStart = true;
        }
        return chunkStart;
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

    private static class Span {
        final String type;
        final int start;
        final int end;

        Span(String type, int start, int end) {
            this.type = type;
            this.start = start;
            this.end = end;
        }
    }

    private static class Vocab {
        private final Map<String, Integer> tokenToId;

        private Vocab(Map<String, Integer> tokenToId) {
            this.tokenToId = tokenToId;
        }

        static Vocab fromFile(String path) throws Exception {
            Map<String, Integer> map = new HashMap<>();
            try (BufferedReader reader = new BufferedReader(new FileReader(path))) {
                String line;
                int idx = 0;
                while ((line = reader.readLine()) != null) {
                    map.put(line.trim(), idx);
                    idx++;
                }
            }
            return new Vocab(map);
        }

        boolean contains(String token) {
            return tokenToId.containsKey(token);
        }

        int id(String token) {
            Integer v = tokenToId.get(token);
            if (v != null) {
                return v;
            }
            return tokenToId.getOrDefault(UNK, 0);
        }
    }
}
