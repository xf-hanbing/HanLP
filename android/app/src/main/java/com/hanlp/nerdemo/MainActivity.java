package com.hanlp.nerdemo;

import android.os.Bundle;
import android.content.res.AssetManager;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.EditText;
import android.widget.Spinner;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class MainActivity extends AppCompatActivity {
    private static final String[] MODEL_FILES = {
            "ner_float32.tflite",
            "ner_float16.tflite",
            "ner_int8.tflite",
            "ner_full_int8.tflite"
    };

    private NerEngine engine;
    private String selectedModelFile;
    private final ExecutorService executor = Executors.newSingleThreadExecutor();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        EditText inputText = findViewById(R.id.input_text);
        Button runButton = findViewById(R.id.run_button);
        TextView resultText = findViewById(R.id.result_text);
        Spinner precisionSpinner = findViewById(R.id.precision_spinner);

        inputText.setText("上海华安工业（集团）公司董事长谭旭光和秘书张晚霞来到美国纽约现代艺术博物馆参观。");

        setupPrecisionSpinner(precisionSpinner);

        runButton.setOnClickListener(v -> {
            String text = inputText.getText().toString().trim();
            if (text.isEmpty()) {
                resultText.setText("请输入文本后再试");
                return;
            }
            resultText.setText("运行中...");
            executor.execute(() -> runInference(text, resultText));
        });
    }

    private void runInference(String text, TextView resultText) {
        try {
            if (engine == null) {
                engine = NerEngine.fromAssets(getAssets(), selectedModelFile);
            }
            List<NerEngine.Span> spans = engine.predict(text);
            StringBuilder sb = new StringBuilder();
            if (spans.isEmpty()) {
                sb.append("没有识别到实体");
            } else {
                for (NerEngine.Span span : spans) {
                    sb.append(span.text)
                            .append("\t")
                            .append(span.type)
                            .append("\t")
                            .append(span.start)
                            .append("\t")
                            .append(span.end)
                            .append("\n");
                }
            }
            runOnUiThread(() -> resultText.setText(sb.toString()));
        } catch (Exception e) {
            runOnUiThread(() -> resultText.setText("运行失败: " + e.getMessage()));
        }
    }

    private void setupPrecisionSpinner(Spinner spinner) {
        AssetManager assets = getAssets();
        String[] labels = getResources().getStringArray(R.array.precision_options);
        List<String> availableLabels = new ArrayList<>();
        List<String> availableModels = new ArrayList<>();
        try {
            String[] assetList = assets.list("");
            Set<String> assetSet = assetList == null ? new HashSet<>() : new HashSet<>(Arrays.asList(assetList));
            for (int i = 0; i < MODEL_FILES.length; i++) {
                String modelFile = MODEL_FILES[i];
                if (assetSet.contains(modelFile)) {
                    availableModels.add(modelFile);
                    availableLabels.add(labels[i]);
                }
            }
        } catch (IOException ignored) {
            // Fall back to defaults when assets can't be listed.
        }

        if (availableModels.isEmpty()) {
            availableModels.add(MODEL_FILES[0]);
            availableLabels.add(labels[0]);
        }

        selectedModelFile = availableModels.get(0);

        ArrayAdapter<String> adapter = new ArrayAdapter<>(
                this, android.R.layout.simple_spinner_item, availableLabels);
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        spinner.setAdapter(adapter);
        spinner.setEnabled(availableModels.size() > 1);

        spinner.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> parent, android.view.View view, int position, long id) {
                selectedModelFile = availableModels.get(position);
                if (engine != null) {
                    engine.close();
                    engine = null;
                }
            }

            @Override
            public void onNothingSelected(AdapterView<?> parent) {
            }
        });
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        executor.shutdownNow();
        if (engine != null) {
            engine.close();
        }
    }
}
