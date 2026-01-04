package com.hanlp.nerdemo;

import android.os.Bundle;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;

import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class MainActivity extends AppCompatActivity {
    private NerEngine engine;
    private final ExecutorService executor = Executors.newSingleThreadExecutor();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        EditText inputText = findViewById(R.id.input_text);
        Button runButton = findViewById(R.id.run_button);
        TextView resultText = findViewById(R.id.result_text);

        inputText.setText("上海华安工业（集团）公司董事长谭旭光和秘书张晚霞来到美国纽约现代艺术博物馆参观。");

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
                engine = NerEngine.fromAssets(getAssets());
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

    @Override
    protected void onDestroy() {
        super.onDestroy();
        executor.shutdownNow();
        if (engine != null) {
            engine.close();
        }
    }
}
