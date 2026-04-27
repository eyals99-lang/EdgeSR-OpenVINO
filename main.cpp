#include <print>
#include <vector>
#include <string>
#include <exception>
#include <algorithm>
#include <chrono>
#include <openvino/openvino.hpp>
#include <opencv2/opencv.hpp>

int main() {
    std::print("Starting Super-Resolution Minimal PoC...\n\n");

    try {
        // ========================================================
        // 1. יצירת קלט הבדיקה (תמונה 224x224 בשחור לבן)
        // ========================================================
        const int input_size = 224;
        cv::Mat input_img = cv::Mat::zeros(input_size, input_size, CV_8UC1);

        // נצייר עיגול לבן עם טקסט שחור כדי ליצור קצוות חדים (Edges) שהמודל אמור לשפר
        cv::circle(input_img, cv::Point(112, 112), 80, cv::Scalar(255), -1);
        cv::putText(input_img, "AI", cv::Point(60, 140), cv::FONT_HERSHEY_SIMPLEX, 3.0, cv::Scalar(0), 5);

        cv::imwrite("1_input_224x224.png", input_img);
        std::print("[+] Created dummy input image: 1_input_224x224.png\n");

        // ========================================================
        // 2. הכנת טנזור הקלט (Normalization)
        // ========================================================
        std::vector<float> input_tensor_data(input_size * input_size);
        for (int i = 0; i < input_size * input_size; ++i) {
            // המרה ל-float ונרמול ל-0.0 - 1.0
            input_tensor_data[i] = input_img.data[i] / 255.0f;
        }

        // ========================================================
        // 3. אתחול מנוע OpenVINO וטעינת המודל
        // ========================================================
        std::print("[+] Initializing OpenVINO and loading model...\n");
        ov::Core core;
        auto model = core.read_model("super-resolution-10.onnx");

        // קימפול המודל למעבד ויצירת בקשת הרצה
        auto compiled_model = core.compile_model(model, "CPU");
        auto infer_request = compiled_model.create_infer_request();

        // עטיפת הזיכרון שלנו בטנזור של OpenVINO (Zero-copy)
        // Shape: [Batch=1, Channels=1, Height=224, Width=224]
        ov::Tensor input_tensor(ov::element::f32, ov::Shape{1, 1, 224, 224}, input_tensor_data.data());
        infer_request.set_input_tensor(input_tensor);

        // ========================================================
        // 4. הרצת ה-AI (Inference)
        // ========================================================
        std::print("[+] Running Inference...\n");
        auto start = std::chrono::high_resolution_clock::now();

        infer_request.infer(); // <--- כאן קורה כל הקסם המתמטי

        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        std::print("[+] Inference completed in {:.2f} ms.\n", time_ms);

        // ========================================================
        // 5. חילוץ התוצאה ושמירת התמונה המוגדלת
        // ========================================================
        ov::Tensor output_tensor = infer_request.get_output_tensor();
        ov::Shape out_shape = output_tensor.get_shape();

        size_t out_h = out_shape[2]; // אמור להיות 672
        size_t out_w = out_shape[3]; // אמור להיות 672
        std::print("[+] Output Tensor Shape: {}x{}\n", out_w, out_h);

        const float* out_data = output_tensor.data<const float>();
        cv::Mat output_img(out_h, out_w, CV_8UC1);

        for (size_t i = 0; i < out_h * out_w; ++i) {
            // יש להגן מפני חריגות (מודלים יכולים לפלוט מספרים כמו -0.01 או 1.05)
            float clamped_val = std::clamp(out_data[i], 0.0f, 1.0f);

            // המרה חזרה מ-float מנורמל ל-uint8 (0-255)
            output_img.data[i] = static_cast<uint8_t>(clamped_val * 255.0f);
        }

        cv::imwrite("2_output_672x672.png", output_img);
        std::print("[+] Saved enhanced output image: 2_output_672x672.png\n");

    } catch (const std::exception& e) {
        std::print("[-] Error: {}\n", e.what());
        return -1;
    }

    return 0;
}