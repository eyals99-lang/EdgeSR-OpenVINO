#include <print>
#include <vector>
#include <string>
#include <exception>
#include <cmath>
#include <chrono>
#include <openvino/openvino.hpp>
#include <opencv2/opencv.hpp>

int main() {
    std::print("Starting Video Super-Resolution Pipeline with Benchmarks...\n");

    try {
        ov::Core core;
        auto model = core.read_model("super-resolution-10.onnx");
        model->reshape({ov::Dimension(-1), ov::Dimension(1), ov::Dimension(224), ov::Dimension(224)});
        auto compiled_model = core.compile_model(model, "CPU");
        auto infer_request = compiled_model.create_infer_request();

        cv::VideoCapture cap("input.mp4");
        if (!cap.isOpened()) {
            throw std::runtime_error("Cannot open input.mp4. Please provide a valid video file.");
        }

        int orig_w = cap.get(cv::CAP_PROP_FRAME_WIDTH);
        int orig_h = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
        double fps = cap.get(cv::CAP_PROP_FPS);
        int total_frames = cap.get(cv::CAP_PROP_FRAME_COUNT);

        int target_w = orig_w * 3;
        int target_h = orig_h * 3;

        cv::VideoWriter writer("output_enhanced.mp4", cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, cv::Size(target_w, target_h));

        std::print("Video Loaded: {}x{} @ {} FPS. Output will be {}x{}\n", orig_w, orig_h, fps, target_w, target_h);

        const int TILE_IN = 224;
        const int TILE_OUT = 672;

        int rows = std::ceil((float)orig_h / TILE_IN);
        int cols = std::ceil((float)orig_w / TILE_IN);
        int num_tiles = rows * cols;

        std::print("Tiling grid: {} rows x {} cols (Total {} tiles per frame)\n\n", rows, cols, num_tiles);

        cv::Mat frame;
        int frame_idx = 0;
        double total_infer_time_ms = 0.0;

        // טיימר כולל לכל תהליך העיבוד
        auto start_total = std::chrono::high_resolution_clock::now();

        while (cap.read(frame)) {
            frame_idx++;
            std::print("Processing Frame {}/{}\r", frame_idx, total_frames);

            cv::Mat ycrcb;
            cv::cvtColor(frame, ycrcb, cv::COLOR_BGR2YCrCb);
            std::vector<cv::Mat> channels(3);
            cv::split(ycrcb, channels);
            cv::Mat y_channel = channels[0];

            int pad_bottom = (rows * TILE_IN) - orig_h;
            int pad_right = (cols * TILE_IN) - orig_w;
            cv::Mat y_padded;
            cv::copyMakeBorder(y_channel, y_padded, 0, pad_bottom, 0, pad_right, cv::BORDER_CONSTANT, cv::Scalar(0));

            std::vector<float> input_batch(num_tiles * TILE_IN * TILE_IN);
            for (int r = 0; r < rows; ++r) {
                for (int c = 0; c < cols; ++c) {
                    cv::Rect roi(c * TILE_IN, r * TILE_IN, TILE_IN, TILE_IN);
                    cv::Mat tile = y_padded(roi);
                    int tile_idx = r * cols + c;
                    int offset = tile_idx * TILE_IN * TILE_IN;
                    for (int i = 0; i < TILE_IN * TILE_IN; ++i) {
                        input_batch[offset + i] = tile.data[i] / 255.0f;
                    }
                }
            }

            ov::Tensor input_tensor(ov::element::f32, ov::Shape{(size_t)num_tiles, 1, TILE_IN, TILE_IN}, input_batch.data());
            infer_request.set_input_tensor(input_tensor);

            // טיימר פנימי נטו עבור ה-AI (Inference)
            auto start_infer = std::chrono::high_resolution_clock::now();
            infer_request.infer();
            auto end_infer = std::chrono::high_resolution_clock::now();
            total_infer_time_ms += std::chrono::duration<double, std::milli>(end_infer - start_infer).count();

            ov::Tensor output_tensor = infer_request.get_output_tensor();
            const float* out_data = output_tensor.data<const float>();

            cv::Mat y_out_padded(rows * TILE_OUT, cols * TILE_OUT, CV_8UC1);
            for (int r = 0; r < rows; ++r) {
                for (int c = 0; c < cols; ++c) {
                    cv::Rect roi(c * TILE_OUT, r * TILE_OUT, TILE_OUT, TILE_OUT);
                    cv::Mat tile_out = y_out_padded(roi);
                    int tile_idx = r * cols + c;
                    int offset = tile_idx * TILE_OUT * TILE_OUT;
                    for (int i = 0; i < TILE_OUT * TILE_OUT; ++i) {
                        float val = std::clamp(out_data[offset + i], 0.0f, 1.0f);
                        tile_out.data[i] = static_cast<uint8_t>(val * 255.0f);
                    }
                }
            }

            cv::Rect valid_roi(0, 0, target_w, target_h);
            cv::Mat y_final = y_out_padded(valid_roi);

            cv::Mat cr_final, cb_final;
            cv::resize(channels[1], cr_final, cv::Size(target_w, target_h), 0, 0, cv::INTER_CUBIC);
            cv::resize(channels[2], cb_final, cv::Size(target_w, target_h), 0, 0, cv::INTER_CUBIC);

            std::vector<cv::Mat> final_channels = {y_final, cr_final, cb_final};
            cv::Mat final_ycrcb, final_bgr;
            cv::merge(final_channels, final_ycrcb);
            cv::cvtColor(final_ycrcb, final_bgr, cv::COLOR_YCrCb2BGR);

            writer.write(final_bgr);
        }

        auto end_total = std::chrono::high_resolution_clock::now();
        double total_time_sec = std::chrono::duration<double>(end_total - start_total).count();

        cap.release();
        writer.release();

        double avg_fps = total_frames / total_time_sec;
        double avg_infer_ms = total_infer_time_ms / total_frames;

        std::print("\n\n=== Performance Benchmarks ===\n");
        std::print("Total Frames Processed : {}\n", total_frames);
        std::print("Total Processing Time  : {:.2f} seconds\n", total_time_sec);
        std::print("Avg Inference Time/Frame: {:.2f} ms\n", avg_infer_ms);
        std::print("End-to-End Pipeline FPS: {:.2f} FPS\n", avg_fps);
        std::print("==============================\n");

    } catch (const std::exception& e) {
        std::print("\nError: {}\n", e.what());
        return -1;
    }

    return 0;
}