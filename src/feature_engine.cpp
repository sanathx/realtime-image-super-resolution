#include <sycl/sycl.hpp>
#include <iostream>
#include <fstream>
#include <cmath>

using namespace sycl;

int main() {
    property_list props{property::queue::enable_profiling()};
    queue q(gpu_selector_v, props);
    std::cout << "Neural Engine Active: " << q.get_device().get_info<info::device::name>() << "\n";

    const int in_w = 854;   const int in_h = 480;
    const int out_w = 1920; const int out_h = 1080;
    const int channels = 3;

    const size_t in_size = in_w * in_h * channels;
    const size_t out_size = out_w * out_h * channels;

    uint8_t* input_frame = malloc_shared<uint8_t>(in_size, q);
    uint8_t* output_frame = malloc_shared<uint8_t>(out_size, q);

    std::ifstream infile("C:/Users/sandb/Desktop/iSR_Project/data/input_480p.raw", std::ios::binary);
    if (!infile) {
        std::cout << "Error: Cannot read input_480p.raw\n";
        return 1;
    }
    infile.read(reinterpret_cast<char*>(input_frame), in_size);
    infile.close();

    // Define a trained 5x5 FSRCNN-style Feature Extraction Weight Kernel (Gaussian Edge Detector)
    // In a full model, these numbers are loaded from the trained model file
    const float kernel_5x5[5][5] = {
        {-1, -1, -1, -1, -1},
        {-1,  1,  2,  1, -1},
        {-1,  2,  8,  2, -1},
        {-1,  1,  2,  1, -1},
        {-1, -1, -1, -1, -1}
    };
    const float bias = 0.0f;

    // Allocate memory for weights on the device via USM
    float* weights = malloc_shared<float>(25, q);
    for(int i=0; i<5; i++)
        for(int j=0; j<5; j++)
            weights[i*5 + j] = kernel_5x5[i][j];

    // Asynchronous Execution Grid over the 1080p Output Space
    auto execution_event = q.submit([&](handler& h) {
        h.parallel_for(range<2>{static_cast<size_t>(out_h), static_cast<size_t>(out_w)}, [=](id<2> item) {
            size_t out_y = item[0];
            size_t out_x = item[1];

            // 1. Coordinate Projection (1080p -> 480p map)
            float scale_x = static_cast<float>(in_w) / static_cast<float>(out_w);
            float scale_y = static_cast<float>(in_h) / static_cast<float>(out_h);
            int in_x = static_cast<int>(out_x * scale_x);
            int in_y = static_cast<int>(out_y * scale_y);

            // 2. 2D Convolution Neighborhood Loop (5x5 spatial window)
            // Each color channel is convolved independently for this layer visualization
            for (int c = 0; c < channels; c++) {
                float pixel_sum = 0.0f;

                for (int ky = -2; ky <= 2; ky++) {
                    for (int kx = -2; kx <= 2; kx++) {
                        // Apply zero-padding boundaries
                        int px = in_x + kx;
                        int py = in_y + ky;
                        
                        if (px >= 0 && px < in_w && py >= 0 && py < in_h) {
                            size_t in_idx = (py * in_w + px) * channels + c;
                            float weight = weights[(ky + 2) * 5 + (kx + 2)];
                            pixel_sum += input_frame[in_idx] * weight;
                        }
                    }
                }

                // 3. Apply Model Bias and Activation Function (ReLU Loophole: max(0, x))
                float activated_value = pixel_sum + bias;
                if (activated_value < 0.0f) activated_value = 0.0f;
                if (activated_value > 255.0f) activated_value = 255.0f; // Saturation clamp

                size_t out_idx = (out_y * out_w + out_x) * channels + c;
                output_frame[out_idx] = static_cast<uint8_t>(activated_value);
            }
        });
    });

    execution_event.wait();

    unsigned long long start_time = execution_event.get_profiling_info<info::event_profiling::command_start>();
    unsigned long long end_time = execution_event.get_profiling_info<info::event_profiling::command_end>();
    double kernel_time_ms = (end_time - start_time) / 1000000.0;

    std::ofstream outfile("C:/Users/sandb/Desktop/iSR_Project/data/output_1080p.raw", std::ios::binary);
    outfile.write(reinterpret_cast<char*>(output_frame), out_size);
    outfile.close();

    free(input_frame, q);
    free(output_frame, q);
    free(weights, q);

    std::cout << "========================================" << "\n";
    std::cout << "NEURAL CONVOLUTION KERNEL LATENCY: " << kernel_time_ms << " ms" << "\n";
    std::cout << "========================================" << "\n";

    return 0;
}