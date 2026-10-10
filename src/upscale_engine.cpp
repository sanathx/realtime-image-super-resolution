#include <sycl/sycl.hpp>
#include <iostream>
#include <fstream>

using namespace sycl;

int main() {
    // 1. Initialize Queue with Hardware Profiling enabled
    property_list props{property::queue::enable_profiling()};
    queue q(gpu_selector_v, props);
    std::cout << "SYCL Engine Active: " << q.get_device().get_info<info::device::name>() << "\n";

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

    // 2. Capture the asynchronous event returned by the hardware submission
    auto execution_event = q.submit([&](handler& h) {
        h.parallel_for(range<2>{static_cast<size_t>(out_h), static_cast<size_t>(out_w)}, [=](id<2> item) {
            size_t out_y = item[0];
            size_t out_x = item[1];

            float scale_x = static_cast<float>(in_w) / static_cast<float>(out_w);
            float scale_y = static_cast<float>(in_h) / static_cast<float>(out_h);

            size_t in_x = static_cast<size_t>(out_x * scale_x);
            size_t in_y = static_cast<size_t>(out_y * scale_y);

            if (in_x >= in_w) in_x = in_w - 1;
            if (in_y >= in_h) in_y = in_h - 1;

            size_t out_idx = (out_y * out_w + out_x) * channels;
            size_t in_idx = (in_y * in_w + in_x) * channels;

            output_frame[out_idx]     = input_frame[in_idx];
            output_frame[out_idx + 1] = input_frame[in_idx + 1];
            output_frame[out_idx + 2] = input_frame[in_idx + 2];
        });
    });

    // 3. Force synchronization block to ensure the kernel has fully cleared execution
    execution_event.wait();

    // 4. Query the hardware clocks (returns time in nanoseconds)
    unsigned long long start_time = execution_event.get_profiling_info<info::event_profiling::command_start>();
    unsigned long long end_time = execution_event.get_profiling_info<info::event_profiling::command_end>();
    
    // Convert nanoseconds to milliseconds: (end - start) / 1,000,000.0
    double kernel_time_ms = (end_time - start_time) / 1000000.0;

    std::ofstream outfile("C:/Users/sandb/Desktop/iSR_Project/data/output_1080p.raw", std::ios::binary);
    outfile.write(reinterpret_cast<char*>(output_frame), out_size);
    outfile.close();

    free(input_frame, q);
    free(output_frame, q);

    std::cout << "========================================" << "\n";
    std::cout << "PURE iGPU KERNEL COMPUTE LATENCY: " << kernel_time_ms << " ms" << "\n";
    std::cout << "========================================" << "\n";

    return 0;
}