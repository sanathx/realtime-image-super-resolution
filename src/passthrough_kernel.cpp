#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>

using namespace sycl;

int main() {
    // 1. Initialize the iGPU Queue via Level Zero
    queue q(gpu_selector_v);
    std::cout << "Running on: " << q.get_device().get_info<info::device::name>() << "\n";

    // 2. Simulate a 480p Video Frame Frame Buffer (Width: 854, Height: 480, 3 Channels BGR)
    const int width = 854;
    const int height = 480;
    const int channels = 3;
    const size_t num_pixels = width * height * channels;
    const size_t buffer_size = num_pixels * sizeof(uint8_t);

    std::cout << "Simulating 480p Frame Buffer Size: " << (buffer_size / 1024.0) << " KB\n";

    // 3. Allocate Unified Shared Memory (USM) on the Silicon
    // This pointer is directly accessible by both CPU and iGPU Vector Lanes
    uint8_t* frame_buffer = malloc_shared<uint8_t>(num_pixels, q);

    // 4. Host (CPU) Initialization: Fill the frame with a dummy gradient pattern
    for (size_t i = 0; i < num_pixels; i++) {
        frame_buffer[i] = static_cast<uint8_t>(i % 256);
    }

    std::cout << "Original Pixel[0]: " << static_cast<int>(frame_buffer[0]) << "\n";

    // 5. Compute Offload: Execute the Parallel Kernel on the iGPU
    // We submit a command group to the hardware queue
    q.submit([&](handler& h) {
        
        // This lambda function executes across the entire iGPU fabric in parallel
        h.parallel_for(range<1>{num_pixels}, [=](id<1> idx) {
            // Each execution unit thread handles one pixel manipulation
            // For this test, we perform a simple bitwise NOT operation (Invert Color)
            frame_buffer[idx] = 255 - frame_buffer[idx];
        });
    }).wait(); // Synchronize: Wait for the iGPU hardware to finish processing

    // 6. Verification: Read the data back on the Host (CPU)
    std::cout << "Processed Pixel[0] (Should be 255): " << static_cast<int>(frame_buffer[0]) << "\n";

    // Clean up allocation
    free(frame_buffer, q);
    std::cout << "Pipeline clean. Memory safely released.\n";

    return 0;
}