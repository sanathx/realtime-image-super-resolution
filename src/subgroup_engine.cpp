#include <sycl/sycl.hpp>
#include <iostream>
#include <fstream>

using namespace sycl;

int main() {
    // 1. Initialize Level Zero Queue with Hardware Profiling Properties
    property_list props{property::queue::enable_profiling()};
    queue q(gpu_selector_v, props);
    
    auto device = q.get_device();
    std::cout << "Target Hardware: " << device.get_info<info::device::name>() << "\n";

    // 2. Define Rigorous Spatial Dimensions
    const int in_w = 854;   const int in_h = 480;
    const int out_w = 1920; const int out_h = 1080;
    const int channels = 3;

    const size_t in_size = in_w * in_h * channels;
    const size_t out_size = out_w * out_h * channels;

    // 3. Allocate Virtual Coherent Unified Shared Memory (USM)
    uint8_t* input_frame = malloc_shared<uint8_t>(in_size, q);
    uint8_t* output_frame = malloc_shared<uint8_t>(out_size, q);

    // 4. Ingest raw 480p frame buffer from disk
    std::ifstream infile("C:/Users/sandb/Desktop/iSR_Project/data/input_480p.raw", std::ios::binary);
    if (!infile) {
        std::cerr << "CRITICAL: Input data buffer read failure.\n";
        return 1;
    }
    infile.read(reinterpret_cast<char*>(input_frame), in_size);
    infile.close();

    // 5. Statically allocated 5x5 Feature Extraction Weights (Gaussian Edge Filter Proxy)
    const float kernel_5x5[5][5] = {
        {-1, -1, -1, -1, -1},
        {-1,  1,  2,  1, -1},
        {-1,  2,  8,  2, -1},
        {-1,  1,  2,  1, -1},
        {-1, -1, -1, -1, -1}
    };
    
    float* weights = malloc_shared<float>(25, q);
    for(int i=0; i<5; i++)
        for(int j=0; j<5; j++)
            weights[i*5 + j] = kernel_5x5[i][j];

    // 6. Hardware Tuning: Define Work-Group dimensions aligned to Subgroup boundaries
    // Local size X must equal 32 to lock one Subgroup per row slice precisely
    range<2> local_size{8, 32}; 
    range<2> global_size{1080, 1920}; // Perfect spatial multiples of local execution shapes

    // 7. Asynchronous Execution Submission Block
    auto execution_event = q.submit([&](handler& h) {
        h.parallel_for(nd_range<2>{global_size, local_size}, [=](nd_item<2> item) {
            // Extract core hardware primitives
            auto sg = item.get_sub_group();
            size_t out_y = item.get_global_id(0);
            size_t out_x = item.get_global_id(1);
            
            // Get local lane ID within current active hardware subgroup (0 to 31)
            size_t lane_id = sg.get_local_id()[0];

            // Boundary guard for spatial safety
            if (out_y >= out_h || out_x >= out_w) return;

            float scale_x = static_cast<float>(in_w) / static_cast<float>(out_w);
            float scale_y = static_cast<float>(in_h) / static_cast<float>(out_h);
            
            int in_y = static_cast<int>(out_y * scale_y);
            int in_x = static_cast<int>(out_x * scale_x);

            for (int c = 0; c < channels; c++) {
                float pixel_sum = 0.0f;

                // Loop over the vertical height of the 5x5 convolution stencil
                for (int ky = -2; ky <= 2; ky++) {
                    int py = in_y + ky;
                    if (py < 0) py = 0;
                    if (py >= in_h) py = in_h - 1;

                    // REGISTER SHUFFLE ENGINE OPTIMIZATION:
                    // Pre-cache primary center target pixel into an internal register
                    size_t center_idx = (py * in_w + in_x) * channels + c;
                    float private_register_pixel = static_cast<float>(input_frame[center_idx]);

                    // Standard SYCL 2020 Group Barrier Collective execution execution
                    sycl::group_barrier(sg);

                    // Loop over the horizontal width of the 5x5 stencil
                    for (int kx = -2; kx <= 2; kx++) {
                        float neighbor_pixel;
                        int target_lane = static_cast<int>(lane_id) + kx;

                        // Intra-Subgroup Register Communication Conditionals
                        if (target_lane >= 0 && target_lane < 32) {
                            // FAST PATH: Standard SYCL 2020 Register Shuffle Algorithm
                            neighbor_pixel = sycl::select_from_group(sg, private_register_pixel, target_lane);
                        } else {
                            // EDGE PATH: Out-of-bounds boundary fallback
                            int px = in_x + kx;
                            if (px < 0) px = 0;
                            if (px >= in_w) px = in_w - 1;
                            size_t boundary_idx = (py * in_w + px) * channels + c;
                            neighbor_pixel = static_cast<float>(input_frame[boundary_idx]);
                        }

                        // Accumulate filtered matrix products
                        float w = weights[(ky + 2) * 5 + (kx + 2)];
                        pixel_sum += neighbor_pixel * w;
                    }
                }

                // Mathematical Activation Layer (Non-linear ReLU function)
                if (pixel_sum < 0.0f) pixel_sum = 0.0f;
                if (pixel_sum > 255.0f) pixel_sum = 255.0f;

                size_t out_idx = (out_y * out_w + out_x) * channels + c;
                output_frame[out_idx] = static_cast<uint8_t>(pixel_sum);
            }
        });
    });

    // 8. Block host thread execution until hardware clears execution pipelines
    execution_event.wait();

    // 9. Extract exact execution time from internal hardware clock registers
    unsigned long long start_time = execution_event.get_profiling_info<info::event_profiling::command_start>();
    unsigned long long end_time = execution_event.get_profiling_info<info::event_profiling::command_end>();
    double kernel_time_ms = (end_time - start_time) / 1000000.0;

    // 10. Egest processed 1080p frame buffer back to disk
    std::ofstream outfile("C:/Users/sandb/Desktop/iSR_Project/data/output_1080p.raw", std::ios::binary);
    outfile.write(reinterpret_cast<char*>(output_frame), out_size);
    outfile.close();

    // 11. Clean up Silicon Allocation Pointers
    free(input_frame, q);
    free(output_frame, q);
    free(weights, q);

    std::cout << "\n========================================\n";
    std::cout << "SUBGROUP SHUFFLE KERNEL LATENCY: " << kernel_time_ms << " ms\n";
    std::cout << "========================================\n";

    return 0;
}