#include <sycl/sycl.hpp>
#include <iostream>
#include <fstream>

using namespace sycl;

int main() {
    property_list props{property::queue::enable_profiling()};
    queue q(gpu_selector_v, props);
    
    auto device = q.get_device();
    std::cout << "Decoupled Engine Active on: " << device.get_info<info::device::name>() << "\n";

    // Structural Dimensions
    const int in_w = 854;   const int in_h = 480;
    const int out_w = 1920; const int out_h = 1080;
    const int channels = 3;

    const size_t in_size = in_w * in_h * channels;
    const size_t out_size = out_w * out_h * channels;

    // Allocation of Silicon Memory Spaces (Input, Intermediate Feature Map, Output)
    uint8_t* input_frame = malloc_shared<uint8_t>(in_size, q);
    uint8_t* feature_map = malloc_shared<uint8_t>(in_size, q); // Intermediate 480p buffer
    uint8_t* output_frame = malloc_shared<uint8_t>(out_size, q);

    std::ifstream infile("C:/Users/sandb/Desktop/iSR_Project/data/input_480p.raw", std::ios::binary);
    if (!infile) {
        std::cerr << "CRITICAL: Data buffer ingestion failure.\n";
        return 1;
    }
    infile.read(reinterpret_cast<char*>(input_frame), in_size);
    infile.close();

    // 5x5 Feature Extraction Weight Array
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

    // Tuning Kernel 1 Execution Grid: Fits the 480p boundaries perfectly
    // 854 columns rounded up to next multiple of 32 = 864
    range<2> k1_local{8, 32}; 
    range<2> k1_global{480, 864}; 

    // ========================================================================
    // STAGE 1: LOW-RESOLUTION CONVOLUTION WITH SUBGROUP REGISTER SHUFFLING
    // ========================================================================
    auto event_k1 = q.submit([&](handler& h) {
        h.parallel_for(nd_range<2>{k1_global, k1_local}, [=](nd_item<2> item) {
            auto sg = item.get_sub_group();
            size_t y = item.get_global_id(0);
            size_t x = item.get_global_id(1);
            size_t lane_id = sg.get_local_id()[0];

            if (y >= in_h || x >= in_w) return;

            for (int c = 0; c < channels; c++) {
                float pixel_sum = 0.0f;

                for (int ky = -2; ky <= 2; ky++) {
                    int py = static_cast<int>(y) + ky;
                    if (py < 0) py = 0;
                    if (py >= in_h) py = in_h - 1;

                    size_t center_idx = (py * in_w + x) * channels + c;
                    float private_register_pixel = static_cast<float>(input_frame[center_idx]);

                    sycl::group_barrier(sg);

                    for (int kx = -2; kx <= 2; kx++) {
                        float neighbor_pixel;
                        int target_lane = static_cast<int>(lane_id) + kx;

                        if (target_lane >= 0 && target_lane < 32 && (x + kx) < in_w) {
                            // FAST PATH: Direct register-to-register communication
                            neighbor_pixel = sycl::select_from_group(sg, private_register_pixel, target_lane);
                        } else {
                            // EDGE PATH: Global memory fallback clamping
                            int px = static_cast<int>(x) + kx;
                            if (px < 0) px = 0;
                            if (px >= in_w) px = in_w - 1;
                            size_t boundary_idx = (py * in_w + px) * channels + c;
                            neighbor_pixel = static_cast<float>(input_frame[boundary_idx]);
                        }

                        pixel_sum += neighbor_pixel * weights[(ky + 2) * 5 + (kx + 2)];
                    }
                }

                if (pixel_sum < 0.0f) pixel_sum = 0.0f;
                if (pixel_sum > 255.0f) pixel_sum = 255.0f;

                size_t out_idx = (y * in_w + x) * channels + c;
                feature_map[out_idx] = static_cast<uint8_t>(pixel_sum);
            }
        });
    });

    // ========================================================================
    // STAGE 2: SPATIAL GRID PROJECTION (480p Feature Map -> 1080p Spatial Frame)
    // ========================================================================
    // We pass event_k1 as a dependency rule to ensure sequential execution execution
    auto event_k2 = q.submit([&](handler& h) {
        h.depends_on(event_k1);
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

            output_frame[out_idx]     = feature_map[in_idx];
            output_frame[out_idx + 1] = feature_map[in_idx + 1];
            output_frame[out_idx + 2] = feature_map[in_idx + 2];
        });
    });

    event_k2.wait(); // Block host until the entire pipeline clears the hardware

    // Profiling Extractions
    double k1_ms = (event_k1.get_profiling_info<info::event_profiling::command_end>() - 
                    event_k1.get_profiling_info<info::event_profiling::command_start>()) / 1000000.0;
                    
    double k2_ms = (event_k2.get_profiling_info<info::event_profiling::command_end>() - 
                    event_k2.get_profiling_info<info::event_profiling::command_start>()) / 1000000.0;

    std::ofstream outfile("C:/Users/sandb/Desktop/iSR_Project/data/output_1080p.raw", std::ios::binary);
    outfile.write(reinterpret_cast<char*>(output_frame), out_size);
    outfile.close();

    free(input_frame, q);
    free(feature_map, q);
    free(output_frame, q);
    free(weights, q);

    std::cout << "\n============================================\n";
    std::cout << "STAGE 1: 480p CONVOLUTION LATENCY : " << k1_ms << " ms\n";
    std::cout << "STAGE 2: 1080p SPATIAL PROJECTION : " << k2_ms << " ms\n";
    std::cout << "TOTAL COMBINED SILICON LATENCY    : " << (k1_ms + k2_ms) << " ms\n";
    std::cout << "============================================\n";

    return 0;
}