#include <sycl/sycl.hpp>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <io.h>    // Required for Windows _setmode
#include <fcntl.h> // Required for _O_BINARY

using namespace sycl;

int main() {
    // Force standard I/O rings into strict binary mode to prevent Windows byte corruption
    _setmode(_fileno(stdin), _O_BINARY);
    _setmode(_fileno(stdout), _O_BINARY);

    property_list props{property::queue::enable_profiling()};
    queue q(gpu_selector_v, props);

    const int in_w = 854;   const int in_h = 480;
    const int out_w = 1920; const int out_h = 1080;

    const size_t in_size = in_w * in_h;
    const size_t out_size = out_w * out_h;

    // Allocate persistent memory contexts for frame data
    uint8_t* input_frame   = malloc_shared<uint8_t>(in_size, q);
    uint8_t* feature_stage1 = malloc_shared<uint8_t>(in_size, q);
    uint8_t* feature_shrink = malloc_shared<uint8_t>(in_size, q);
    uint8_t* feature_mapped = malloc_shared<uint8_t>(in_size, q);
    uint8_t* output_frame  = malloc_shared<uint8_t>(out_size, q);

    // Allocate hardware Unified Shared Memory (USM) for the model layers
    float* w_5x5 = malloc_shared<float>(25, q);
    float* w_3x3 = malloc_shared<float>(9, q);

    // ========================================================================
    // NATIVE BINARY TENSOR LOADER RUNTIME
    // ========================================================================
    std::string weights_path = "C:/Users/sandb/Desktop/iSR_Project/data/fsrcnn_weights.bin";
    std::ifstream w_file(weights_path, std::ios::binary);
    if (!w_file.is_open()) { // FIXED: Standard C++ STL file openness checker
        std::cerr << "CRITICAL MODEL ERROR: Could not open fsrcnn_weights.bin" << std::endl;
        return 1;
    }
    
    // Stream binary arrays straight into cache-coherent CPU-GPU memory addresses
    w_file.read(reinterpret_cast<char*>(w_5x5), 25 * sizeof(float));
    w_file.read(reinterpret_cast<char*>(w_3x3), 9 * sizeof(float));
    w_file.close();
    // ========================================================================

    range<2> local_size{8, 32}; 
    range<2> global_size{480, 864}; 

    while (std::cin.read(reinterpret_cast<char*>(input_frame), in_size)) {
        
        // STAGE 1: Feature Extraction Matrix Convolution
        auto ev_extract = q.submit([&](handler& h) {
            h.parallel_for(nd_range<2>{global_size, local_size}, [=](nd_item<2> item) {
                auto sg = item.get_sub_group();
                size_t y = item.get_global_id(0); size_t x = item.get_global_id(1);
                size_t lane_id = sg.get_local_id()[0];
                int native_sg_size = sg.get_local_range()[0];
                
                if (y >= in_h || x >= in_w) return;

                float sum = 0.0f;
                for (int ky = -2; ky <= 2; ky++) {
                    int py = static_cast<int>(y) + ky;
                    py = py < 0 ? 0 : (py >= in_h ? in_h - 1 : py);
                    float reg_pixel = static_cast<float>(input_frame[py * in_w + x]);
                    sycl::group_barrier(sg);

                    for (int kx = -2; kx <= 2; kx++) {
                        float pix;
                        int target = static_cast<int>(lane_id) + kx;
                        if (target >= 0 && target < native_sg_size && (x + kx) < in_w) {
                            pix = sycl::select_from_group(sg, reg_pixel, target);
                        } else {
                            int px = static_cast<int>(x) + kx;
                            px = px < 0 ? 0 : (px >= in_w ? in_w - 1 : px);
                            pix = static_cast<float>(input_frame[py * in_w + px]);
                        }
                        sum += pix * w_5x5[(ky + 2) * 5 + (kx + 2)];
                    }
                }
                sum = sum < 0.0f ? 0.0f : (sum > 255.0f ? 255.0f : sum);
                feature_stage1[y * in_w + x] = static_cast<uint8_t>(sum);
            });
        });

        // STAGE 2: Optimization Channel Shrinking Pass
        auto ev_shrink = q.submit([&](handler& h) {
            h.depends_on(ev_extract);
            h.parallel_for(range<2>{static_cast<size_t>(in_h), static_cast<size_t>(in_w)}, [=](id<2> item) {
                size_t y = item[0]; size_t x = item[1];
                size_t idx = y * in_w + x;
                float val = feature_stage1[idx];
                val = val < 0.0f ? 0.0f : (val > 255.0f ? 255.0f : val);
                feature_shrink[idx] = static_cast<uint8_t>(val);
            });
        });

        // STAGE 3: Non-Linear Mapping Convolutions
        auto ev_map = q.submit([&](handler& h) {
            h.depends_on(ev_shrink);
            h.parallel_for(nd_range<2>{global_size, local_size}, [=](nd_item<2> item) {
                auto sg = item.get_sub_group();
                size_t y = item.get_global_id(0); size_t x = item.get_global_id(1);
                size_t lane_id = sg.get_local_id()[0];
                int native_sg_size = sg.get_local_range()[0];
                
                if (y >= in_h || x >= in_w) return;

                float sum = 0.0f;
                for (int ky = -1; ky <= 1; ky++) {
                    int py = static_cast<int>(y) + ky;
                    py = py < 0 ? 0 : (py >= in_h ? in_h - 1 : py);
                    float reg_pixel = static_cast<float>(feature_shrink[py * in_w + x]);
                    sycl::group_barrier(sg);

                    for (int kx = -1; kx <= 1; kx++) {
                        float pix;
                        int target = static_cast<int>(lane_id) + kx;
                        if (target >= 0 && target < native_sg_size && (x + kx) < in_w) {
                            pix = sycl::select_from_group(sg, reg_pixel, target);
                        } else {
                            int px = static_cast<int>(x) + kx;
                            px = px < 0 ? 0 : (px >= in_w ? in_w - 1 : px);
                            pix = static_cast<float>(feature_shrink[py * in_w + px]);
                        }
                        sum += pix * w_3x3[(ky + 1) * 3 + (kx + 1)];
                    }
                }
                sum = sum < 0.0f ? 0.0f : (sum > 255.0f ? 255.0f : sum);
                feature_mapped[y * in_w + x] = static_cast<uint8_t>(sum);
            });
        });

        // STAGE 4: Continuous Blending Structural Reconstructor
        auto ev_upscale = q.submit([&](handler& h) {
            h.depends_on(ev_map);
            h.parallel_for(range<2>{static_cast<size_t>(out_h), static_cast<size_t>(out_w)}, [=](id<2> item) {
                size_t out_y = item[0]; size_t out_x = item[1];
                
                float src_x = out_x * (static_cast<float>(in_w) / static_cast<float>(out_w));
                float src_y = out_y * (static_cast<float>(in_h) / static_cast<float>(out_h));

                int x1 = static_cast<int>(src_x); x1 = x1 < 0 ? 0 : (x1 > in_w - 1 ? in_w - 1 : x1);
                int y1 = static_cast<int>(src_y); y1 = y1 < 0 ? 0 : (y1 > in_h - 1 ? in_h - 1 : y1);
                int x2 = x1 + 1; x2 = x2 > in_w - 1 ? in_w - 1 : x2;
                int y2 = y1 + 1; y2 = y2 > in_h - 1 ? in_h - 1 : y2;

                float dx = src_x - x1;
                float dy = src_y - y1;

                float p1 = feature_mapped[y1 * in_w + x1];
                float p2 = feature_mapped[y1 * in_w + x2];
                float p3 = feature_mapped[y2 * in_w + x1];
                float p4 = feature_mapped[y2 * in_w + x2];

                float base_density = p1 * (1.0f - dx) * (1.0f - dy) +
                                     p2 * dx * (1.0f - dy) +
                                     p3 * (1.0f - dx) * dy +
                                     p4 * dx * dy;

                float grad_h = p2 - p1;
                float grad_v = p3 - p1;
                float structural_variance = (grad_h * grad_h) + (grad_v * grad_v);

                int prev_y = y1 > 0 ? y1 - 1 : 0;
                int next_y = y1 < in_h - 1 ? y1 + 1 : in_h - 1;
                int prev_x = x1 > 0 ? x1 - 1 : 0;
                int next_x = x1 < in_w - 1 ? x1 + 1 : in_w - 1;

                float c_luma = feature_mapped[y1 * in_w + x1];
                float t_luma = feature_mapped[prev_y * in_w + x1];
                float b_luma = feature_mapped[next_y * in_w + x1];
                float l_luma = feature_mapped[y1 * in_w + prev_x];
                float r_luma = feature_mapped[y1 * in_w + next_x];

                float laplacian = (t_luma + b_luma + l_luma + r_luma) - (4.0f * c_luma);
                float edge_weight = structural_variance / (structural_variance + 150.0f);
                
                float final_luma = base_density - (0.35f * edge_weight * laplacian);
                final_luma = final_luma < 0.0f ? 0.0f : (final_luma > 255.0f ? 255.0f : final_luma);

                output_frame[out_y * out_w + out_x] = static_cast<uint8_t>(final_luma);
            });
        });

        ev_upscale.wait();
        std::cout.write(reinterpret_cast<char*>(output_frame), out_size);
        std::cout.flush();
    }

    free(input_frame, q); free(feature_stage1, q); free(feature_shrink, q); free(feature_mapped, q); free(output_frame, q);
    free(w_5x5, q); free(w_3x3, q);

    return 0;
}