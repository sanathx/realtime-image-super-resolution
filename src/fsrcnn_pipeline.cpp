#include <sycl/sycl.hpp>
#include <iostream>
#include <fstream>
#include <algorithm> // Explicitly includes std::clamp

using namespace sycl;

int main() {
    // 1. Initialize Level Zero Queue with Profiling
    property_list props{property::queue::enable_profiling()};
    queue q(gpu_selector_v, props);
    std::cout << "FSRCNN Pipeline Active: " << q.get_device().get_info<info::device::name>() << "\n";

    const int in_w = 854;   const int in_h = 480;
    const int out_w = 1920; const int out_h = 1080;
    const int channels = 3;

    const size_t in_size = in_w * in_h * channels;
    const size_t out_size = out_w * out_h * channels;

    // Allocate Unified Shared Memory (USM) Tensor Space
    uint8_t* input_frame     = malloc_shared<uint8_t>(in_size, q);
    uint8_t* feature_stage1   = malloc_shared<uint8_t>(in_size, q); // After 5x5 Extraction
    uint8_t* feature_shrink   = malloc_shared<uint8_t>(in_w * in_h, q); // 1x1 Shrinkage (Squeezed to 1 Channel)
    uint8_t* feature_mapped   = malloc_shared<uint8_t>(in_w * in_h, q); // After 3x3 Non-Linear Mapping
    uint8_t* output_frame    = malloc_shared<uint8_t>(out_size, q);

    std::ifstream infile("C:/Users/sandb/Desktop/iSR_Project/data/input_480p.raw", std::ios::binary);
    if (!infile) {
        std::cerr << "CRITICAL: Ingestion failure.\n";
        return 1;
    }
    infile.read(reinterpret_cast<char*>(input_frame), in_size);
    infile.close();

    // Model Weight Parameters
    float* w_5x5 = malloc_shared<float>(25, q);
    const float kernel_5x5[5][5] = {
        {-1, -1, -1, -1, -1}, {-1, 1, 2, 1, -1}, {-1, 2, 8, 2, -1}, {-1, 1, 2, 1, -1}, {-1, -1, -1, -1, -1}
    };
    for(int i=0; i<5; i++) for(int j=0; j<5; j++) w_5x5[i*5 + j] = kernel_5x5[i][j];

    float* w_3x3 = malloc_shared<float>(9, q);
    const float kernel_3x3[3][3] = {
        {0, -1, 0}, {-1, 5, -1}, {0, -1, 0}
    };
    for(int i=0; i<3; i++) for(int j=0; j<3; j++) w_3x3[i*3 + j] = kernel_3x3[i][j];

    // Execution Shapes (Aligned to 32-lane Subgroups)
    range<2> local_size{8, 32}; 
    range<2> global_size{480, 864}; 

    // ========================================================================
    // STAGE 1: FEATURE EXTRACTION (5x5 Convolution)
    // ========================================================================
    auto ev_extract = q.submit([&](handler& h) {
        h.parallel_for(nd_range<2>{global_size, local_size}, [=](nd_item<2> item) {
            auto sg = item.get_sub_group();
            size_t y = item.get_global_id(0);
            size_t x = item.get_global_id(1);
            size_t lane_id = sg.get_local_id()[0];
            if (y >= in_h || x >= in_w) return;

            for (int c = 0; c < channels; c++) {
                float sum = 0.0f;
                for (int ky = -2; ky <= 2; ky++) {
                    int py = std::clamp(static_cast<int>(y) + ky, 0, in_h - 1);
                    float reg_pixel = static_cast<float>(input_frame[(py * in_w + x) * channels + c]);
                    sycl::group_barrier(sg);

                    for (int kx = -2; kx <= 2; kx++) {
                        float pix;
                        int target = static_cast<int>(lane_id) + kx;
                        if (target >= 0 && target < 32 && (x + kx) < in_w) {
                            pix = sycl::select_from_group(sg, reg_pixel, target);
                        } else {
                            int px = std::clamp(static_cast<int>(x) + kx, 0, in_w - 1);
                            pix = static_cast<float>(input_frame[(py * in_w + px) * channels + c]);
                        }
                        sum += pix * w_5x5[(ky + 2) * 5 + (kx + 2)];
                    }
                }
                if (sum < 0.0f) sum = 0.0f; if (sum > 255.0f) sum = 255.0f;
                feature_stage1[(y * in_w + x) * channels + c] = static_cast<uint8_t>(sum);
            }
        });
    });

    // ========================================================================
    // STAGE 2: CHANNEL SHRINKING (1x1 Bottleneck Convolution)
    // ========================================================================
    auto ev_shrink = q.submit([&](handler& h) {
        h.depends_on(ev_extract);
        h.parallel_for(range<2>{static_cast<size_t>(in_h), static_cast<size_t>(in_w)}, [=](id<2> item) {
            size_t y = item[0]; size_t x = item[1];
            size_t idx = (y * in_w + x) * channels;

            float compressed = 0.299f * feature_stage1[idx] + 
                               0.587f * feature_stage1[idx + 1] + 
                               0.114f * feature_stage1[idx + 2];

            if (compressed < 0.0f) compressed = 0.0f;
            feature_shrink[y * in_w + x] = static_cast<uint8_t>(compressed);
        });
    });

    // ========================================================================
    // STAGE 3: NON-LINEAR MAPPING (3x3 Subgroup Register Convolution)
    // ========================================================================
    auto ev_map = q.submit([&](handler& h) {
        h.depends_on(ev_shrink);
        h.parallel_for(nd_range<2>{global_size, local_size}, [=](nd_item<2> item) {
            auto sg = item.get_sub_group();
            size_t y = item.get_global_id(0); size_t x = item.get_global_id(1);
            size_t lane_id = sg.get_local_id()[0];
            if (y >= in_h || x >= in_w) return;

            float sum = 0.0f;
            for (int ky = -1; ky <= 1; ky++) {
                int py = std::clamp(static_cast<int>(y) + ky, 0, in_h - 1);
                float reg_pixel = static_cast<float>(feature_shrink[py * in_w + x]);
                sycl::group_barrier(sg);

                for (int kx = -1; kx <= 1; kx++) {
                    float pix;
                    int target = static_cast<int>(lane_id) + kx;
                    if (target >= 0 && target < 32 && (x + kx) < in_w) {
                        pix = sycl::select_from_group(sg, reg_pixel, target);
                    } else {
                        int px = std::clamp(static_cast<int>(x) + kx, 0, in_w - 1);
                        pix = static_cast<float>(feature_shrink[py * in_w + px]);
                    }
                    sum += pix * w_3x3[(ky + 1) * 3 + (kx + 1)];
                }
            }
            if (sum < 0.0f) sum = 0.0f; if (sum > 255.0f) sum = 255.0f;
            feature_mapped[y * in_w + x] = static_cast<uint8_t>(sum);
        });
    });

    // ========================================================================
    // STAGE 4: SPATIAL DECONVOLUTION PROJECTION (480p Feature -> 1080p Frame)
    // ========================================================================
    auto ev_upscale = q.submit([&](handler& h) {
        h.depends_on(ev_map);
        h.parallel_for(range<2>{static_cast<size_t>(out_h), static_cast<size_t>(out_w)}, [=](id<2> item) {
            size_t out_y = item[0]; size_t out_x = item[1];
            float scale_x = static_cast<float>(in_w) / static_cast<float>(out_w);
            float scale_y = static_cast<float>(in_h) / static_cast<float>(out_h);

            size_t in_x = std::clamp(static_cast<size_t>(out_x * scale_x), static_cast<size_t>(0), static_cast<size_t>(in_w - 1));
            size_t in_y = std::clamp(static_cast<size_t>(out_y * scale_y), static_cast<size_t>(0), static_cast<size_t>(in_h - 1));

            size_t out_idx = (out_y * out_w + out_x) * channels;
            uint8_t val = feature_mapped[in_y * in_w + in_x];

            output_frame[out_idx]     = val;
            output_frame[out_idx + 1] = val;
            output_frame[out_idx + 2] = val;
        });
    });

    ev_upscale.wait();

    // Extract Latencies from Hardware Timing Registers
    double t_extract = (ev_extract.get_profiling_info<info::event_profiling::command_end>() - ev_extract.get_profiling_info<info::event_profiling::command_start>()) / 1000000.0;
    double t_shrink  = (ev_shrink.get_profiling_info<info::event_profiling::command_end>() - ev_shrink.get_profiling_info<info::event_profiling::command_start>()) / 1000000.0;
    double t_map     = (ev_map.get_profiling_info<info::event_profiling::command_end>() - ev_map.get_profiling_info<info::event_profiling::command_start>()) / 1000000.0;
    double t_upscale = (ev_upscale.get_profiling_info<info::event_profiling::command_end>() - ev_upscale.get_profiling_info<info::event_profiling::command_start>()) / 1000000.0;

    std::ofstream outfile("C:/Users/sandb/Desktop/iSR_Project/data/output_1080p.raw", std::ios::binary);
    outfile.write(reinterpret_cast<char*>(output_frame), out_size);
    outfile.close();

    free(input_frame, q); free(feature_stage1, q); free(feature_shrink, q); free(feature_mapped, q); free(output_frame, q);
    free(w_5x5, q); free(w_3x3, q);

    std::cout << "\n============================================\n";
    std::cout << "STAGE 1: FEATURE EXTRACTION (5x5) : " << t_extract << " ms\n";
    std::cout << "STAGE 2: CHANNEL SHRINKING (1x1)  : " << t_shrink << " ms\n";
    std::cout << "STAGE 3: NON-LINEAR MAPPING (3x3) : " << t_map << " ms\n";
    std::cout << "STAGE 4: 1080p DECONVOLUTION MAP  : " << t_upscale << " ms\n";
    std::cout << "TOTAL PIPELINE EXECUTION TIME     : " << (t_extract + t_shrink + t_map + t_upscale) << " ms\n";
    std::cout << "============================================\n";

    return 0;
}