#include <sycl/sycl.hpp>
#include <iostream>

using namespace sycl;

int main() {
    // 1. Target the integrated GPU specifically
    queue q(gpu_selector_v);

    auto device = q.get_device();
    std::cout << "====================================" << "\n";
    std::cout << "--- INTEL iGPU HARDWARE HANDSHAKE ---" << "\n";
    std::cout << "====================================" << "\n";
    std::cout << "Targeting Device: " << device.get_info<info::device::name>() << "\n";
    std::cout << "Driver Backend:   " << q.get_backend() << "\n"; // We want Level-Zero!

    // 2. Query Sub-group sizes (Crucial for our internal register shuffling)
    auto sg_sizes = device.get_info<info::device::sub_group_sizes>();
    std::cout << "Supported Sub-groups: ";
    for (auto s : sg_sizes) std::cout << s << " ";
    std::cout << "\n";

    // 3. Test the 'Zero-Copy' Unified Shared Memory (USM) allocation
    // We try to allocate a chunk of memory visible to BOTH CPU and iGPU simultaneously
    float* shared_data = malloc_shared<float>(1024 * 1024, q);

    if (shared_data != nullptr) {
        std::cout << "USM Handshake:    SUCCESS (Zero-Copy Link Active)" << "\n";
        free(shared_data, q); // Always clean up your memory pointers
    } else {
        std::cout << "USM Handshake:    FAILED (Check Level-Zero configuration)" << "\n";
    }
    std::cout << "====================================" << "\n";

    return 0;
}