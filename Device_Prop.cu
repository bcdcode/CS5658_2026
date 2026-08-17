#include <iostream>
#include <iomanip>
#include <cuda_runtime.h>

int getCoresPerSM(int major, int minor)
{
    switch (major)
    {
        case 2: // Fermi
            return (minor == 1) ? 48 : 32;

        case 3: // Kepler
            return 192;

        case 5: // Maxwell
            return 128;

        case 6: // Pascal
            if (minor == 0) return 64;
            return 128;

        case 7: // Volta / Turing
            return 64;

        case 8: // Ampere / Ada
            if (minor == 0) return 64;   // GA100
            if (minor == 6) return 128;  // GA10x
            if (minor == 7) return 64;   // GA10B
            if (minor == 9) return 128;  // Ada
            return 64;

        case 9: // Hopper
            return 128;

        default:
            return 128;
    }
}

void printDeviceProperties(const cudaDeviceProp& prop)
{
    int coresPerSM = getCoresPerSM(prop.major, prop.minor);
    long long totalCores =
        (long long)prop.multiProcessorCount * coresPerSM;

    std::cout << "\n============================================================\n";
    std::cout << "                    CUDA DEVICE INFORMATION\n";
    std::cout << "============================================================\n";

    // ---------------------------------------------------------
    // Basic Information
    // ---------------------------------------------------------
    std::cout << "\n[ Basic Information ]\n";

    std::cout << "GPU Name                     : "
              << prop.name << "\n";

    std::cout << "Compute Capability           : "
              << prop.major << "." << prop.minor << "\n";

    std::cout << "Device Clock Rate            : "
              << prop.clockRate / 1000.0 << " MHz\n";

    std::cout << "Memory Clock Rate            : "
              << prop.memoryClockRate / 1000.0 << " MHz\n";

    std::cout << "Memory Bus Width             : "
              << prop.memoryBusWidth << " bits\n";

    // ---------------------------------------------------------
    // SM / CUDA Core Information
    // ---------------------------------------------------------
    std::cout << "\n[ SM / CUDA Core Information ]\n";

    std::cout << "Streaming Multiprocessors    : "
              << prop.multiProcessorCount << "\n";

    std::cout << "CUDA Cores per SM            : "
              << coresPerSM << "\n";

    std::cout << "Total CUDA Cores             : "
              << totalCores << "\n";

    std::cout << "Warp Size                    : "
              << prop.warpSize << " threads\n";

    std::cout << "Max Threads per SM           : "
              << prop.maxThreadsPerMultiProcessor << "\n";

    std::cout << "Max Threads per Block        : "
              << prop.maxThreadsPerBlock << "\n";

    std::cout << "Max Blocks per SM            : "
              << prop.maxBlocksPerMultiProcessor << "\n";

    // ---------------------------------------------------------
    // Global Memory
    // ---------------------------------------------------------
    std::cout << "\n[ Global Memory ]\n";

    std::cout << "Total Global Memory          : "
              << prop.totalGlobalMem / (1024.0 * 1024 * 1024)
              << " GB\n";

    std::cout << "Memory Bus Width             : "
              << prop.memoryBusWidth << " bits\n";

    // Approximate theoretical memory bandwidth
    double memoryBandwidth =
        2.0 * prop.memoryClockRate * 1000.0 *
        (prop.memoryBusWidth / 8.0) / 1e9;

    std::cout << "Theoretical Memory Bandwidth : "
              << memoryBandwidth << " GB/s\n";

    // ---------------------------------------------------------
    // Shared Memory
    // ---------------------------------------------------------
    std::cout << "\n[ Shared Memory ]\n";

    std::cout << "Shared Memory per Block      : "
              << prop.sharedMemPerBlock / 1024.0
              << " KB\n";

    std::cout << "Shared Memory per SM         : "
              << prop.sharedMemPerMultiprocessor / 1024.0
              << " KB\n";

    // ---------------------------------------------------------
    // Registers
    // ---------------------------------------------------------
    std::cout << "\n[ Registers ]\n";

    std::cout << "Registers per Block          : "
              << prop.regsPerBlock << "\n";

    std::cout << "Registers per SM             : "
              << prop.regsPerMultiprocessor << "\n";

    // ---------------------------------------------------------
    // L1 / L2 Cache
    // ---------------------------------------------------------
    std::cout << "\n[ Cache ]\n";

    std::cout << "L2 Cache Size                : "
              << prop.l2CacheSize / 1024.0
              << " KB\n";

    std::cout << "L1 Cache / Shared Memory    : "
              << prop.sharedMemPerMultiprocessor / 1024.0
              << " KB per SM\n";

    // ---------------------------------------------------------
    // Execution Limits
    // ---------------------------------------------------------
    std::cout << "\n[ Execution Limits ]\n";

    std::cout << "Max Threads Dimension       : "
              << prop.maxThreadsDim[0] << " x "
              << prop.maxThreadsDim[1] << " x "
              << prop.maxThreadsDim[2] << "\n";

    std::cout << "Max Grid Size               : "
              << prop.maxGridSize[0] << " x "
              << prop.maxGridSize[1] << " x "
              << prop.maxGridSize[2] << "\n";

    std::cout << "Max Registers per Block     : "
              << prop.regsPerBlock << "\n";

    std::cout << "Max Shared Memory per Block : "
              << prop.sharedMemPerBlock / 1024.0
              << " KB\n";

    // ---------------------------------------------------------
    // Texture Information
    // ---------------------------------------------------------
    std::cout << "\n[ Texture Limits ]\n";

    std::cout << "Max Texture 1D              : "
              << prop.maxTexture1D << "\n";

    std::cout << "Max Texture 2D              : "
              << prop.maxTexture2D[0] << " x "
              << prop.maxTexture2D[1] << "\n";

    std::cout << "Max Texture 3D              : "
              << prop.maxTexture3D[0] << " x "
              << prop.maxTexture3D[1] << " x "
              << prop.maxTexture3D[2] << "\n";

    // ---------------------------------------------------------
    // Concurrent Execution
    // ---------------------------------------------------------
    std::cout << "\n[ Concurrency / Execution ]\n";

    std::cout << "Concurrent Kernels           : "
              << (prop.concurrentKernels ? "Yes" : "No") << "\n";

    std::cout << "Concurrent Managed Access   : "
              << (prop.concurrentManagedAccess ? "Yes" : "No") << "\n";

    std::cout << "Can Map Host Memory          : "
              << (prop.canMapHostMemory ? "Yes" : "No") << "\n";

    std::cout << "Unified Addressing           : "
              << (prop.unifiedAddressing ? "Yes" : "No") << "\n";

    std::cout << "Pageable Memory Access       : "
              << (prop.pageableMemoryAccess ? "Yes" : "No") << "\n";

    // ---------------------------------------------------------
    // Architecture Features
    // ---------------------------------------------------------
    std::cout << "\n[ Architecture Features ]\n";

    std::cout << "ECC Enabled                  : "
              << (prop.ECCEnabled ? "Yes" : "No") << "\n";

    std::cout << "Integrated GPU               : "
              << (prop.integrated ? "Yes" : "No") << "\n";

    std::cout << "Kernel Execution Timeout     : "
              << (prop.kernelExecTimeoutEnabled ? "Yes" : "No") << "\n";

    std::cout << "Memory Mapping Supported    : "
              << (prop.canMapHostMemory ? "Yes" : "No") << "\n";

    // ---------------------------------------------------------
    // PCI / Hardware Information
    // ---------------------------------------------------------
    std::cout << "\n[ PCI / Hardware Information ]\n";

    std::cout << "PCI Bus                      : "
              << prop.pciBusID << "\n";

    std::cout << "PCI Device                   : "
              << prop.pciDeviceID << "\n";

    std::cout << "PCI Domain                   : "
              << prop.pciDomainID << "\n";

    // ---------------------------------------------------------
    // Cooperative Groups
    // ---------------------------------------------------------
    std::cout << "\n[ Cooperative Execution ]\n";

    std::cout << "Cooperative Launch           : "
              << (prop.cooperativeLaunch ? "Yes" : "No") << "\n";

    std::cout << "Multi-Device Cooperative    : "
              << (prop.cooperativeMultiDeviceLaunch ? "Yes" : "No") << "\n";

    // ---------------------------------------------------------
    // Compute Mode
    // ---------------------------------------------------------
    std::cout << "\n[ Compute Mode ]\n";

    std::cout << "Compute Mode                 : ";

    switch (prop.computeMode)
    {
        case cudaComputeModeDefault:
            std::cout << "Default\n";
            break;

        case cudaComputeModeExclusive:
            std::cout << "Exclusive\n";
            break;

        case cudaComputeModeProhibited:
            std::cout << "Prohibited\n";
            break;

        case cudaComputeModeExclusiveProcess:
            std::cout << "Exclusive Process\n";
            break;

        default:
            std::cout << "Unknown\n";
    }

    // ---------------------------------------------------------
    // Host / Device Performance
    // ---------------------------------------------------------
    std::cout << "\n[ Performance Related ]\n";

    std::cout << "Async Engine Count          : "
              << prop.asyncEngineCount << "\n";

    std::cout << "Memory Bus Width            : "
              << prop.memoryBusWidth << " bits\n";

    std::cout << "Memory Clock Rate           : "
              << prop.memoryClockRate / 1000.0
              << " MHz\n";

    std::cout << "GPU Clock Rate              : "
              << prop.clockRate / 1000.0
              << " MHz\n";

    std::cout << "\n============================================================\n";
}

int main()
{
    int deviceCount = 0;

    cudaError_t error = cudaGetDeviceCount(&deviceCount);

    if (error != cudaSuccess)
    {
        std::cerr << "CUDA Error: "
                  << cudaGetErrorString(error)
                  << std::endl;

        return 1;
    }

    std::cout << "Found "
              << deviceCount
              << " CUDA device(s).\n";

    for (int i = 0; i < deviceCount; ++i)
    {
        cudaDeviceProp prop;

        error = cudaGetDeviceProperties(&prop, i);

        if (error != cudaSuccess)
        {
            std::cerr << "Error getting properties for device "
                      << i << ": "
                      << cudaGetErrorString(error)
                      << std::endl;

            continue;
        }

        std::cout << "\n\nDevice " << i << "\n";

        printDeviceProperties(prop);
    }

    return 0;
}