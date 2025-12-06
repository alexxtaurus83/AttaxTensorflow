# Setting Up TensorFlow.NET with GPU Support on WSL (Ubuntu 22.04)

This guide provides step-by-step instructions for setting up TensorFlow.NET (v0.150.0 based on TensorFlow v2.15.0) with GPU support on Windows Subsystem for Linux (WSL) using Ubuntu 22.04. This setup enables .NET developers to leverage GPU acceleration for machine learning workloads.

## Prerequisites

- **Windows 10/11** with WSL 2 enabled
- **NVIDIA GPU** with compatible drivers (installed on Windows side)
- **WSL 2** with Ubuntu 22.04 distribution
- Administrative privileges for package installation

---

## Table of Contents
1. [WSL Setup and Configuration](#1-wsl-setup-and-configuration)
2. [Installing Required Dependencies](#2-installing-required-dependencies)
3. [Installing TensorFlow C Library](#3-installing-tensorflow-c-library)
4. [Installing cuDNN for GPU Acceleration](#4-installing-cudnn-for-gpu-acceleration)
5. [Configuring Environment Variables](#5-configuring-environment-variables)
6. [Verifying the Installation](#6-verifying-the-installation)
7. [Optional: Debugger Setup](#7-optional-debugger-setup)

---

## 1. WSL Setup and Configuration

### Why we do this:
We need to ensure WSL 2 is properly configured as it provides better performance and full system call compatibility compared to WSL 1, which is essential for GPU passthrough and CUDA functionality.

### Steps:

1. **Install Ubuntu 22.04 on WSL:**
   ```bash
   wsl --install -d Ubuntu-22.04
   ```

2. **Set default WSL version to 2:**
   ```bash
   wsl --set-default-version 2
   ```

3. **Verify WSL installation:**
   ```bash
   wsl -l -v
   ```
   You should see your Ubuntu distribution listed with version 2.

---

## 2. Installing Required Dependencies

### Why we do this:
TensorFlow requires specific versions of CUDA, compilers, and build tools. We install:
- **CUDA Toolkit 12.2**: Required for GPU computations
- **.NET SDK 8.0**: For building .NET applications
- **Essential utilities**: Development tools and libraries

### Steps:

1. **Update package lists and install dependencies:**
   ```bash
   sudo wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
   sudo dpkg -i cuda-keyring_1.1-1_all.deb && sudo apt update && sudo apt upgrade -y
   ```

2. **Install CUDA, .NET SDK, and essential tools:**
   ```bash
   sudo apt-get install -y \
     cuda-toolkit-12-2 \
     cuda-runtime-12-2 \
     dotnet-sdk-8.0 \
     ca-certificates \
     libc6 \
     libgcc-s1 \
     libgssapi-krb5-2 \
     libicu70 \
     libssl3 \
     libstdc++6 \
     tzdata \
     zlib1g \
     curl \
     wget \
     mc \
     htop \
     tar \
     unzip \
     pkg-config
   ```

---

## 3. Installing TensorFlow C Library

### Why we do this:
TensorFlow.NET requires the native TensorFlow C library (`libtensorflow`) as it's a binding library that calls into the native TensorFlow implementation. This library provides the actual TensorFlow functionality that TensorFlow.NET wraps.

### Steps:

1. **Download the TensorFlow C library for GPU:**
   ```bash
   wget https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-gpu-linux-x86_64-2.15.0.tar.gz
   ```

2. **Extract to system directories:**
   ```bash
   sudo tar -C /usr/local -xzf libtensorflow-gpu-linux-x86_64-2.15.0.tar.gz
   ```

3. **Update library cache:**
   ```bash
   sudo ldconfig
   ```

4. **Verify the installation:**
   ```bash
   ls /usr/local/lib | grep libtensorflow
   ```
   Expected output:
   ```
   libtensorflow_cc.so.2
   libtensorflow_framework.so.2
   ```

---

## 4. Installing cuDNN for GPU Acceleration

### Why we do this:
cuDNN (CUDA Deep Neural Network library) is NVIDIA's GPU-accelerated library for deep neural networks. TensorFlow uses cuDNN for highly optimized implementations of neural network operations, significantly improving performance on NVIDIA GPUs.

### Steps:

1. **Download cuDNN library:**
   ```bash
   wget https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/cudnn-linux-x86_64-8.9.7.29_cuda12-archive.tar.xz
   ```

2. **Extract the archive:**
   ```bash
   tar -xf cudnn-linux-x86_64-8.9.7.29_cuda12-archive.tar.xz
   cd cudnn-linux-x86_64-8.9.7.29_cuda12-archive
   ```

3. **Create CUDA directories (if they don't exist):**
   ```bash
   sudo mkdir -p /usr/local/cuda/include
   sudo mkdir -p /usr/local/cuda/lib64
   ```

4. **Copy cuDNN files to CUDA directories:**
   ```bash
   sudo cp include/* /usr/local/cuda/include/
   sudo cp lib/* /usr/local/cuda/lib64/
   ```

5. **Create symbolic links for compatibility:**
   ```bash
   cd /usr/local/cuda/lib64
   sudo ln -sf libcudnn.so.8 libcudnn.so
   sudo ln -sf libcudnn_ops_train.so.8 libcudnn_ops_train.so
   sudo ln -sf libcudnn_ops_infer.so.8 libcudnn_ops_infer.so
   sudo ln -sf libcudnn_cnn_train.so.8 libcudnn_cnn_train.so
   sudo ln -sf libcudnn_cnn_infer.so.8 libcudnn_cnn_infer.so
   sudo ln -sf libcudnn_adv_train.so.8 libcudnn_adv_train.so
   sudo ln -sf libcudnn_adv_infer.so.8 libcudnn_adv_infer.so
   ```

6. **Update library cache:**
   ```bash
   sudo ldconfig
   ```

7. **Verify cuDNN installation:**
   ```bash
   ls -l /usr/local/cuda/lib64 | grep cudnn
   ```

---

## 5. Configuring Environment Variables

### Why we do this:
Environment variables tell the system where to find the CUDA and TensorFlow libraries. The `LD_LIBRARY_PATH` is crucial for the dynamic linker to locate shared libraries at runtime.

### Steps:

1. **Set temporary environment variables (for current session):**
   ```bash
   export LD_LIBRARY_PATH=/usr/local/lib:/usr/local/cuda/lib64:/usr/local/cuda/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
   ```

2. **Make changes permanent by adding to bash profile:**
   ```bash
   echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
   echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/targets/x86_64-linux/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
   ```

3. **Reload the bash configuration:**
   ```bash
   source ~/.bashrc
   ```

---

## 6. Verifying the Installation

### Why we do this:
Verification ensures all components are properly installed and accessible. This helps identify any configuration issues before attempting to run TensorFlow.NET applications.

### Steps:

1. **Check NVIDIA GPU visibility in WSL:**
   ```bash
   nvidia-smi
   ```
   You should see output showing your GPU information, similar to:
   ```
   +---------------------------------------------------------------------------------------+
   | NVIDIA-SMI 535.154.05             Driver Version: 535.154.05   CUDA Version: 12.2     |
   |-----------------------------------------+----------------------+----------------------+
   | GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
   | Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
   |                                         |                      |               MIG M. |
   |=========================================+======================+======================|
   |   0  NVIDIA GeForce RTX 4090        Off | 00000000:01:00.0  On |                  Off |
   |  0%   48C    P8             22W / 450W |    647MiB / 24564MiB |      4%      Default |
   |                                         |                      |                  N/A |
   +-----------------------------------------+----------------------+----------------------+
   ```

2. **Verify library availability:**
   ```bash
   ldconfig -p | grep cudnn
   ldconfig -p | grep libtensorflow
   ```
   Both commands should show the installed libraries.

---

## 7. Optional: Debugger Setup

### Why we do this:
For developers using Visual Studio or VS Code, having the debugger available in WSL enables seamless debugging of .NET applications running in the Linux environment.

### Steps:

1. **Install VSDBG (Visual Studio Debugger):**
   ```bash
   mkdir -p ~/vsdbg && cd ~/vsdbg
   curl -sSL https://aka.ms/getvsdbgsh | bash /dev/stdin -v latest -l ~/vsdbg -r linux-x64
   ```

---

## Troubleshooting

### Common Issues and Solutions:

1. **"libtensorflow.so not found" error:**
   ```bash
   # Verify library path
   echo $LD_LIBRARY_PATH
   
   # Manually add TensorFlow library path
   export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
   ```

2. **CUDA/cuDNN version mismatch:**
   - Ensure you have exactly CUDA 12.2 and cuDNN 8.9.x
   - TensorFlow 2.15.0 is not compatible with other CUDA versions

3. **WSL 2 GPU issues:**
   ```bash
   # Update WSL kernel
   wsl --update
   
   # Check Windows NVIDIA driver version
   # Must be 535.x or later for WSL 2 GPU support
   ```

4. **Permission denied errors:**
   ```bash
   # Check file permissions
   ls -la /usr/local/lib/libtensorflow*
   
   # Fix permissions if needed
   sudo chmod 755 /usr/local/lib/libtensorflow*
   ```

---

## Additional Resources

- [TensorFlow.NET GitHub Repository](https://github.com/SciSharp/TensorFlow.NET)
- [TensorFlow.NET Documentation](https://scisharp.github.io/TensorFlow.NET/)
- [NVIDIA CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/)
- [WSL 2 GPU Support Documentation](https://docs.microsoft.com/en-us/windows/wsl/tutorials/gpu-compute)

---

## Support

If you encounter issues:
1. Check the TensorFlow.NET GitHub issues
2. Verify all version requirements match exactly
3. Ensure WSL 2 and NVIDIA drivers are up to date
4. Check the `LD_LIBRARY_PATH` environment variable

---

*Last updated: Based on TensorFlow.NET v0.150.0 and TensorFlow v2.15.0*