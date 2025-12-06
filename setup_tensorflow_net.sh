#!/bin/bash

# TensorFlow.NET Setup Script for WSL (Ubuntu 22.04)
# Version: 1.0
# Compatible with TensorFlow.NET v0.150.0 based on TensorFlow v2.15.0

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if running in WSL
check_wsl() {
    if ! grep -q Microsoft /proc/version; then
        print_error "This script is designed to run in WSL (Windows Subsystem for Linux)."
        exit 1
    fi
    print_info "Running in WSL environment."
}

# Function to check Ubuntu version
check_ubuntu_version() {
    if [[ $(lsb_release -rs) != "22.04" ]]; then
        print_warning "This script is tested on Ubuntu 22.04. You're running $(lsb_release -ds)"
        read -p "Continue anyway? (y/n): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
}

# Function to check if user has sudo privileges
check_sudo() {
    print_info "Checking sudo privileges..."
    if ! sudo -v; then
        print_error "You need sudo privileges to run this script."
        exit 1
    fi
}

# Function to update and upgrade system
update_system() {
    print_info "Updating system packages..."
    sudo apt update && sudo apt upgrade -y
}

# Function to install required dependencies
install_dependencies() {
    print_info "Installing required dependencies..."
    
    # Download and install CUDA keyring
    print_info "Setting up CUDA repository..."
    sudo wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
    sudo dpkg -i cuda-keyring_1.1-1_all.deb
    
    # Install all required packages
    print_info "Installing CUDA, .NET SDK, and development tools..."
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
}

# Function to install TensorFlow C library
install_tensorflow_lib() {
    print_info "Installing TensorFlow C library..."
    
    # Download TensorFlow GPU library
    if [ ! -f "libtensorflow-gpu-linux-x86_64-2.15.0.tar.gz" ]; then
        wget -q https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-gpu-linux-x86_64-2.15.0.tar.gz
    fi
    
    # Extract to /usr/local
    sudo tar -C /usr/local -xzf libtensorflow-gpu-linux-x86_64-2.15.0.tar.gz
    
    # Update library cache
    sudo ldconfig
    
    # Verify installation
    print_info "Verifying TensorFlow library installation..."
    if ls /usr/local/lib | grep -q libtensorflow; then
        print_info "TensorFlow libraries found:"
        ls /usr/local/lib | grep libtensorflow
    else
        print_error "TensorFlow libraries not found!"
        exit 1
    fi
}

# Function to install cuDNN
install_cudnn() {
    print_info "Installing cuDNN..."
    
    # Download cuDNN
    if [ ! -f "cudnn-linux-x86_64-8.9.7.29_cuda12-archive.tar.xz" ]; then
        wget -q https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/cudnn-linux-x86_64-8.9.7.29_cuda12-archive.tar.xz
    fi
    
    # Extract cuDNN
    tar -xf cudnn-linux-x86_64-8.9.7.29_cuda12-archive.tar.xz
    
    # Create CUDA directories
    sudo mkdir -p /usr/local/cuda/include
    sudo mkdir -p /usr/local/cuda/lib64
    
    # Copy cuDNN files
    sudo cp cudnn-linux-x86_64-8.9.7.29_cuda12-archive/include/* /usr/local/cuda/include/
    sudo cp cudnn-linux-x86_64-8.9.7.29_cuda12-archive/lib/* /usr/local/cuda/lib64/
    
    # Create symbolic links
    cd /usr/local/cuda/lib64
    sudo ln -sf libcudnn.so.8 libcudnn.so
    sudo ln -sf libcudnn_ops_train.so.8 libcudnn_ops_train.so
    sudo ln -sf libcudnn_ops_infer.so.8 libcudnn_ops_infer.so
    sudo ln -sf libcudnn_cnn_train.so.8 libcudnn_cnn_train.so
    sudo ln -sf libcudnn_cnn_infer.so.8 libcudnn_cnn_infer.so
    sudo ln -sf libcudnn_adv_train.so.8 libcudnn_adv_train.so
    sudo ln -sf libcudnn_adv_infer.so.8 libcudnn_adv_infer.so
    
    # Update library cache
    sudo ldconfig
    
    # Verify installation
    print_info "Verifying cuDNN installation..."
    if ls -l /usr/local/cuda/lib64 | grep -q cudnn; then
        print_info "cuDNN libraries installed successfully."
    else
        print_error "cuDNN installation failed!"
        exit 1
    fi
}

# Function to configure environment variables
configure_environment() {
    print_info "Configuring environment variables..."
    
    # Add to bashrc if not already present
    if ! grep -q "/usr/local/cuda/bin" ~/.bashrc; then
        echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
    fi
    
    if ! grep -q "/usr/local/cuda/lib64" ~/.bashrc; then
        echo 'export LD_LIBRARY_PATH=/usr/local/lib:/usr/local/cuda/lib64:/usr/local/cuda/targets/x86_64-linux/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
    fi
    
    # Set temporary environment variables
    export LD_LIBRARY_PATH=/usr/local/lib:/usr/local/cuda/lib64:/usr/local/cuda/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
    export PATH=/usr/local/cuda/bin:$PATH
    
    # Reload bashrc
    source ~/.bashrc
}

# Function to verify installation
verify_installation() {
    print_info "Verifying installation..."
    
    # Check NVIDIA GPU
    print_info "Checking NVIDIA GPU..."
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi
    else
        print_warning "nvidia-smi not found. Make sure NVIDIA drivers are installed on Windows."
    fi
    
    # Check libraries
    print_info "Checking libraries..."
    if ldconfig -p | grep -q libcudnn; then
        print_info "cuDNN library found."
    else
        print_warning "cuDNN library not found in cache."
    fi
    
    if ldconfig -p | grep -q libtensorflow; then
        print_info "TensorFlow library found."
    else
        print_warning "TensorFlow library not found in cache."
    fi
    
    # Check .NET SDK
    print_info "Checking .NET SDK..."
    dotnet --version
    
    # Check CUDA
    print_info "Checking CUDA..."
    if [ -f "/usr/local/cuda/bin/nvcc" ]; then
        nvcc --version | grep release
    else
        print_warning "CUDA compiler (nvcc) not found."
    fi
}

# Function to setup debugger (optional)
setup_debugger() {
    read -p "Install VS Code debugger? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_info "Setting up VS Code debugger..."
        mkdir -p ~/vsdbg && cd ~/vsdbg
        curl -sSL https://aka.ms/getvsdbgsh | bash /dev/stdin -v latest -l ~/vsdbg -r linux-x64
        print_info "Debugger installed in ~/vsdbg"
    fi
}

# Main execution
main() {
    print_info "Starting TensorFlow.NET setup for WSL..."
    print_info "This script will install:"
    echo "  - CUDA Toolkit 12.2"
    echo "  - cuDNN 8.9.7"
    echo "  - .NET SDK 8.0"
    echo "  - TensorFlow C Library 2.15.0"
    echo ""
    
    # Check system
    check_wsl
    check_ubuntu_version
    check_sudo
    
    # Get confirmation
    read -p "Continue with installation? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_info "Installation cancelled."
        exit 0
    fi
    
    # Execute all steps
    update_system
    install_dependencies
    install_tensorflow_lib
    install_cudnn
    configure_environment
    verify_installation
    setup_debugger
    
    print_info ""
    print_info "================================================"
    print_info "✅ Setup completed successfully!"
    print_info "================================================"
    print_info ""
    print_info "Next steps:"
    echo "1. Restart your WSL terminal or run: source ~/.bashrc"
    echo "2. Run the test script: ./test_tensorflow_net.sh"
    echo "3. Then run: cd test_tensorflow && dotnet run"
    print_info ""
    print_info "For troubleshooting, refer to the MD documentation."
}

# Run main function
main "$@"