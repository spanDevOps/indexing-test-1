#!/bin/bash

# Source bashrc to ensure conda is properly initialized
source ~/.bashrc

# Function to check GPU availability
check_gpu() {
    echo "Waiting for GPU to be ready..."
    for i in {1..30}; do
        if nvidia-smi > /dev/null 2>&1; then
            break
        fi
        echo "Waiting for GPU... ($i/30)"
        sleep 1
    done

    nvidia-smi
    if [ $? -ne 0 ]; then
        echo "GPU not available after waiting"
        return 1
    fi
    return 0
}

# Function to test CUDA setup
test_cuda() {
    echo "Testing CUDA setup..."
    python3 -c "
import sys
import torch
import onnxruntime
print('Python executable:', sys.executable)
print('Python version:', sys.version)
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('CUDA version:', torch.version.cuda)
    print('GPU device:', torch.cuda.get_device_name(0))
print('ONNX Runtime version:', onnxruntime.__version__)
print('ONNX Runtime providers:', onnxruntime.get_available_providers())
"
    return $?
}

# Function to verify InsightFace models
check_insightface() {
    echo "Checking InsightFace models..."
    if [ ! -d "/root/.insightface/models/buffalo_l" ]; then
        echo "InsightFace models directory not found"
        return 1
    fi
    return 0
}

# Function to start a service
start_service() {
    local service=$1
    echo "Starting $service service..."
    cd /app
    
    case $service in
        "detection")
            PYTHONPATH=/app python3 -u src/run_detection_service.py &
            echo $! > /tmp/detection_service.pid
            ;;
        "grouping")
            PYTHONPATH=/app python3 -u src/run_grouping_service.py &
            echo $! > /tmp/grouping_service.pid
            ;;
        *)
            echo "Unknown service: $service"
            return 1
            ;;
    esac
}

# Function to stop a service
stop_service() {
    local service=$1
    local pid_file="/tmp/${service}_service.pid"
    
    if [ -f "$pid_file" ]; then
        pid=$(cat "$pid_file")
        echo "Stopping $service service (PID: $pid)..."
        kill -TERM "$pid" 2>/dev/null || true
        rm "$pid_file"
    fi
}

# Function to handle signals
handle_signal() {
    echo "Received shutdown signal"
    stop_service "detection"
    stop_service "grouping"
    exit 0
}

# Register signal handlers
trap 'handle_signal' SIGTERM SIGINT

# Main startup sequence
main() {
    # Perform initial checks
    check_gpu || exit 1
    test_cuda || exit 1
    check_insightface || exit 1

    # Create necessary directories
    mkdir -p /tmp/face_processing
    chmod -R 777 /tmp/face_processing

    # Start services
    start_service "detection"
    start_service "grouping"

    # Keep script running and monitor services
    while true; do
        sleep 10
        
        # Check if services are still running
        for service in "detection" "grouping"; do
            if [ -f "/tmp/${service}_service.pid" ]; then
                pid=$(cat "/tmp/${service}_service.pid")
                if ! kill -0 "$pid" 2>/dev/null; then
                    echo "$service service (PID: $pid) died, restarting..."
                    start_service "$service"
                fi
            else
                echo "$service service not running, starting..."
                start_service "$service"
            fi
        done
    done
}

# Start the main process
main