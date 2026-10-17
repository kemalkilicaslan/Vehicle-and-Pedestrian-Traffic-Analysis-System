# Vehicle and Pedestrian Traffic Analysis System (C++ / OpenCV) - Docker image
FROM ubuntu:24.04

# Build tools plus OpenCV (with dnn/video modules) and video I/O libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake \
        libopencv-dev \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src

# Copy sources and build the executable
COPY CMakeLists.txt ./
COPY src/ ./src/
COPY include/ ./include/
RUN cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build --config Release -j \
    && cp build/vehicle-and-pedestrian-traffic-analysis-system /usr/local/bin/

# Run from the data directory (mounted at runtime) so the binary finds the input
# video and ONNX model by their relative paths and writes outputs next to them.
WORKDIR /data
CMD ["vehicle-and-pedestrian-traffic-analysis-system"]
