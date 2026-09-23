// Small Linux V4L2 MJPEG capture helper. Uses kernel acquisition timestamps.
#include <linux/videodev2.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/select.h>
#include <fcntl.h>
#include <unistd.h>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

void checked_ioctl(int fd, unsigned long request, void* value) {
    int result;
    do { result = ioctl(fd, request, value); } while (result < 0 && errno == EINTR);
    if (result < 0) throw std::runtime_error(std::strerror(errno));
}
struct Camera {
    int fd;
    bool streaming = false;
    struct Mapping { void* address; size_t length; };
    std::vector<Mapping> mappings;
    explicit Camera(const char* path): fd(open(path, O_RDWR | O_NONBLOCK)) {
        if (fd < 0) throw std::runtime_error(std::strerror(errno));
    }
    ~Camera() {
        if (streaming) { int type = V4L2_BUF_TYPE_VIDEO_CAPTURE; ioctl(fd, VIDIOC_STREAMOFF, &type); }
        for (const auto& m : mappings) munmap(m.address, m.length);
        close(fd);
    }
};
int main(int argc, char** argv) {
    try {
        if (argc != 7) throw std::runtime_error("device width height fps jpeg metadata");
        Camera camera(argv[1]);
        const int width = std::stoi(argv[2]), height = std::stoi(argv[3]);
        v4l2_format format{};
        format.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        format.fmt.pix.width = width; format.fmt.pix.height = height;
        format.fmt.pix.pixelformat = V4L2_PIX_FMT_MJPEG;
        format.fmt.pix.field = V4L2_FIELD_ANY;
        checked_ioctl(camera.fd, VIDIOC_S_FMT, &format);
        if (format.fmt.pix.width != static_cast<unsigned>(width) ||
            format.fmt.pix.height != static_cast<unsigned>(height) ||
            format.fmt.pix.pixelformat != V4L2_PIX_FMT_MJPEG)
            throw std::runtime_error("requested MJPEG resolution unavailable; no implicit resize");
        v4l2_streamparm rate{}; rate.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        rate.parm.capture.timeperframe.numerator = 1000;
        rate.parm.capture.timeperframe.denominator = std::stoul(argv[4])*1000;
        checked_ioctl(camera.fd, VIDIOC_S_PARM, &rate);
        v4l2_requestbuffers request{}; request.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        request.memory = V4L2_MEMORY_MMAP; request.count = 4;
        checked_ioctl(camera.fd, VIDIOC_REQBUFS, &request);
        if (!request.count) throw std::runtime_error("no capture buffers");
        for (unsigned i = 0; i < request.count; ++i) {
            v4l2_buffer buffer{}; buffer.type = request.type; buffer.memory = request.memory; buffer.index = i;
            checked_ioctl(camera.fd, VIDIOC_QUERYBUF, &buffer);
            void* address = mmap(nullptr, buffer.length, PROT_READ | PROT_WRITE, MAP_SHARED, camera.fd, buffer.m.offset);
            if (address == MAP_FAILED) throw std::runtime_error("mmap failed");
            camera.mappings.push_back({address, buffer.length});
            checked_ioctl(camera.fd, VIDIOC_QBUF, &buffer);
        }
        int type = request.type; checked_ioctl(camera.fd, VIDIOC_STREAMON, &type); camera.streaming = true;
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(6);
        for (int received = 0; std::chrono::steady_clock::now() < deadline;) {
            fd_set fds; FD_ZERO(&fds); FD_SET(camera.fd, &fds); timeval timeout{1, 0};
            const int ready = select(camera.fd+1, &fds, nullptr, nullptr, &timeout);
            if (ready < 0 && errno != EINTR) throw std::runtime_error("capture select failed");
            if (ready <= 0) continue;
            v4l2_buffer buffer{}; buffer.type = request.type; buffer.memory = request.memory;
            if (ioctl(camera.fd, VIDIOC_DQBUF, &buffer) < 0) {
                if (errno == EAGAIN || errno == EINTR) continue;
                throw std::runtime_error("capture dequeue failed");
            }
            if (buffer.index >= camera.mappings.size()) throw std::runtime_error("invalid capture buffer index");
            if (++received >= 5 && !(buffer.flags & V4L2_BUF_FLAG_ERROR)) {
                if ((buffer.flags & V4L2_BUF_FLAG_TIMESTAMP_MASK) != V4L2_BUF_FLAG_TIMESTAMP_MONOTONIC)
                    throw std::runtime_error("camera lacks monotonic acquisition timestamp");
                if (!buffer.bytesused || buffer.bytesused > camera.mappings[buffer.index].length)
                    throw std::runtime_error("invalid capture payload");
                std::ofstream jpeg(argv[5], std::ios::binary);
                jpeg.write(static_cast<char*>(camera.mappings[buffer.index].address), buffer.bytesused);
                if (!jpeg) throw std::runtime_error("image write failed");
                std::ofstream metadata(argv[6]); metadata.precision(16);
                metadata << "{\"sequence\":" << buffer.sequence << ",\"acquired_monotonic\":"
                         << buffer.timestamp.tv_sec+buffer.timestamp.tv_usec/1e6
                         << ",\"width\":" << width << ",\"height\":" << height
                         << ",\"timestamp_source\":\"v4l2_monotonic\"}\n";
                if (!metadata) throw std::runtime_error("metadata write failed");
                return 0;
            }
            checked_ioctl(camera.fd, VIDIOC_QBUF, &buffer);
        }
        throw std::runtime_error("fresh frame timeout");
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n'; return 1;
    }
}
