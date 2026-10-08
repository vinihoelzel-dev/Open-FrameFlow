#include "protocol.h"

#include <rife.h>

#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

#include <arpa/inet.h>
#include <algorithm>
#include <cerrno>
#include <csignal>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>
#include <vector>

namespace {

volatile std::sig_atomic_t stop_requested = 0;

enum class Algorithm {
    Rife,
    Dis
};

void request_stop(int) {
    stop_requested = 1;
}

bool read_exact(int fd, void *buffer, size_t size) {
    auto *bytes = static_cast<unsigned char *>(buffer);
    size_t offset = 0;
    while (offset < size) {
        const ssize_t count = recv(fd, bytes + offset, size - offset, 0);
        if (count == 0) return false;
        if (count < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        offset += static_cast<size_t>(count);
    }
    return true;
}

bool write_exact(int fd, const void *buffer, size_t size) {
    const auto *bytes = static_cast<const unsigned char *>(buffer);
    size_t offset = 0;
    while (offset < size) {
        const ssize_t count = send(fd, bytes + offset, size - offset, MSG_NOSIGNAL);
        if (count < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        if (count == 0) return false;
        offset += static_cast<size_t>(count);
    }
    return true;
}

uint32_t decode_word(uint32_t value) {
    return ntohl(value);
}

void encode_response(uint32_t (&header)[FRAMEFLOW_HEADER_WORDS],
                     uint32_t status, uint32_t width, uint32_t height,
                     uint32_t factor, uint32_t frame_bytes,
                     uint32_t sequence_hi, uint32_t sequence_lo) {
    const uint32_t values[FRAMEFLOW_HEADER_WORDS] = {
        FRAMEFLOW_MAGIC, FRAMEFLOW_VERSION, status, width, height, factor,
        frame_bytes, sequence_hi, sequence_lo, 0
    };
    for (uint32_t i = 0; i < FRAMEFLOW_HEADER_WORDS; ++i)
        header[i] = htonl(values[i]);
}

class OpticalFlowInterpolator {
public:
    OpticalFlowInterpolator()
        : dis_(cv::DISOpticalFlow::create(cv::DISOpticalFlow::PRESET_FAST)) {}

    void prepare(const unsigned char *input0, const unsigned char *input1,
                 uint32_t width, uint32_t height) {
        const int w = static_cast<int>(width);
        const int h = static_cast<int>(height);
        width_ = w;
        height_ = h;
        const cv::Mat frame0(h, w, CV_8UC3,
                             const_cast<unsigned char *>(input0));
        const cv::Mat frame1(h, w, CV_8UC3,
                             const_cast<unsigned char *>(input1));
        cv::Mat gray0;
        cv::Mat gray1;
        cv::cvtColor(frame0, gray0, cv::COLOR_RGB2GRAY);
        cv::cvtColor(frame1, gray1, cv::COLOR_RGB2GRAY);

        dis_->calc(gray0, gray1, flow01_);
        dis_->calc(gray1, gray0, flow10_);

        grid_x_.create(h, w, CV_32FC1);
        grid_y_.create(h, w, CV_32FC1);
        for (int y = 0; y < h; ++y) {
            float *x_row = grid_x_.ptr<float>(y);
            float *y_row = grid_y_.ptr<float>(y);
            for (int x = 0; x < w; ++x) {
                x_row[x] = static_cast<float>(x);
                y_row[x] = static_cast<float>(y);
            }
        }
    }

    void interpolate(const unsigned char *input0, const unsigned char *input1,
                     float timestep, unsigned char *output) const {
        const int w = width_;
        const int h = height_;
        const cv::Mat frame0(h, w, CV_8UC3,
                             const_cast<unsigned char *>(input0));
        const cv::Mat frame1(h, w, CV_8UC3,
                             const_cast<unsigned char *>(input1));
        cv::Mat map0_x = grid_x_.clone();
        cv::Mat map0_y = grid_y_.clone();
        cv::Mat map1_x = grid_x_.clone();
        cv::Mat map1_y = grid_y_.clone();
        refine_inverse_map(flow01_, timestep, grid_x_, grid_y_,
                           map0_x, map0_y);
        refine_inverse_map(flow10_, 1.f - timestep, grid_x_, grid_y_,
                           map1_x, map1_y);

        cv::Mat sampled_flow0;
        cv::Mat sampled_flow1;
        cv::remap(flow01_, sampled_flow0, map0_x, map0_y, cv::INTER_LINEAR,
                  cv::BORDER_REPLICATE);
        cv::remap(flow10_, sampled_flow1, map1_x, map1_y, cv::INTER_LINEAR,
                  cv::BORDER_REPLICATE);
        std::vector<cv::Mat> flow0_channels;
        std::vector<cv::Mat> flow1_channels;
        cv::split(sampled_flow0, flow0_channels);
        cv::split(sampled_flow1, flow1_channels);

        cv::Mat reverse_at_end0;
        cv::Mat forward_at_end1;
        cv::remap(flow10_, reverse_at_end0,
                  map0_x + flow0_channels[0], map0_y + flow0_channels[1],
                  cv::INTER_LINEAR, cv::BORDER_REPLICATE);
        cv::remap(flow01_, forward_at_end1,
                  map1_x + flow1_channels[0], map1_y + flow1_channels[1],
                  cv::INTER_LINEAR, cv::BORDER_REPLICATE);

        cv::Mat error0;
        cv::Mat error1;
        std::vector<cv::Mat> reverse0_channels;
        std::vector<cv::Mat> forward1_channels;
        cv::split(reverse_at_end0, reverse0_channels);
        cv::split(forward_at_end1, forward1_channels);
        cv::Mat residual0_x = flow0_channels[0] + reverse0_channels[0];
        cv::Mat residual0_y = flow0_channels[1] + reverse0_channels[1];
        cv::Mat residual1_x = flow1_channels[0] + forward1_channels[0];
        cv::Mat residual1_y = flow1_channels[1] + forward1_channels[1];
        cv::magnitude(residual0_x, residual0_y, error0);
        cv::magnitude(residual1_x, residual1_y, error1);
        cv::Mat confidence0;
        cv::Mat confidence1;
        cv::divide(1.0, error0 + 1.0, confidence0);
        cv::divide(1.0, error1 + 1.0, confidence1);

        cv::Mat warped0;
        cv::Mat warped1;
        cv::remap(frame0, warped0, map0_x, map0_y, cv::INTER_LINEAR,
                  cv::BORDER_REPLICATE);
        cv::remap(frame1, warped1, map1_x, map1_y, cv::INTER_LINEAR,
                  cv::BORDER_REPLICATE);

        cv::Mat result(h, w, CV_8UC3, output);
        for (int y = 0; y < h; ++y) {
            const cv::Vec3b *row0 = warped0.ptr<cv::Vec3b>(y);
            const cv::Vec3b *row1 = warped1.ptr<cv::Vec3b>(y);
            const float *confidence0_row = confidence0.ptr<float>(y);
            const float *confidence1_row = confidence1.ptr<float>(y);
            cv::Vec3b *result_row = result.ptr<cv::Vec3b>(y);
            for (int x = 0; x < w; ++x) {
                const float weight0 = (1.f - timestep) * confidence0_row[x];
                const float weight1 = timestep * confidence1_row[x];
                const float weight_sum = weight0 + weight1;
                for (int channel = 0; channel < 3; ++channel) {
                    const float value =
                        (row0[x][channel] * weight0 +
                         row1[x][channel] * weight1) / weight_sum;
                    result_row[x][channel] =
                        cv::saturate_cast<unsigned char>(value);
                }
            }
        }
    }

private:
    static void refine_inverse_map(const cv::Mat &flow, float amount,
                                   const cv::Mat &grid_x,
                                   const cv::Mat &grid_y, cv::Mat &map_x,
                                   cv::Mat &map_y) {
        for (int iteration = 0; iteration < 3; ++iteration) {
            cv::Mat sampled_flow;
            cv::remap(flow, sampled_flow, map_x, map_y, cv::INTER_LINEAR,
                      cv::BORDER_REPLICATE);
            std::vector<cv::Mat> channels;
            cv::split(sampled_flow, channels);
            map_x = grid_x - amount * channels[0];
            map_y = grid_y - amount * channels[1];
        }
    }

    cv::Ptr<cv::DISOpticalFlow> dis_;
    cv::Mat flow01_;
    cv::Mat flow10_;
    cv::Mat grid_x_;
    cv::Mat grid_y_;
    int width_ = 0;
    int height_ = 0;
};

bool handle_client(int fd, RIFE *rife, OpticalFlowInterpolator *optical_flow,
                   Algorithm algorithm, float inference_scale) {
    for (;;) {
        uint32_t request[FRAMEFLOW_HEADER_WORDS];
        if (!read_exact(fd, request, sizeof(request))) return true;
        for (uint32_t &word : request) word = decode_word(word);

        const uint32_t width = request[FRAMEFLOW_REQ_WIDTH];
        const uint32_t height = request[FRAMEFLOW_REQ_HEIGHT];
        const uint32_t factor = request[FRAMEFLOW_REQ_FACTOR];
        const uint32_t frame_bytes = request[FRAMEFLOW_REQ_FRAME_BYTES];
        const uint32_t sequence_hi = request[FRAMEFLOW_REQ_SEQUENCE_HI];
        const uint32_t sequence_lo = request[FRAMEFLOW_REQ_SEQUENCE_LO];
        const uint64_t pixels = static_cast<uint64_t>(width) * height;
        const bool valid =
            request[FRAMEFLOW_REQ_MAGIC] == FRAMEFLOW_MAGIC &&
            request[FRAMEFLOW_REQ_VERSION] == FRAMEFLOW_VERSION &&
            request[FRAMEFLOW_REQ_FORMAT] == FRAMEFLOW_RGB24 &&
            width > 0 && height > 0 && pixels <= FRAMEFLOW_MAX_PIXELS &&
            factor >= FRAMEFLOW_MIN_FACTOR && factor <= FRAMEFLOW_MAX_FACTOR &&
            pixels <= std::numeric_limits<uint32_t>::max() / 3u &&
            frame_bytes == pixels * 3u;
        if (!valid) {
            uint32_t response[FRAMEFLOW_HEADER_WORDS];
            encode_response(response, FRAMEFLOW_STATUS_BAD_REQUEST, width, height,
                            factor, frame_bytes, sequence_hi, sequence_lo);
            write_exact(fd, response, sizeof(response));
            return false;
        }

        std::vector<unsigned char> input0(frame_bytes);
        std::vector<unsigned char> input1(frame_bytes);
        if (!read_exact(fd, input0.data(), input0.size()) ||
            !read_exact(fd, input1.data(), input1.size()))
            return true;

        uint32_t inference_width = width;
        uint32_t inference_height = height;
        std::vector<unsigned char> inference_input0;
        std::vector<unsigned char> inference_input1;
        std::vector<unsigned char> inference_output;
        ncnn::Mat in0;
        ncnn::Mat in1;
        if (algorithm == Algorithm::Rife) {
            inference_width = std::max(
                1u, static_cast<uint32_t>(std::lround(width * inference_scale)));
            inference_height = std::max(
                1u, static_cast<uint32_t>(std::lround(height * inference_scale)));
            const uint64_t inference_pixels =
                static_cast<uint64_t>(inference_width) * inference_height;
            unsigned char *rife_input0 = input0.data();
            unsigned char *rife_input1 = input1.data();
            if (inference_width != width || inference_height != height) {
                const size_t inference_frame_bytes =
                    static_cast<size_t>(inference_pixels) * 3u;
                inference_input0.resize(inference_frame_bytes);
                inference_input1.resize(inference_frame_bytes);
                ncnn::Mat resized0 = ncnn::Mat::from_pixels_resize(
                    input0.data(), ncnn::Mat::PIXEL_RGB, width, height,
                    inference_width, inference_height);
                ncnn::Mat resized1 = ncnn::Mat::from_pixels_resize(
                    input1.data(), ncnn::Mat::PIXEL_RGB, width, height,
                    inference_width, inference_height);
                resized0.to_pixels(inference_input0.data(), ncnn::Mat::PIXEL_RGB);
                resized1.to_pixels(inference_input1.data(), ncnn::Mat::PIXEL_RGB);
                rife_input0 = inference_input0.data();
                rife_input1 = inference_input1.data();
            }

            // RIFE recebe RGB24 empacotado em Mat views; seu preprocessamento
            // interno normaliza os valores para a inferencia.
            in0 = ncnn::Mat(static_cast<int>(inference_width),
                            static_cast<int>(inference_height),
                            rife_input0, 3u, 1u);
            in1 = ncnn::Mat(static_cast<int>(inference_width),
                            static_cast<int>(inference_height),
                            rife_input1, 3u, 1u);
            if (inference_width != width || inference_height != height)
                inference_output.resize(
                    static_cast<size_t>(inference_pixels) * 3u);
        }

        std::vector<unsigned char> outputs(static_cast<size_t>(frame_bytes) *
                                           (factor - 1u));
        uint32_t status = FRAMEFLOW_STATUS_OK;
        if (algorithm == Algorithm::Dis) {
            try {
                optical_flow->prepare(input0.data(), input1.data(), width, height);
            } catch (const cv::Exception &error) {
                std::fprintf(stderr,
                    "[frameflow] DIS preparo falhou (seq=%u:%u): %s\n",
                    sequence_hi, sequence_lo, error.what());
                status = FRAMEFLOW_STATUS_INFERENCE_FAILED;
            }
        }

        for (uint32_t step = 1; status == FRAMEFLOW_STATUS_OK &&
             step < factor; ++step) {
            unsigned char *output = outputs.data() +
                static_cast<size_t>(step - 1u) * frame_bytes;
            const float timestep = static_cast<float>(step) / factor;
            if (algorithm == Algorithm::Dis) {
                try {
                    optical_flow->interpolate(input0.data(), input1.data(),
                                              timestep, output);
                } catch (const cv::Exception &error) {
                    std::fprintf(stderr,
                        "[frameflow] DIS falhou (seq=%u:%u, t=%.3f): %s\n",
                        sequence_hi, sequence_lo, timestep, error.what());
                    status = FRAMEFLOW_STATUS_INFERENCE_FAILED;
                    break;
                }
            } else {
                unsigned char *rife_output =
                    inference_output.empty() ? output : inference_output.data();

                // A API RIFE escreve RGB24 no buffer fornecido; ela nao aloca a saida.
                ncnn::Mat out_mat(static_cast<int>(inference_width),
                                  static_cast<int>(inference_height),
                                  rife_output, 3u, 1u);
                const int result = rife->process(in0, in1, timestep, out_mat);
                if (result != 0) {
                    std::fprintf(stderr,
                        "[frameflow] RIFE process falhou: %d "
                        "(seq=%u:%u, t=%.3f).\n",
                        result, sequence_hi, sequence_lo, timestep);
                    status = FRAMEFLOW_STATUS_INFERENCE_FAILED;
                    break;
                }
                if (!inference_output.empty()) {
                    ncnn::Mat resized = ncnn::Mat::from_pixels_resize(
                        inference_output.data(), ncnn::Mat::PIXEL_RGB,
                        inference_width, inference_height, width, height);
                    resized.to_pixels(output, ncnn::Mat::PIXEL_RGB);
                }
            }
        }

        uint32_t response[FRAMEFLOW_HEADER_WORDS];
        encode_response(response, status, width, height, factor, frame_bytes,
                        sequence_hi, sequence_lo);
        if (!write_exact(fd, response, sizeof(response))) return true;
        if (status != FRAMEFLOW_STATUS_OK) return false;
        if (!write_exact(fd, outputs.data(), outputs.size())) return true;
    }
}

bool parse_options(int argc, char **argv, std::string &socket_path,
                   std::filesystem::path &model_path, int &gpu_id,
                   float &inference_scale, bool &require_int8,
                   Algorithm &algorithm) {
    socket_path = "/tmp/open-frameflow.sock";
    gpu_id = 0;
    inference_scale = 1.f;
    require_int8 = false;
    algorithm = Algorithm::Rife;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--algorithm") == 0 && i + 1 < argc) {
            const char *value = argv[++i];
            if (std::strcmp(value, "rife") == 0) {
                algorithm = Algorithm::Rife;
            } else if (std::strcmp(value, "dis") == 0) {
                algorithm = Algorithm::Dis;
            } else {
                return false;
            }
        } else if (std::strcmp(argv[i], "--socket") == 0 && i + 1 < argc) {
            socket_path = argv[++i];
        } else if (std::strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (std::strcmp(argv[i], "--gpu") == 0 && i + 1 < argc) {
            char *end = nullptr;
            const long parsed = std::strtol(argv[++i], &end, 10);
            if (!end || *end || parsed < 0 || parsed > INT32_MAX) return false;
            gpu_id = static_cast<int>(parsed);
        } else if (std::strcmp(argv[i], "--inference-scale") == 0 &&
                   i + 1 < argc) {
            char *end = nullptr;
            inference_scale = std::strtof(argv[++i], &end);
            if (!end || *end || !std::isfinite(inference_scale) ||
                inference_scale < 0.25f || inference_scale > 1.f)
                return false;
        } else if (std::strcmp(argv[i], "--require-int8") == 0) {
            require_int8 = true;
        } else {
            return false;
        }
    }
    sockaddr_un address {};
    if (socket_path.empty() || socket_path.size() >= sizeof(address.sun_path))
        return false;
    if (algorithm == Algorithm::Rife && model_path.empty()) return false;
    if (algorithm == Algorithm::Dis &&
        (require_int8 || inference_scale != 1.f))
        return false;
    return true;
}

bool model_has_int8_convolution(const std::filesystem::path &model_path) {
    std::ifstream param(model_path / "flownet.param");
    if (!param) return false;

    std::string line;
    while (std::getline(param, line)) {
        std::istringstream fields(line);
        std::string type;
        std::string name;
        int bottom_count = 0;
        int top_count = 0;
        if (!(fields >> type >> name >> bottom_count >> top_count) ||
            (type != "Convolution" && type != "ConvolutionDepthWise"))
            continue;

        std::string token;
        for (int i = 0; i < bottom_count + top_count; ++i) {
            if (!(fields >> token)) break;
        }
        while (fields >> token) {
            const size_t separator = token.find('=');
            if (separator == std::string::npos ||
                token.compare(0, separator, "8") != 0)
                continue;
            char *end = nullptr;
            const long scale_term =
                std::strtol(token.c_str() + separator + 1, &end, 10);
            if (end && *end == '\0' && scale_term > 0) return true;
        }
    }
    return false;
}

} // namespace

int main(int argc, char **argv) {
    std::string socket_path;
    std::filesystem::path model_path;
    int gpu_id;
    float inference_scale;
    bool require_int8;
    Algorithm algorithm;
    if (!parse_options(argc, argv, socket_path, model_path, gpu_id,
                       inference_scale, require_int8, algorithm)) {
        std::fprintf(stderr,
            "Uso: %s [--algorithm rife|dis] [--model DIRETORIO_RIFE] [--gpu ID] "
            "[--inference-scale 0.25..1.0] [--require-int8] "
            "[--socket CAMINHO]\n",
            argv[0]);
        return 2;
    }

    if (algorithm == Algorithm::Rife && require_int8 &&
        !model_has_int8_convolution(model_path)) {
        std::fprintf(stderr,
            "[frameflow] modelo nao parece quantizado em INT8 "
            "(nenhuma convolucao com int8_scale_term em %s/flownet.param).\n",
            model_path.c_str());
        return 2;
    }

    std::unique_ptr<RIFE> rife;
    const bool use_vulkan = algorithm == Algorithm::Rife;
    if (use_vulkan) {
        ncnn::create_gpu_instance();
        const int gpu_count = ncnn::get_gpu_count();
        for (int i = 0; i < gpu_count; ++i) {
            std::printf("[frameflow] Vulkan GPU %d: %s\n",
                        i, ncnn::get_gpu_info(i).device_name());
        }
        if (gpu_id >= gpu_count) {
            std::fprintf(stderr,
                "[frameflow] GPU %d indisponivel (detectadas: %d).\n",
                gpu_id, gpu_count);
            ncnn::destroy_gpu_instance();
            return 1;
        }

        rife = std::make_unique<RIFE>(
            gpu_id, false, false, false, 1, false, true);
        if (rife->load(model_path) != 0) {
            std::fprintf(stderr, "[frameflow] falha ao carregar modelo: %s\n",
                         model_path.c_str());
            ncnn::destroy_gpu_instance();
            return 1;
        }
    }
    std::unique_ptr<OpticalFlowInterpolator> optical_flow;
    if (!use_vulkan) {
        try {
            optical_flow = std::make_unique<OpticalFlowInterpolator>();
        } catch (const cv::Exception &error) {
            std::fprintf(stderr, "[frameflow] nao foi possivel iniciar DIS: %s\n",
                         error.what());
            return 1;
        }
    }

    struct sigaction action {};
    action.sa_handler = request_stop;
    sigemptyset(&action.sa_mask);
    if (sigaction(SIGINT, &action, nullptr) != 0 ||
        sigaction(SIGTERM, &action, nullptr) != 0) {
        std::perror("[frameflow] sigaction");
        if (use_vulkan) ncnn::destroy_gpu_instance();
        return 1;
    }

    const int server = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (server < 0) {
        std::perror("[frameflow] socket");
        if (use_vulkan) ncnn::destroy_gpu_instance();
        return 1;
    }

    sockaddr_un address {};
    address.sun_family = AF_UNIX;
    std::memcpy(address.sun_path, socket_path.c_str(), socket_path.size() + 1);
    if (bind(server, reinterpret_cast<sockaddr *>(&address), sizeof(address)) != 0) {
        std::perror("[frameflow] bind");
        close(server);
        if (use_vulkan) ncnn::destroy_gpu_instance();
        return 1;
    }
    if (chmod(socket_path.c_str(), S_IRUSR | S_IWUSR) != 0 || listen(server, 1) != 0) {
        std::perror("[frameflow] chmod/listen");
        close(server);
        unlink(socket_path.c_str());
        if (use_vulkan) ncnn::destroy_gpu_instance();
        return 1;
    }

    if (use_vulkan) {
        std::printf("[frameflow] RIFE v4 pronto; Vulkan GPU %d (%s); "
                    "escala de inferencia %.2fx; socket %s\n",
                    gpu_id, ncnn::get_gpu_info(gpu_id).device_name(),
                    inference_scale, socket_path.c_str());
        if (require_int8)
            std::printf("[frameflow] modelo INT8 verificado em %s\n",
                        model_path.c_str());
    } else {
        std::printf("[frameflow] Optical Flow DIS (CPU, preset fast) pronto; "
                    "socket %s\n", socket_path.c_str());
    }
    while (!stop_requested) {
        const int client = accept4(server, nullptr, nullptr, SOCK_CLOEXEC);
        if (client < 0) {
            if (errno == EINTR) continue;
            std::perror("[frameflow] accept");
            break;
        }
        if (!handle_client(client, rife.get(),
                           optical_flow.get(),
                           algorithm, inference_scale))
            std::fprintf(stderr, "[frameflow] cliente desconectado ou requisicao invalida.\n");
        close(client);
    }

    close(server);
    unlink(socket_path.c_str());
    if (use_vulkan) ncnn::destroy_gpu_instance();
    return 0;
}