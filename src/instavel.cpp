#include "protocol.h"

#include <rife.h>

#include <arpa/inet.h>
#include <cerrno>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <string>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>
#include <vector>

namespace {

volatile std::sig_atomic_t stop_requested = 0;

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

bool handle_client(int fd, RIFE &rife) {
    const float norm_vals[3] = {1.f / 255.f, 1.f / 255.f, 1.f / 255.f};

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

        // from_pixels devolve Mat float (elemsize=4) com c=3, valores em [0,255].
        ncnn::Mat in0 = ncnn::Mat::from_pixels(
            input0.data(), ncnn::Mat::PIXEL_RGB, static_cast<int>(width),
            static_cast<int>(height));
        ncnn::Mat in1 = ncnn::Mat::from_pixels(
            input1.data(), ncnn::Mat::PIXEL_RGB, static_cast<int>(width),
            static_cast<int>(height));

        // RIFE foi treinado com entradas normalizadas para [0,1].
        in0.substract_mean_normalize(nullptr, norm_vals);
        in1.substract_mean_normalize(nullptr, norm_vals);

        std::vector<unsigned char> outputs(static_cast<size_t>(frame_bytes) *
                                           (factor - 1u));
        uint32_t status = FRAMEFLOW_STATUS_OK;

        for (uint32_t step = 1; step < factor; ++step) {
            unsigned char *output = outputs.data() +
                static_cast<size_t>(step - 1u) * frame_bytes;

            // Deixa o RIFE alocar o Mat de saida no formato que ele produz
            // (float, c=3). NAO passe um buffer RGB24 pre-alocado aqui.
            ncnn::Mat out_mat;
            const float timestep = static_cast<float>(step) / factor;
            const int result = rife.process(in0, in1, timestep, out_mat);
            if (result != 0) {
                std::fprintf(stderr,
                    "[frameflow] RIFE process falhou: %d (seq=%u:%u, t=%.3f).\n",
                    result, sequence_hi, sequence_lo, timestep);
                status = FRAMEFLOW_STATUS_INFERENCE_FAILED;
                break;
            }

            if (out_mat.w != static_cast<int>(width) ||
                out_mat.h != static_cast<int>(height) ||
                out_mat.c != 3 ||
                out_mat.elemsize != 4u) {
                std::fprintf(stderr,
                    "[frameflow] formato inesperado na saida RIFE: "
                    "%dx%dx%d elemsize=%zu (esperado float RGB %ux%u).\n",
                    out_mat.w, out_mat.h, out_mat.c,
                    out_mat.elemsize, width, height);
                status = FRAMEFLOW_STATUS_INFERENCE_FAILED;
                break;
            }

            // Converte float [0,1] -> uint8 [0,255] RGB24 empacotado.
            out_mat.to_pixels(output, ncnn::Mat::PIXEL_RGB);
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
                   std::filesystem::path &model_path, int &gpu_id) {
    socket_path = "/tmp/open-frameflow.sock";
    gpu_id = 0;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--socket") == 0 && i + 1 < argc) {
            socket_path = argv[++i];
        } else if (std::strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (std::strcmp(argv[i], "--gpu") == 0 && i + 1 < argc) {
            char *end = nullptr;
            const long parsed = std::strtol(argv[++i], &end, 10);
            if (!end || *end || parsed < 0 || parsed > INT32_MAX) return false;
            gpu_id = static_cast<int>(parsed);
        } else {
            return false;
        }
    }
    sockaddr_un address {};
    if (model_path.empty() || socket_path.empty() ||
        socket_path.size() >= sizeof(address.sun_path)) return false;
    return true;
}

} // namespace

int main(int argc, char **argv) {
    std::string socket_path;
    std::filesystem::path model_path;
    int gpu_id;
    if (!parse_options(argc, argv, socket_path, model_path, gpu_id)) {
        std::fprintf(stderr,
            "Uso: %s --model DIRETORIO_RIFE [--gpu ID] [--socket CAMINHO]\n",
            argv[0]);
        return 2;
    }

    ncnn::create_gpu_instance();
    if (gpu_id >= ncnn::get_gpu_count()) {
        std::fprintf(stderr, "[frameflow] GPU %d indisponivel (detectadas: %d).\n",
                     gpu_id, ncnn::get_gpu_count());
        ncnn::destroy_gpu_instance();
        return 1;
    }

    RIFE rife(gpu_id, false, false, false, 1, false, true);
    if (rife.load(model_path) != 0) {
        std::fprintf(stderr, "[frameflow] falha ao carregar modelo: %s\n",
                     model_path.c_str());
        ncnn::destroy_gpu_instance();
        return 1;
    }

    struct sigaction action {};
    action.sa_handler = request_stop;
    sigemptyset(&action.sa_mask);
    if (sigaction(SIGINT, &action, nullptr) != 0 ||
        sigaction(SIGTERM, &action, nullptr) != 0) {
        std::perror("[frameflow] sigaction");
        ncnn::destroy_gpu_instance();
        return 1;
    }

    const int server = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (server < 0) {
        std::perror("[frameflow] socket");
        ncnn::destroy_gpu_instance();
        return 1;
    }

    sockaddr_un address {};
    address.sun_family = AF_UNIX;
    std::memcpy(address.sun_path, socket_path.c_str(), socket_path.size() + 1);
    if (bind(server, reinterpret_cast<sockaddr *>(&address), sizeof(address)) != 0) {
        std::perror("[frameflow] bind");
        close(server);
        ncnn::destroy_gpu_instance();
        return 1;
    }
    if (chmod(socket_path.c_str(), S_IRUSR | S_IWUSR) != 0 || listen(server, 1) != 0) {
        std::perror("[frameflow] chmod/listen");
        close(server);
        unlink(socket_path.c_str());
        ncnn::destroy_gpu_instance();
        return 1;
    }

    std::printf("[frameflow] RIFE v4 pronto; GPU %d; socket %s\n",
                gpu_id, socket_path.c_str());
    while (!stop_requested) {
        const int client = accept4(server, nullptr, nullptr, SOCK_CLOEXEC);
        if (client < 0) {
            if (errno == EINTR) continue;
            std::perror("[frameflow] accept");
            break;
        }
        if (!handle_client(client, rife))
            std::fprintf(stderr, "[frameflow] cliente desconectado ou requisicao invalida.\n");
        close(client);
    }

    close(server);
    unlink(socket_path.c_str());
    ncnn::destroy_gpu_instance();
    return 0;
}