# Open-FrameFlow

Frame-interpolation service with RIFE/ncnn/Vulkan and DIS optical flow on the
CPU. Open-FrameFlow runs as an independent process; it does not link to or embed
Open-Scaling. Open-Scaling connects only when frame generation is explicitly
enabled and otherwise remains fully usable on its own.

## Build

Open-FrameFlow uses the separately built `rife-ncnn-vulkan` library and model
files for RIFE mode, plus OpenCV core, imgproc, and video modules for DIS mode.
Point CMake at the RIFE checkout; no RIFE sources are copied into this
repository.

```sh
cmake -S . -B build \
  -DRIFE_ROOT=/path/to/rife-ncnn-vulkan \
  -DNCNN_ROOT=/path/to/ncnn-install
cmake --build build -j
```

Install `libopencv-dev` and the Vulkan development files before configuring.
Select RIFE with a model directory:

```sh
./build/open-frameflow \
  --algorithm rife \
  --model /path/to/rife-ncnn-vulkan/models/rife-v4 \
  --gpu 0 \
  --inference-scale 1.0 \
  --socket /tmp/open-frameflow.sock
```

`--gpu` seleciona o índice Vulkan usado pelo ncnn/RIFE; o serviço imprime os
índices e nomes detectados ao iniciar. Para selecionar o dispositivo de índice
1, use `--gpu 1`. Esse índice é independente de `--vk-gpu` do Open-Scaling,
que seleciona a GPU usada para renderizar/apresentar a imagem.

`--inference-scale` controla a resolução usada internamente pelo RIFE e aceita
valores de `0.25` a `1.0` (padrão `1.0`). Em `0.5`, cada dimensão de entrada é
reduzida à metade, processando aproximadamente um quarto dos pixels; os quadros
gerados são redimensionados de volta para a resolução original. Valores menores
reduzem a carga da GPU, com perda de detalhe.

## Optical Flow DIS

To generate interpolated frames with OpenCV's fast DIS optical-flow preset on
the CPU, no RIFE model or Vulkan GPU is needed:

```sh
./build/open-frameflow \
  --algorithm dis \
  --socket /tmp/open-frameflow.sock
```

DIS estimates dense motion in both directions once for each input pair and
reuses the flows to make the requested intermediate frames with backward
warping. Forward/backward consistency weights reduce blending in uncertain
regions, but disocclusions, fast motion, and newly revealed details can still
produce artifacts. This mode is intended to reduce frame-generation GPU work;
it shifts computation to the CPU and does not guarantee a real-time frame rate.
The input resolution is used as-is. `--inference-scale` and `--require-int8`
apply only to RIFE.

## Modelos quantizados INT8

O serviço habilita o suporte a inferência INT8 do ncnn e carrega os pesos do
diretório passado em `--model`. Para exigir um modelo quantizado, passe
`--require-int8`; o serviço rejeita o modelo se não encontrar escalas INT8 nas
camadas convolucionais do `flownet.param`. Os pesos RIFE v4 padrão não são
quantizados, portanto não serão aceitos com essa opção.

Para produzir um modelo INT8, use as ferramentas de quantização pós-treino do
ncnn (`ncnn2table` e `ncnn2int8`) com dados de calibração representativos dos
frames capturados e dos timesteps. RIFE v4 usa múltiplas entradas e a camada
customizada `rife.Warp`; a ferramenta de calibração precisa registrar essa
camada e usar tensores de entrada compatíveis com o pré-processamento do RIFE.
Mantenha os arquivos `.param` e `.bin` convertidos em um diretório separado,
sem sobrescrever os pesos originais. A quantização reduz a precisão e pode
introduzir artefatos; o ganho de desempenho depende do suporte INT8 do Vulkan.

The service accepts one local client at a time. The socket is created with
user-only permissions. It accepts RGB24 frame pairs and factors 2x, 3x, or 4x;
each response contains the intermediate frames at evenly spaced RIFE
timesteps. The wire protocol is versioned and duplicated in the two projects
so neither project needs the other's headers or libraries to build.

## Open-Scaling

Build Open-Scaling separately. The text and graphical launchers can start this
service using a selected RIFE model folder and Vulkan GPU; direct CLI use still
requires the service to be started separately:

```sh
./build/open-scaling fsr 0xWINDOW_ID quality --framegen 2
```

The launchers can start either RIFE or CPU DIS; direct CLI use requires starting
the service separately with the desired `--algorithm`.
Use `--framegen 3` or `--framegen 4` for more intermediate frames, and
`--frameflow-socket PATH` to override the default socket. If the service is
unavailable or disconnects, Open-Scaling logs the condition and continues
with its regular FSR pipeline.

Higher factors require more output-frame warping in DIS mode, or additional
serial RIFE inferences per input pair, and can increase processing time.
Open-Scaling targets up to 60 times the selected factor,
capped by the monitor refresh rate; actual throughput depends on inference
time. The service returns RGB frames through a bounded local IPC queue; it is
an initial modular integration, not a zero-copy Vulkan path.
