# Realtime Ditto avatar

LiveTalking can use Ditto as a persistent, streaming avatar backend:

1. LiveTalking starts one Ditto worker process and waits until its model is loaded.
2. The worker stays warm and drives Ditto with silence between utterances, producing idle motion.
3. LiveTalking sends 16 kHz mono PCM packets to that worker as they are played.
4. Ditto's online HuBERT pipeline returns frames continuously; LiveTalking sends the frames and matching audio through its existing WebRTC, RTMP, or virtual-camera output.

No generated MP4 is used in this path. The default Edge TTS backend is also decoded progressively for Ditto, rather than waiting for the full utterance to finish synthesizing.

## Setup

Download Ditto's checkpoints as described in [Ditto's README](../../ditto-talkinghead/README.md). The checked-in ONNX models are used by default. Install LiveTalking and Ditto dependencies in an environment that can run Ditto; set `ditto_python` to that environment's Python executable if it differs from LiveTalking's Python. FFmpeg must be available on `PATH` for progressive Edge TTS decoding. Install a GPU-enabled ONNX Runtime provider compatible with the GPU/driver (`onnxruntime-gpu` for NVIDIA CUDA, or Ditto's documented OpenVINO provider for Intel); otherwise the runtime may fall back to CPU.

In `config.yaml`, set:

```yaml
model: ditto
ditto_python: 'C:\path\to\ditto\python.exe'
ditto_data_root: .\checkpoints\ditto_onnx
ditto_cfg_pkl: .\checkpoints\ditto_cfg\v0.4_hubert_cfg_trt_online.pkl
ditto_source_path: ..\ditto-talkinghead\example\image.png
ditto_cpu_threads: 8
```

Paths for `ditto_data_root` and `ditto_cfg_pkl` are relative to `ditto_root`. `ditto_source_path` is relative to the LiveTalking working directory. Start LiveTalking from its repository directory:

```powershell
python app.py
```

The Ditto worker loads the online pipeline at startup and processes silence to stay warm. Startup can take several minutes while models and the source image are loaded. Ditto's ONNX runtime prefers an available GPU execution provider; the ONNX LMDM model uses CPU by default in this Ditto revision because its expression output collapsed on the tested OpenVINO GPU path. Other ONNX components remain GPU-first. For the highest throughput, use compatible Ditto TensorRT engines by setting `ditto_data_root` to the downloaded TensorRT directory; Ditto's online config remains `v0.4_hubert_cfg_trt_online.pkl`.

## Current scope

The current Ditto SDK has mutable per-avatar streaming state. This integration therefore limits LiveTalking to **one active Ditto session** and one configured source image/video. Other avatar backends retain their existing concurrency behavior. A future multi-session implementation should share model instances while isolating per-session avatar and audio state.

Ditto's online audio context and model inference impose some startup delay before the first speech-driven frame. End-to-end latency also depends on TTS generation, GPU throughput, and output transport; verify sustained output near 25 video fps on the target GPU.
