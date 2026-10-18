# Realtime Ditto avatar

LiveTalking uses Ditto as a persistent avatar backend with utterance-level generation:

1. LiveTalking starts one Ditto worker process and waits until its model is loaded.
2. The worker keeps model weights loaded but skips inference on silence to reduce idle GPU power; LiveTalking shows the source image while idle.
3. LiveTalking sends each complete 16 kHz mono TTS utterance to that worker.
4. Ditto flushes the final audio context and passes a completion barrier through every render stage before LiveTalking starts playback. The complete generated video and corresponding audio are then sent together through the existing WebRTC, RTMP, or virtual-camera output.

Generated frames are kept as compressed JPEGs in memory until their response is ready; no generated MP4 is required. Edge TTS still decodes progressively, but generated audio is held and not played until Ditto has finished rendering the matching video. This adds response latency in exchange for avoiding speech being played ahead of its video.

## Setup

Download Ditto's checkpoints as described in [Ditto's README](../../ditto-talkinghead/README.md). The checked-in ONNX models are used by default. Install LiveTalking and Ditto dependencies in an environment that can run Ditto; set `ditto_python` to that environment's Python executable if it differs from LiveTalking's Python. FFmpeg must be available on `PATH` for progressive Edge TTS decoding. Install a GPU-enabled ONNX Runtime provider compatible with the GPU/driver (`onnxruntime-gpu` for NVIDIA CUDA, or Ditto's documented OpenVINO provider for Intel); otherwise the runtime may fall back to CPU.

In `config.yaml`, set:

```yaml
model: ditto
ditto_python: 'C:\path\to\ditto\python.exe'
ditto_data_root: .\checkpoints\ditto_onnx
ditto_cfg_pkl: .\checkpoints\ditto_cfg\v0.4_hubert_cfg_trt_online.pkl
ditto_source_path: ..\ditto-talkinghead\example\idle.mp4
ditto_idle_video_path: ..\ditto-talkinghead\example\idle.mp4
ditto_patch_region: lower_face
ditto_sampling_timesteps: 15
ditto_prebuffer_frames: 25
ditto_cpu_threads: 4
ditto_max_size: 1024
```

Use the same idle footage for `ditto_source_path` and `ditto_idle_video_path`. Ditto registers the video frames as source footage, and its existing PutBack stage composites the generated lower-face patch onto those frames. Set `ditto_patch_region: face` to restore Ditto's wider face mask. The silent idle player separately caches sampled frames in JPEG form and loops them without running inference. Paths for `ditto_data_root` and `ditto_cfg_pkl` are relative to `ditto_root`; the source and idle-video paths are relative to the LiveTalking working directory. Start LiveTalking from its repository directory:

```powershell
python app.py
```

The Ditto worker loads the online pipeline at startup; startup can take several minutes while models and all source video frames are registered. Inference pauses on silence but the model stays resident for the next response. Ditto's ONNX runtime prefers an available GPU execution provider; the ONNX LMDM model uses CPU by default in this Ditto revision because its expression output collapsed on the tested OpenVINO GPU path. Other ONNX components remain GPU-first. For the highest throughput, use compatible Ditto TensorRT engines by setting `ditto_data_root` to the downloaded TensorRT directory; Ditto's online config remains `v0.4_hubert_cfg_trt_online.pkl`.

## Current scope

The current Ditto SDK has mutable per-avatar streaming state. This integration therefore limits LiveTalking to **one active Ditto session** and one configured source image/video. Other avatar backends retain their existing concurrency behavior. A future multi-session implementation should share model instances while isolating per-session avatar and audio state.

The lower-face mask limits where generated pixels are composited; it does not reduce neural-network inference, which still generates the full aligned face crop. `ditto_sampling_timesteps` controls the LMDM diffusion iterations (default 15 instead of Ditto's 50-step default); fewer steps can improve throughput, with a possible reduction in motion quality. Since playback waits for the whole utterance, expect the first audio to arrive only after TTS and Ditto generation are complete; at the measured 3–10 generated FPS, longer utterances may take substantially longer than their spoken duration to prepare. Compressed generated frames remain buffered in memory until playback. During generation, the Ditto writer progress bar shows rendered frames against an estimate based on the utterance audio duration, rounded to the model's five-frame generation step; after the render pipeline drains, the total is corrected to the exact rendered frame count. The 100-frame throughput log retains its frame count and elapsed time and also reports writer progress. Reduce `ditto_max_size` or the sampling steps to lower inference load; CPU threads can be tuned with `ditto_cpu_threads`.
