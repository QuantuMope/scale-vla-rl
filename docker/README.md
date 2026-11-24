# Docker

Build the image from the repository root. The image provides CUDA 12.8.2 on Ubuntu 22.04 and a Python 3.11 virtual environment at `/app/venv`. The setup script installs PyTorch 2.7.1 and the other dependencies from `uv.lock`.

```bash
docker build -f docker/Dockerfile -t scale_vla_rl:cuda12.8.2-torch2.7.1-py3.11-ubuntu22.04 .
```

Run the following from a writable Git checkout. Export `EMBODIED_GEN_SCENES`, `MS_ASSET_DIR`, and `TRANSFORMERS_CACHE` and prepare the assets as described in the [main README](../README.md#setup).

The container runs `scripts/setup.sh` to initialize submodules, install the locked dependencies into `/app/venv`, and build `kincpp` using a container-local CMake build directory. The smoke run saves a checkpoint in `runs/docker-smoke`.

```bash
mkdir -p runs/docker-smoke
docker run --rm --gpus all --shm-size=4g \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp \
  -e EMBODIED_GEN_SCENES=/assets/scenes \
  -e MS_ASSET_DIR=/assets/maniskill \
  -e TRANSFORMERS_CACHE=/assets/model-cache \
  -e HF_HUB_OFFLINE=1 \
  -v "$PWD:/app/scale-vla-rl" \
  -w /app/scale-vla-rl \
  -v "$EMBODIED_GEN_SCENES:/assets/scenes:ro" \
  -v "$MS_ASSET_DIR/data:/assets/maniskill/data:ro" \
  -v "$TRANSFORMERS_CACHE:/assets/model-cache:ro" \
  -v "$PWD/runs/docker-smoke:/runs/smoke" \
  scale_vla_rl:cuda12.8.2-torch2.7.1-py3.11-ubuntu22.04 bash -euc '
    KINCPP_BUILD_DIR=/tmp/kincpp-build ./scripts/setup.sh
    python -m alf.bin.train \
      --root_dir=/runs/smoke \
      --conf=src/confs/embodied_gen_ppo_flow.py \
      --conf_param=DEBUG=True \
      --conf_param=_USER.max_num_scenes=1 \
      --conf_param=TrainerConfig.num_iterations=1
  '
```
