# Stage 1: Base image with common dependencies
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 as base

# Prevents prompts from packages asking for user input during installation
ENV DEBIAN_FRONTEND=noninteractive
# Prefer binary wheels over source distributions for faster pip installations
ENV PIP_PREFER_BINARY=1
# Ensures output from python is printed immediately to the terminal without buffering
ENV PYTHONUNBUFFERED=1 

# Install Python, git and other necessary tools
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget

# Clean up to reduce image size
RUN apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# Clone ComfyUI repository
RUN git clone https://github.com/comfyanonymous/ComfyUI.git /comfyui

# Change working directory to ComfyUI
WORKDIR /comfyui

# Install ComfyUI dependencies
RUN pip3 install --upgrade --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 \
    && pip3 install --upgrade -r requirements.txt

# Install runpod
RUN pip3 install runpod requests

# Support for the network volume
ADD src/extra_model_paths.yaml ./

# Go back to the root
WORKDIR /

# Add the start and the handler
ADD src/start.sh src/rp_handler.py test_input.json ./
RUN chmod +x /start.sh

# Stage 2: Download models
FROM base as downloader

ARG HUGGINGFACE
ARG MODEL_TYPE

# Change working directory to ComfyUI
WORKDIR /comfyui

# Download checkpoints/vae/LoRA to include in image based on model type
RUN if [ "$MODEL_TYPE" = "sdxl" ]; then \
      wget -O models/checkpoints/sd_xl_base_1.0.safetensors https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors && \
      wget -O models/vae/sdxl_vae.safetensors https://huggingface.co/stabilityai/sdxl-vae/resolve/main/sdxl_vae.safetensors && \
      wget -O models/vae/sdxl-vae-fp16-fix.safetensors https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/resolve/main/sdxl_vae.safetensors; \
    elif [ "$MODEL_TYPE" = "sd3" ]; then \
      wget --header="Authorization: Bearer ${HUGGINGFACE}" -O models/checkpoints/sd3_medium_incl_clips_t5xxlfp8.safetensors https://huggingface.co/stabilityai/stable-diffusion-3-medium/resolve/main/sd3_medium_incl_clips_t5xxlfp8.safetensors; \
    elif [ "$MODEL_TYPE" = "flux1-schnell" ]; then \
      wget -O models/unet/flux1-schnell.safetensors https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/flux1-schnell.safetensors && \
      wget -O models/clip/clip_l.safetensors https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors && \
      wget -O models/clip/t5xxl_fp8_e4m3fn.safetensors https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn.safetensors && \
      wget -O models/vae/ae.safetensors https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/ae.safetensors; \
    elif [ "$MODEL_TYPE" = "flux1-dev" ]; then \
      wget --header="Authorization: Bearer ${HUGGINGFACE}" -O models/unet/flux1-dev.safetensors https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/flux1-dev.safetensors && \
      wget -O models/clip/clip_l.safetensors https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors && \
      wget -O models/clip/t5xxl_fp8_e4m3fn.safetensors https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn.safetensors && \
      wget --header="Authorization: Bearer ${HUGGINGFACE}" -O models/vae/ae.safetensors https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/ae.safetensors; \
    fi

# Clone IDM-VTON into the custom_nodes directory
RUN git clone https://github.com/TemryL/ComfyUI-IDM-VTON.git custom_nodes/ComfyUI-IDM-VTON

# Navigate to the IDM-VTON directory and install dependencies
WORKDIR /comfyui/custom_nodes/ComfyUI-IDM-VTON
RUN python3 install.py

# Clone comfyui_controlnet_aux
WORKDIR /comfyui/custom_nodes
RUN git clone https://github.com/Fannovel16/comfyui_controlnet_aux/

# Install requirements
WORKDIR /comfyui/custom_nodes/comfyui_controlnet_aux
RUN pip3 install --no-cache-dir -r requirements.txt

# Set appropriate permissions
RUN chmod -R 755 /comfyui/custom_nodes/comfyui_controlnet_aux


# Download SAM model
RUN mkdir -p /comfyui/models/sams && \
    wget -O /comfyui/models/sams/sam_vit_l_0b3195.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth

# Download GroundingDINO config and model
RUN mkdir -p /comfyui/models/grounding-dino && \
    wget -O /comfyui/models/grounding-dino/GroundingDINO_SwinB.cfg.py https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/GroundingDINO_SwinB.cfg.py && \
    wget -O /comfyui/models/grounding-dino/groundingdino_swinb_cogcoor.pth https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swinb_cogcoor.pth


# Clone comfyui-mixlab-nodes
WORKDIR /comfyui/custom_nodes
RUN git clone https://github.com/shadowcz007/comfyui-mixlab-nodes.git

# Install requirements
WORKDIR /comfyui/custom_nodes/comfyui-mixlab-nodes
RUN pip3 install --no-cache-dir -r requirements.txt

# Set appropriate permissions
RUN chmod -R 755 /comfyui/custom_nodes/comfyui-mixlab-nodes

# # Download bert-base-uncased model (optional)
# RUN mkdir -p /comfyui/models/bert-base-uncased && \
#     wget -O /comfyui/models/bert-base-uncased/config.json https://huggingface.co/google-bert/bert-base-uncased/resolve/main/config.json && \
#     wget -O /comfyui/models/bert-base-uncased/model.safetensors https://huggingface.co/google-bert/bert-base-uncased/resolve/main/model.safetensors && \
#     wget -O /comfyui/models/bert-base-uncased/tokenizer_config.json https://huggingface.co/google-bert/bert-base-uncased/resolve/main/tokenizer_config.json && \
#     wget -O /comfyui/models/bert-base-uncased/tokenizer.json https://huggingface.co/google-bert/bert-base-uncased/resolve/main/tokenizer.json && \
#     wget -O /comfyui/models/bert-base-uncased/vocab.txt https://huggingface.co/google-bert/bert-base-uncased/resolve/main/vocab.txt

# Stage 3: Final image
FROM base as final

# Copy models from stage 2 to the final image
COPY --from=downloader /comfyui/models /comfyui/models
COPY --from=downloader /comfyui/custom_nodes /comfyui/custom_nodes


# Start the container
CMD /start.sh