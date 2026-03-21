from huggingface_hub import snapshot_download

# download qwen 3 0.6 emebedding model

snapshot_download(
    repo_id="Qwen/Qwen3-Embedding-0.6B", local_dir="/models/qwen3_0_6b_embedding"
)

# download blip2 model for image captioning
#
snapshot_download(
    repo_id="Salesforce/blip-image-captioning-base",
    local_dir="/models/blip_image_captioning_base",
)
