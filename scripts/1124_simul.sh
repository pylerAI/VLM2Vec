torchrun --nproc_per_node=8 my_files/VLM2Vec/generate_embed_parquet.py \
    --parquet_path /gpfs/public/artifacts/aid-global/1124_simulation/annotation_parquets/text_snippet_with_summary_truncated.parquet \
    --image_dir /gpfs/public/artifacts/aid-global/1124_simulation/image \
    --save_dir /gpfs/public/artifacts/aid-global/1124_simulation/embeds/vlm2vec_1124_categorical/general \
    --checkpoint_path /gpfs/public/artifacts/models/PylerAI/aid-global/experimental/VLM2Vec/1124_categorical \
    --label_path /gpfs/public/artifacts/aid-global/1124_simulation/annotation_parquets/text_snippet_with_summary_truncated.parquet \
    --batch_size 8 \
    --metadata_type general