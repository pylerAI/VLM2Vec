import logging
import time
import torch.distributed as dist
import numpy as np
import os
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor
import argparse
from tqdm import tqdm
from typing import Optional, List, Dict
from PIL import Image
import pandas as pd

from src.arguments import ModelArguments, DataArguments
from src.model.model import MMEBModel
from src.model.processor import load_processor, Qwen2_VL_process_fn, VLM_IMAGE_TOKENS, QWEN2_VL
from src.utils import batch_to_device

# Example usage (Generate embed for AiD-Global v0.4.0 training set):
# torchrun --nproc_per_node=8 generate_embed_parquet.py \
# --parquet_path /gpfs/public/artifacts/aid-global/v0.4.0/train/annotation_parquets/train_v0.1.6_with_en_summary.parquet \
# --save_dir /gpfs/public/artifacts/aid-global/v0.1.0/train/all/embeds/vlm2vec_2b_raw \
# --checkpoint /gpfs/public/artifacts/aid-global/2026-01-13-deploy/models/VLM2Vec-0113 \
# --merge_chunks

def setup(rank, world_size):
    """Initialize the distributed environment."""
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    """Clean up the distributed environment."""
    dist.destroy_process_group()

def is_rank_zero():
    return not dist.is_initialized() or dist.get_rank() == 0

def get_logger(name=__name__):
    logger = logging.getLogger(name)
    if is_rank_zero():
        logging.basicConfig(level=logging.INFO)
    else:
        logger.setLevel(logging.ERROR)  # or higher
    return logger


# NOTE: Currently use same prompt for query and cand.
QUERY_PROMPT = """
Represent the given video thumbnail with the following metadata: {summary}
"""

CAND_PROMPT = """
Represent the given video thumbnail with the following metadata: {summary}
"""

class YCMDataset(Dataset):
    def __init__(self,
            parquet_path: str,
            image_dir: str,
            processor: AutoProcessor,
            max_size: int = 640,
            video_ids: Optional[List[str]] = None,
            prompt_type: str = "cand",
            use_summary: bool = False,
            uuid_to_vid: Optional[Dict[str, str]] = None):
        super().__init__()
        self.parquet_path = parquet_path
        self.image_dir = image_dir
        self.processor = processor
        self.max_size = max_size
        self.prompt_type = prompt_type
        self.use_summary = use_summary
        assert self.prompt_type in ["cand", "query", "text_only"], "Invalid prompt type"

        self.uuid_to_vid = uuid_to_vid
        self.df = pd.read_parquet(parquet_path)

        if video_ids is not None:
            self.df = self.df[self.df["video_id"].isin(video_ids)]

        self.video_ids = self.df["video_id"].tolist()
        self.df.set_index("video_id", inplace=True)

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, idx):
        video_id = self.video_ids[idx]
        image_path = os.path.join(self.image_dir, f"{video_id}.jpg")
        data = self.df.loc[video_id]

        if "en_summary" in data.keys() and not self.use_summary:
            metadata_text = data["en_summary"]
        elif "summary" in data.keys():
            metadata_text = data["summary"]
        else:
            required_columns = ["title", "description", "tags"]
            missing_columns = [column for column in required_columns if column not in data.keys()]
            if missing_columns:
                raise KeyError(
                    f"Missing required metadata columns for video_id={video_id}: {missing_columns}"
                )

            tags = data["tags"]
            if isinstance(tags, np.ndarray):
                tags = tags.tolist()

            if isinstance(tags, list):
                tags = ", ".join(map(str, tags))

            metadata_text = (
                f"- title: {data['title']}\n"
                f"- description: {data['description']}\n"
                f"- tags: {tags}\n"
            )

        if self.prompt_type == "cand":
            text = CAND_PROMPT.format(summary=metadata_text)
        elif self.prompt_type == "query":
            text = QUERY_PROMPT.format(summary=metadata_text)
        else:
            raise ValueError(f"Invalid prompt type: {self.prompt_type}")

        if self.prompt_type == "text_only":
            image = None
        else:
            # Add Image Token
            text = f"{VLM_IMAGE_TOKENS[QWEN2_VL]}\n{text}"

            image = Image.open(image_path)
            if self.max_size < max(image.size):
                scale = self.max_size / max(image.size)
                image = image.resize((int(image.size[0] * scale), int(image.size[1] * scale)))

        if self.uuid_to_vid:
            video_id = self.uuid_to_vid[video_id]
        return video_id, text, image


def collate_fn(batch):
    video_ids = [item[0] for item in batch]
    texts = [item[1] for item in batch]
    images = [item[2] for item in batch]

    return video_ids, texts, images


def load_model(device, model_args):
    data_args = DataArguments()
    processor = load_processor(model_args, data_args)
    model = MMEBModel.load(model_args)
    model = model.to(device).eval()

    return model, processor


def parse_args():
    parser = argparse.ArgumentParser(description="Generate embeddings")
    parser.add_argument("--parquet_path", type=str, required=True, help="Path to the parquet file")
    parser.add_argument("--image_dir", type=str, required=True, help="Path to the image directory")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save the embeddings")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for processing")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to the checkpoint file")
    parser.add_argument("--label_path", type=str, help="Path to the label file")
    parser.add_argument("--chunk_size", type=int, default=40000, help="Chunk size for processing")
    parser.add_argument("--prompt_type", type=str, default="cand", choices=["cand", "query", "text_only"], help="Prompt type for the model")
    parser.add_argument("--use_summary", action="store_true", help="Use summary instead of en_summary")
    parser.add_argument("--merge_chunks", action="store_true", help="Merge the chunks after processing")
    # Performance optimization arguments
    parser.add_argument("--num_workers", type=int, default=4, help="Number of DataLoader workers per GPU")
    parser.add_argument("--prefetch_factor", type=int, default=2, help="Prefetch factor for DataLoader")
    parser.add_argument("--use_compile", action="store_true", help="Use torch.compile for model optimization")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    setup(rank, world_size)
    log = get_logger(f"Rank [{rank}]")

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    checkpoint_path = args.checkpoint_path if args.checkpoint_path else 'TIGER-Lab/VLM2Vec-Qwen2VL-2B'
    model_args = ModelArguments(
        model_name='Qwen/Qwen2-VL-2B-Instruct',
        checkpoint_path=checkpoint_path,
        pooling='last',
        normalize=True,
        model_backbone='qwen2_vl',
        lora=True
    )

    # Load the model and processor
    model, processor = load_model(device, model_args)
    
    # Optional: torch.compile for additional speedup (PyTorch 2.0+)
    if args.use_compile:
        log.info("Compiling model with torch.compile...")
        model = torch.compile(model, mode="reduce-overhead")

    # Load video id from label file if provided
    if args.label_path:
        df = pd.read_parquet(args.label_path)
    else:
        df = pd.read_parquet(args.parquet_path)
    whole_video_ids = df["video_id"].tolist()
    print(f"Total video ids: {len(whole_video_ids)}")

    # split the whole_video_ids into chunks
    os.makedirs(f"{args.save_dir}", exist_ok=True)
    chunk_size = args.chunk_size
    chunked_video_ids = [whole_video_ids[i:i+chunk_size] for i in range(0, len(whole_video_ids), chunk_size)]
    for chunk_idx in range(len(chunked_video_ids)):
        if os.path.exists(f"{args.save_dir}/mm_features_{chunk_idx}.npy"):
            print(f"Chunk {chunk_idx} already exists, skipping...")
            continue
        video_id_chunk = chunked_video_ids[chunk_idx]
        # Load dataset
        dataset = YCMDataset(
            args.parquet_path,
            args.image_dir,
            processor,
            video_ids=video_id_chunk,
            prompt_type=args.prompt_type,
            use_summary=args.use_summary)
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
        dataloader = DataLoader(
            dataset, 
            batch_size=args.batch_size, 
            shuffle=False, 
            collate_fn=collate_fn, 
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
            persistent_workers=args.num_workers > 0
        )

        log.info(f"Processing dataset with chunk {chunk_idx}...")

        # Local storage for each rank - avoid per-batch all_gather
        local_features = []
        local_video_ids = []

        # Use inference_mode for faster inference without gradient tracking
        with torch.inference_mode():
            for batch_idx, batch in enumerate(tqdm(dataloader, disable=not is_rank_zero())):
                video_ids, texts, images = batch

                model_inputs = {
                    "text": texts,
                    "images": images,
                }
                inputs = Qwen2_VL_process_fn(model_inputs, processor, None)
                inputs = batch_to_device(inputs, device)
                
                # Use autocast for BF16 inference on B200
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    output = model(qry=inputs)["qry_reps"]

                # Store locally without per-batch communication
                local_features.append(output.float().cpu())
                local_video_ids.extend(video_ids)

        # Gather all features at the end (much more efficient than per-batch)
        if world_size > 1:
            # Save each rank's results to disk temporarily
            rank_save_path = f"{args.save_dir}/temp_rank_{rank}_chunk_{chunk_idx}"
            os.makedirs(rank_save_path, exist_ok=True)
            
            local_features_tensor = torch.cat(local_features, dim=0)
            np.save(f"{rank_save_path}/features.npy", local_features_tensor.numpy())
            np.save(f"{rank_save_path}/video_ids.npy", np.array(local_video_ids, dtype=object))
            
            # dist.barrier()
            
            # Create done flag after saving (file-based synchronization)
            done_flag_path = f"{rank_save_path}/done.flag"
            with open(done_flag_path, 'w') as f:
                f.write('done')
            
            # Only rank 0 merges all results
            if rank == 0:
                # Wait for all ranks to finish saving (polling-based)
                for r in range(world_size):
                    flag_path = f"{args.save_dir}/temp_rank_{r}_chunk_{chunk_idx}/done.flag"
                    while not os.path.exists(flag_path):
                        time.sleep(0.1)
                
                all_features = []
                all_video_ids = []
                for r in range(world_size):
                    rank_path = f"{args.save_dir}/temp_rank_{r}_chunk_{chunk_idx}"
                    all_features.append(np.load(f"{rank_path}/features.npy"))
                    all_video_ids.extend(np.load(f"{rank_path}/video_ids.npy", allow_pickle=True).tolist())
                    # Cleanup temp files
                    os.remove(f"{rank_path}/features.npy")
                    os.remove(f"{rank_path}/video_ids.npy")
                    os.remove(f"{rank_path}/done.flag")
                    os.rmdir(rank_path)
                
                all_features = np.concatenate(all_features, axis=0)
        else:
            all_features = torch.cat(local_features, dim=0).numpy()
            all_video_ids = local_video_ids

        if rank == 0:
            # Reorder the features based on the label video ids
            id_to_index = {vid: idx for idx, vid in enumerate(all_video_ids)}
            sorted_indices = [id_to_index[vid] for vid in video_id_chunk]
            all_features = np.stack([all_features[i] for i in sorted_indices], axis=0)
            all_video_ids = [all_video_ids[i] for i in sorted_indices]
            
            np.save(f"{args.save_dir}/mm_features_{chunk_idx}.npy", all_features)
            np.save(f"{args.save_dir}/video_ids_{chunk_idx}.npy", all_video_ids)
            log.info(f"Features saved at {args.save_dir}/mm_features_{chunk_idx}.npy")
            log.info(f"Video ids saved at {args.save_dir}/video_ids_{chunk_idx}.npy")
            log.info(f"Total feature shape: {all_features.shape}")
        
        # Clear memory
        del local_features, local_video_ids
        if rank == 0:
            del all_features, all_video_ids
        torch.cuda.empty_cache()

    if args.merge_chunks and rank == 0:
        if len(chunked_video_ids) == 1:
            log.info(f"Only one chunk, skipping merge")
        else:
            log.info(f"Merging chunks...")
            all_features = np.concatenate([np.load(f"{args.save_dir}/mm_features_{i}.npy") for i in range(len(chunked_video_ids))], axis=0)
            all_video_ids = np.concatenate([np.load(f"{args.save_dir}/video_ids_{i}.npy") for i in range(len(chunked_video_ids))], axis=0)
            # Remove original chunks
            for i in range(len(chunked_video_ids)):
                os.remove(f"{args.save_dir}/mm_features_{i}.npy")
                os.remove(f"{args.save_dir}/video_ids_{i}.npy")
            np.save(f"{args.save_dir}/mm_features_0.npy", all_features)
            np.save(f"{args.save_dir}/video_ids_0.npy", all_video_ids)
            log.info(f"Features saved at {args.save_dir}/mm_features_0.npy")
            log.info(f"Video ids saved at {args.save_dir}/video_ids_0.npy")
            log.info(f"Total feature shape: {all_features.shape}")

    cleanup()
