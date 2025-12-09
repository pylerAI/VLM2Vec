import logging
import torch.distributed as dist
import numpy as np
import os
import json
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

# Example usage:
# torchrun --nproc_per_node=4 generate_embed_parquet.py \
# --artifact_path /gpfs/public/artifacts/aid-global/v0.1.0/train/all \
# --save_dir /gpfs/public/artifacts/aid-global/v0.1.0/train/all/embeds/vlm2vec_2b_raw \
# --label_path /gpfs/public/artifacts/aid-global/v0.1.0/train/all/train_ground_truth_gpt_5_mini.json \
# --chunk_size 50000 \
# --metadata_type raw


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


CATEGORICAL_QUERY_TEMPLATE = """
Based on the video thumbnail and the description below, explain how {category} elements are visually or contextually represented in this video. Description: {summary}
"""

GENERAL_QUERY_TEMPLATE = """
Represent the given video thumbnail with the following metadata: {summary}
"""

METADATA_TYPES = ["general", "sexual", "violence", "hate", "politics", "controversy"]

class YCMDataset(Dataset):
    def __init__(self,
                 parquet_path: str,
                 image_dir: str,
                 processor: AutoProcessor,
                 max_size: int = 640,
                 video_ids: Optional[List[str]] = None,
                 metadata_type: str = "general",
                 uuid_to_vid: Optional[Dict[str, str]] = None):
        super().__init__()
        self.parquet_path = parquet_path
        self.image_dir = image_dir
        self.processor = processor
        self.max_size = max_size
        self.metadata_type = metadata_type
        assert self.metadata_type in METADATA_TYPES, "Invalid metadata type"

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

        if "en_summary" in data.keys():
            metadata_text = data['en_summary']
        else:
            metadata_text = data['summary']

        if self.metadata_type == "general":
            text = GENERAL_QUERY_TEMPLATE.format(summary=metadata_text)
        else:
            text = CATEGORICAL_QUERY_TEMPLATE.format(category=self.metadata_type, summary=metadata_text)

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
    parser.add_argument("--label_path", type=str, default="/gpfs/public/artifacts/aid-global/v0.1.0/test/annotation_parquets/gt_v0.3.0.parquet", help="Path to the label file")
    parser.add_argument("--chunk_size", type=int, default=40000, help="Chunk size for processing")
    parser.add_argument("--metadata_type", type=str, default="general", help="general or category name (sexual, violence, hate, politics, controversy)")
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

    df = pd.read_parquet(args.label_path)
    whole_video_ids = df["video_id"].tolist()
    print(f"Total video ids: {len(whole_video_ids)}")

    # split the whole_video_ids into chunks
    os.makedirs(f"{args.save_dir}", exist_ok=True)
    chunk_size = args.chunk_size
    chunked_video_ids = [whole_video_ids[i:i+chunk_size] for i in range(0, len(whole_video_ids), chunk_size)]
    for chunk_idx in range(len(chunked_video_ids)):
        video_id_chunk = chunked_video_ids[chunk_idx]
        # Load dataset
        dataset = YCMDataset(args.parquet_path, args.image_dir, processor, video_ids=video_id_chunk, metadata_type=args.metadata_type)
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, sampler=sampler)

        log.info(f"Processing dataset with chunk {chunk_idx}...")

        all_features = []
        all_video_ids = []

        for batch_idx, batch in enumerate(tqdm(dataloader)):
            video_ids, texts, images = batch

            model_inputs = {
                "text": texts,
                "images": images,
            }
            # Print max length text video id
            # max_len_idx = np.argmax([len(text) for text in texts])
            # print(f"max_len text: {len(texts[max_len_idx])}, video_id: {video_ids[max_len_idx]}")

            inputs = Qwen2_VL_process_fn(model_inputs, processor, model_args)

            # print("max_len: ", inputs['input_ids'].shape)

            inputs = batch_to_device(inputs, device)
            output = model(qry=inputs)["qry_reps"]

            if world_size > 1:
                # Gather outputs from all ranks
                output = output.contiguous()
                dist_outputs = [torch.empty_like(output) for _ in range(world_size)]
                dist.all_gather(dist_outputs, output)
                dist_outputs[rank] = output
                dist_outputs = torch.cat(dist_outputs, dim=0).detach().cpu()

                # Gather video ids from all ranks
                dist_video_ids = [None] * world_size
                dist.all_gather_object(dist_video_ids, video_ids)
                dist_video_ids[rank] = video_ids
                dist_video_ids = [item for sublist in dist_video_ids for item in sublist]
            else:
                dist_outputs = output.detach().cpu()
                dist_video_ids = video_ids

            all_features.append(dist_outputs)
            all_video_ids.extend(dist_video_ids)

        if all_features and rank == 0:
            all_features = torch.cat(all_features, dim=0)

            # Reorder the features based on the label video ids
            id_to_index = {vid: idx for idx, vid in enumerate(all_video_ids)}
            sorted_indices = [id_to_index[vid] for vid in video_id_chunk]
            all_features = torch.stack([all_features[i] for i in sorted_indices], dim=0)
            all_video_ids = [all_video_ids[i] for i in sorted_indices]
            all_features = all_features.float().numpy()
            np.save(f"{args.save_dir}/mm_features_{chunk_idx}.npy", all_features)
            np.save(f"{args.save_dir}/video_ids_{chunk_idx}.npy", all_video_ids)
            log.info(f"Features saved at {args.save_dir}/mm_features_{chunk_idx}.npy")
            log.info(f"Video ids saved at {args.save_dir}/video_ids_{chunk_idx}.npy")
            log.info(f"Total feature shape: {all_features.shape}")
            all_features = []
            all_video_ids = []

    cleanup()