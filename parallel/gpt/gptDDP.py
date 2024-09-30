import os ,pandas as pd , torch , argparse
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.nn.functional import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.utils.data import DataLoader

from transformers import GPT2Tokenizer , GPT2LMHeadModel

dist.init_process_group("nccl")
local_rank = int(os.environ['LOCAL_RANK'])
global_rank = int(os.environ['RANK'])

EPOCHS = 12
BATCH_SIZE = 64 // int(os.environ["WORLD_SIZE"])
WORKERS = 48



def main():
    print("\n\n\n\nIniciando treinamento do modelo GPT2 com DDP \n\n\n\n\n")

    parser = argparse.ArgumentParser("GPT2 com DDP ")

