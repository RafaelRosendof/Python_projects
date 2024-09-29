import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os


def ddp_setup():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    init_process_group(backend="nccl")


class Trainer:
    def __init__(self, model, train_data, optimizer, save_every, snapshot_path):
        self.gpu_id = int(os.environ["LOCAL_RANK"])
        self.model = model.to(self.gpu_id)
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.epochs_run = 0
        self.snapshot_path = snapshot_path
        if os.path.exists(snapshot_path):
            print("Loading snapshot")
            self._load_snapshot(snapshot_path)

        self.model = DDP(self.model, device_ids=[self.gpu_id])

    def _load_snapshot(self, snapshot_path):
        loc = f"cuda:{self.gpu_id}"
        snapshot = torch.load(snapshot_path, map_location=loc)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()
        outputs = self.model(**source)
        loss = F.cross_entropy(outputs.logits, targets)
        loss.backward()
        self.optimizer.step()

    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        self.train_data.sampler.set_epoch(epoch)
        for batch in self.train_data:
            source = {k: v.to(self.gpu_id) for k, v in batch.items() if k != "labels"}
            targets = batch["labels"].to(self.gpu_id)
            self._run_batch(source, targets)

    def _save_snapshot(self, epoch):
        snapshot = {
            "MODEL_STATE": self.model.module.state_dict(),
            "EPOCHS_RUN": epoch,
        }
        torch.save(snapshot, self.snapshot_path)
        print(f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")

    def train(self, max_epochs):
        for epoch in range(self.epochs_run, max_epochs):
            self._run_epoch(epoch)
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_snapshot(epoch)


def load_train_objs():
    # Carregar o BERT pré-treinado para classificação de sequência
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)  # IMDb é binário
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)  # AdamW é recomendado para transformers
    return model, optimizer


def prepare_dataloader(batch_size):
    # Carregar o dataset IMDb
    dataset = load_dataset("imdb", split="train")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # Função de tokenização
    def tokenize_fn(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=512)

    # Tokenizar o dataset
    tokenized_dataset = dataset.map(tokenize_fn, batched=True)
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    # Usar DistributedSampler para dividir o dataset entre GPUs
    sampler = DistributedSampler(tokenized_dataset)

    # Criar o DataLoader
    dataloader = DataLoader(
        tokenized_dataset,
        batch_size=batch_size,
        sampler=sampler
    )
    
    return dataloader


def main(save_every, total_epochs, batch_size, snapshot_path="snapshot.pt"):
    ddp_setup()
    model, optimizer = load_train_objs()
    train_data = prepare_dataloader(batch_size)
    trainer = Trainer(model, train_data, optimizer, save_every, snapshot_path)
    trainer.train(total_epochs)
    destroy_process_group()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Distributed BERT Training on IMDb Dataset")
    parser.add_argument("total_epochs", type=int, help="Total epochs to train the model")
    parser.add_argument("save_every", type=int, help="How often to save a snapshot")
    parser.add_argument("--batch_size", default=32, type=int, help="Input batch size on each device (default: 32)")
    args = parser.parse_args()

    main(args.save_every, args.total_epochs, args.batch_size)


#torchrun --nproc_per_node=4 parallel/testBertDDP.py 10 2 --batch_size 16