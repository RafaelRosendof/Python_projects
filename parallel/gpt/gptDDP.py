import os ,pandas as pd , torch , argparse
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.nn.functional import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer , GPT2LMHeadModel

#Set device cuda and enviroment
dist.init_process_group(backend='nccl')
local_rank = int(os.environ['LOCAL_RANK'])
global_rank = int(os.environ['RANK'])
torch.cuda.set_device(local_rank)
torch.cuda.empty_cache()

EPOCHS = 128
BATCH_SIZE = 64 // int(os.environ["WORLD_SIZE"])
WORKERS = 48


class MyData(Dataset):
    def __init__(self , data_path):
        super().__init__()

        sjoke = os.path.join(data_path , 'shortjokes.csv')
        self.joke_list = []
        self.end_of_text_token = "<|endoftext|>"

        with open(sjoke ) as csv_file:
            csv_reader = csv.reader(csv_file , delimiter=',')

            x = 0

            for row in csv_reader:
                joke_str = f"JOKE:{row[1]}{self.end_of_text_token}"
                self.joke_list.append(joke_str)
            
    def __len__(self):
        return len(self.joke_list)

    def __getitem__(self , item):
        return self.joke_list[item]


def read_data(data_path):
    df = pd.read_csv(data_path)
    return df
        
class MyData2(Dataset):
    def __init__(self , tokenizer , data , max_len):
        self.tokenizer = tokenizer
        self.data = data
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self , idx):
        row = self.data.iloc[idx] #explica isso?
        content = row['Joke']
        number = row['ID']

        inputs = self.tokenizer.encode_plus(
            content,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True
        )

        return {
            'ids': torch.tensor(inputs['input_ids'], dtype=torch.long),
            'mask': torch.tensor(inputs['attention_mask'], dtype=torch.long),
            'targets': torch.tensor(inputs['input_ids'] , dtype=torch.long)
        }

class MyModel1(nn.Module):
    def __init__(self):
        super(MyModel1 , self).__init__()
        self.gpt2 = GPT2LMHeadModel.from_pretrained('gpt2-medium')
    
    def forward(self , input_ids , attention_mask):
        outputs = self.gpt2(input_ids , attention_mask = attention_mask)
        return outputs.logits()


def train(model , train_loader , optimizer , loss_fn , epochs):
    for epoch in range(EPOCHS):
        model.train()
        train_bar = tqdm(train_loader , desc = f'Epoch {epoch + 1} - Treinamento')
        for batch in train_bar:
            ids = batch['ids'].to(local_rank)    
            mask = batch['mask'].to(local_rank)
            target = batch['targets'].to(local_rank)

            optimizer.zero_grad()
            outputs = model(ids , mask)

            # Flatten the outputs and target for CrossEntropyLoss
            outputs = outputs.view(-1, outputs.size(-1))  # Flatten to (batch_size * seq_len, vocab_size)
            target = target.view(-1)  # Flatten to (batch_size * seq_len)

            loss = loss_fn(outputs , target)
            loss.backward()
            optimizer.step()

            train_bar.set_postfix(loss=loss.item())

        if global_rank == 0:
            print(f"Epoch {epoch+1} - Loss: {loss.item()}")
        
def test(model, test_loader, loss_fn):
    model.eval()
    total_loss = 0
    total_steps = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            ids = batch['ids'].to(local_rank)
            mask = batch['mask'].to(local_rank)
            target = batch['targets'].to(local_rank)

            outputs = model(ids, mask)
            outputs = outputs.view(-1, outputs.size(-1))
            target = target.view(-1)

            loss = loss_fn(outputs, target)
            total_loss += loss.item()
            total_steps += 1

    avg_loss = total_loss / total_steps
    print(f"Average test loss: {avg_loss}")
 

def main():
    print("\n\n\n\nIniciando treinamento do modelo GPT2 com DDP \n\n\n\n\n")

    parser = argparse.ArgumentParser("GPT2 com DDP ")
    df = read_data('shortjokes.csv')
    sampler = DistributedSampler(df)
    train_df, test_df = train_test_split(df, test_size=0.1)

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')

    train_data = MyData2(tokenizer, train_df, 128)
    test_data = MyData2(tokenizer, test_df, 128)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, num_workers=WORKERS, shuffle=False, sampler=sampler)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, num_workers=WORKERS, shuffle=False, sampler=sampler)

    model = MyModel1()
    model = model.to('cuda:' + str(local_rank))
    model = DDP(model, device_ids=[local_rank])

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss = nn.CrossEntropyLoss()

    train(model, train_loader, optimizer, loss, EPOCHS)
    test(model, test_loader, loss)  # Run the test after training

    torch.save(model.module.state_dict(), 'gpt2_jokes.pth')
    print("\n\n\n ################ Treinamento Finalizado ################ \n\n\n")

if __name__ == '__main__':
    main()

