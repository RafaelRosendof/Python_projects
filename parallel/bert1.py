import torch , os , pandas as pd , argparse , time 
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import BertTokenizer , BertForSequenceClassification , AdamW
from torch.nn.functional import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist , torch.optim as optim
from sklearn.model_selection import train_test_split

import functools 
from torch.distributed.fsdp.wrap import (size_based_auto_wrap_policy,)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

def clean_data(data):
    df = pd.read_csv(data)
    return df 
    
BATCH_SIZE = 256 // int(os.environ["WORLD_SIZE"])
WORKERS = 48
    
class MyData(Dataset):

    def __init__(self , tokenizer , data , max_len):
       self.tokenizer = tokenizer  
       self.data = data
       self.max_len = max_len 


    def __len__(self):
        return len(self.data)

    def __getitem__(self , idx):
        row = self.data.iloc[idx]
        content = row['clean_text']
        target = row['category']

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
            'targets': torch.tensor(target, dtype=torch.long)
        }

'''     
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel , self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.3)
        self.output = nn.Linear(768, 2)

    def forward(self , ids , mask):
        output = self.bert(ids , attention_mask = mask)
        output = self.dropout(output)
        output = self.output(output)
        return output
'''

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

    def forward(self, ids, mask):
        output = self.bert(ids, attention_mask=mask)
        return output.logits

    
def train(model , train_loader , optimizer , loss_fn , epochs , local_rank , global_rank):
    for epoch in range(epochs):
        model.train()
        train_bar = tqdm(train_loader , desc=f'Epoca {epoch+1} - Treinamento')
        for batch in train_bar:
            ids = batch['ids'].to(local_rank)
            mask = batch['mask'].to(local_rank)
            target = batch['targets'].to(local_rank)

            optimizer.zero_grad()
            outputs = model(ids , mask)
            loss = loss_fn(outputs , target)
            loss.backward()
            optimizer.step()

            train_bar.set_postfix(loss=loss.item())
        
        if global_rank == 0:
            print(f'Epoca {epoch+1} - Loss: {loss.item()}')

#        model.eval()
#        total_Loss = 0
#        total_correct = 0
#        total_sample = 0


def test(model, test_loader, loss_fn, epochs, local_rank):
    for epoch in range(epochs):
        total_loss = 0
        total_correct = 0
        total_samples = 0
        test_bar = tqdm(test_loader, desc=f'Epoca {epoch+1} - Teste')
        model.eval()

        for batch in test_bar:
            ids = batch['ids'].to(local_rank)
            mask = batch['mask'].to(local_rank)
            target = batch['targets'].to(local_rank)

            with torch.no_grad():
                outputs = model(ids, mask)
                loss = loss_fn(outputs, target)
                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total_correct += (predicted == target).sum().item()
                total_samples += target.size(0)

                test_bar.set_postfix(loss=loss.item())
        avg_loss = total_loss / len(test_loader)
        accuracy = total_correct / total_samples

        print(f'Loss: {avg_loss} - Accuracy: {accuracy}')



#def train and def test 

def main():
    parser = argparse.ArgumentParser('Treinamento do bert distribuido')
    parser.add_argument('-bs' , '--batch-size', type=int, default=8)
    parser.add_argument('-e' , '--epochs' , type=int, default=5)
    parser.add_argument('-lr' , '--local_rank' , type=int)
    args = parser.parse_args()

    #parte do backend em cuda e fsdp
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    global_rank = int(os.environ['RANK'])
    torch.cuda.set_device(local_rank)
    torch.cuda.empty_cache()


    df = clean_data('Twitter_Data.csv')
    sampler = DistributedSampler(df)
    train_df, test_df = train_test_split(df, test_size=0.1)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_data = MyData(tokenizer, train_df, 512)
    test_data = MyData(tokenizer, test_df, 512)

    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, num_workers=WORKERS, shuffle=False, sampler=sampler)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, num_workers=WORKERS, shuffle=False, sampler=sampler)

    model = MyModel().to('cuda:' + str(local_rank))
    optimizer = AdamW(model.parameters(), lr=5e-5)
    loss = nn.CrossEntropyLoss()

    my_auto = functools.partial(size_based_auto_wrap_policy, min_num_params=20000)
    model = FSDP(model, auto_wrap_policy=my_auto)

    train(model, train_dataloader, optimizer, loss, args.epochs, local_rank=local_rank, global_rank=global_rank)
    test(model, test_loader, loss, args.epochs, local_rank=local_rank)

    torch.save(model.state_dict(), 'modelBERTFSDP.pth')
    print("Treinamento finalizado")

if __name__ == "__main__":
    main()

    
