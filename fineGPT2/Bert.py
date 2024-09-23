import  torch , argparse 
import pandas as pd
from transformers import GPT2Tokenizer , GPT2MHeadModel, AdamW
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

def load_csv(file):
    df = pd.read_csv(file)
    return df

def load_txt(file):
    df = pd.read_csv(file , sep='\n' , header=None) #checar isso ou sep?
    return df


#### Pytorch dataset(Vetificar essa class para o meu tipo de dado )

class MyData(Dataset):
    
    def __init__(self , data , tokenizer , max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self , idx):
        text = self.data.iloc[idx , 0]
        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length = self.max_len,
            return_tensors='pt',
            padding='max_length',
            truncation=True
        )
        
        input_ids = inputs['input_ids'].flatten()

        return {
            'input_ids': input_ids,
            'attention_mask': inputs['attention_mask'].flatten(),
            'label': input_ids
        }

'''
#Proposta de segundo modelo mudar o bert e colocar o gpt fazer essa alteração 
class MyBert2(nn.Module):
    def __init__(self , model):
        super(MyBert2 , self).__init__()
        self.atten1 = nn.MultiheadAttention(768 , 12)
        self.atten2 = nn.MultiheadAttention(768 , 12)
        self.softmax = nn.LogSofmax(dim=1)

    def forward(self , input_ids , attention_mask):
        x = self.atten1(input_ids , input_ids , input_ids)
        x = self.atten2(x , x , x)

        x = self.softmax(x)

        return x
'''

#Proposta de treino pytorch
def train(model , optimizer , data_loader , device):
    model.train()
    total_loss = 0.0

    for step , batch in enumerate(data_loader):
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        label = batch['label'].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids = input_ids , attention_mask = attention_mask)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    return total_loss / len(data_loader)

#Proposta de validação pytorch
def evaluate(model , data_loader , device):
    print("Avaliando")

    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for step , batch in enumerate(data_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids = input_ids , attention_mask = attention_mask)
            loss = outputs.loss
            total_loss += loss.item()

    print("Avaliação finalizada")

    return total_loss / len(data_loader)

def split_data(df , test_size=0.2):
    train_size = int((i-test_size)*len(df))
    train_df = df[:train_size]
    val_df = df[train_size:]
    return train_df , val_df

def main():
    print('Iniciando o fine tune')

    parser = argparse.ArgumentParser('Fine tune Bert')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--i', type=str, required=True)
    parser.add_argument('--o', type=str, required=True)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Carrega o dataset
    df = load_csv(args.i)  # Supondo que a entrada seja um arquivo CSV

    # Divide o dataset em treino e validação
    train_df, val_df = split_data(df)

    tokenizer = GPT2Tokenizer.from_pretained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2MHeadModel.from_pretrained('gpt2')
    model.resize_token_embeddings(len(tokenizer))

    
    # Cria os datasets e dataloaders
    train_dataset = MyData(train_df, tokenizer, max_len=128)
    val_dataset = MyData(val_df, tokenizer, max_len=128)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=5e-5)
    model.to(device)

    
    for epoch in range(args.epochs):
        print(f'Epoch {epoch + 1}/{args.epochs}')
        train_loss = train(model, optimizer, train_loader, device)
        val_loss = evaluate(model, val_loader, device)

        print(f'Train Loss: {train_loss}, Validation Loss: {val_loss}')

    # Salva o modelo
    torch.save(model.state_dict(), args.o)

   
if __name__ == "__main__":
    main()
   






