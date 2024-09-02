import torch , argparse
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.utils.data import Dataset , DataLoader
import pandas as pd

from datasets import load_dataset, load_metric
from transformers import TrainingArguments, Trainer


class myData(Dataset):
    
    def __init__(self , file , tokenizer , max_len=2048):
        self.data = pd.read_csv(file)
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        self.texts = self.data['content']
        
    def __len__(self):
        return len(self.texts)

    def __getitem__(self , idx):
        
        text = str(self.texts[idx])

        inputs = self.tokenizer(text , truncation=True,
                                padding='max_length',
                                max_length=self.max_len,
                                 return_tensors='pt')

        input_ids = inputs['input_ids'].squeeze()
        attention_mask = inputs['attention_mask'].squeeze()
        
        max_token_id = self.tokenizer.vocab_size - 1
        assert torch.all(input_ids <= max_token_id), "Há IDs de tokens fora do intervalo do vocabulário."
        
        # Verificação do tamanho dos tensors
        assert input_ids.size() == attention_mask.size(), "O tamanho do attention_mask deve corresponder ao tamanho do input_ids."


        return{
            'input_ids': input_ids,#precisa do label?
            'attention_mask': attention_mask,    
        }
        
def tokenize_function(examples,tokenizer):
    #inputs = tokenizer(examples['Bad_Practices'], return_tensors='pt', padding='max_length', max_length=512, truncation=True)
    labels = tokenizer(examples['content'], return_tensors='pt', padding='max_length', max_length=512, truncation=True)
    return {'labels': labels['input_ids']}

def main():
    parser = argparse.ArgumentParser(description='Fine-tuning GPT-2 on a custom dataset')
    parser.add_argument('--e' , type=int , default=1 , help='Number of epochs')
    parser.add_argument('--lr' , type=float , default=1e-4 , help='Learning rate')
    parser.add_argument('--d' , type=str , help='Path to the dataset')
    parser.add_argument('--o' , type=str , help='Path to the output model')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')
    
  #  model = net().to(device)
  #  otimization = torch.optim.Adam(model.parameters() , lr = args.lr)
  #  loss_fn = nn.CrossEntropyLoss()
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    model_name = "gpt2"
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    #data é um csv agora 
    
    #dataset = myData(args.d, tokenizer)
    #train_loader = DataLoader(dataset , batch_size=1)
   # passa(train_loader)
    print("Dataset carregado")
    #train(model , train_loader , otimization , loss_fn , device , args.e)
    
    #torch.save(model.state_dict() , args.o)
    data = load_dataset('csv', data_files=args.d)
    #train_data = data["train"].select([i for i in range(len(data["train"])) if i % 10 != 0])
    #val_data = data["train"].select([i for i in range(len(data["train"])) if i % 10 == 0])
    token = tokenize_function(data, tokenizer)
    train_data = data.map(token, batched=True)
    
    # Define training arguments
    training_args = TrainingArguments(
    output_dir='./model',
    overwrite_output_dir=True,
    num_train_epochs=0.5,
    per_device_train_batch_size=2,
    evaluation_strategy="steps",
    eval_steps=100,
    save_steps=500,
    logging_steps=100,
    logging_dir='./logs',
)

# Create a Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        
)

# Fine-tune the model
    trainer.train()
    
if __name__ == '__main__':
    main()
    
    
    
    
    
'''      

class net(nn.Module):
    def __init__(self , num_labels=768):
        super(net, self).__init__()
        
        self.gpt = GPT2LMHeadModel.from_pretrained('gpt2')

        #self.att = nn.MultiheadAttention(768, 12)

        #self.att1 = nn.MultiheadAttention(768, 12)
        
        self.fc = nn.Linear(768, num_labels)

    
    def forward(self, x , mask, labels):
        outputs = self.gpt(input_ids=x, attention_mask=mask, labels=labels)
        loss = outputs.loss
        logits = outputs.logits
        return loss, logits
 
        
def train(model , train_loader , optimizer , loss_fn , device , epochs):
    print("Começou")
    for epoch in range(epochs):
        print("Iniciando Treinamento")
        model.train()
        loss_total = 0.0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = input_ids.clone()
            assert input_ids.size() == labels.size()
            optimizer.zero_grad()

            # Forward pass
            #outputs = model(input_ids, attention_mask)
            #loss = loss_fn(outputs, input_ids).to(device)

            # Backward pass and optimization
#            loss.backward()
#            optimizer.step()

#            loss_total += loss.item()

    # Forward pass
            loss, _ = model(input_ids, attention_mask, labels)
            loss = loss.to(device)

    # Backward pass and optimization
            loss.backward()
            optimizer.step()

            loss_total += loss.item()

        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss_total/len(train_loader)}')

def passa(dataloader):
    for i in dataloader:
        print(i)
        break
            '''