import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import re, argparse
from collections import Counter

#Falta o tokenizador substituir pelo llama ou bert ou gpt2


# Tokenizador customizado para português
def custom_tokenizer(text):
    text = re.sub(r'[^\w\s]', '', text.lower())  # Remove pontuação e converte para minúsculas
    tokens = text.split()  # Divide por espaços
    return tokens

# Função para converter texto em índices de tokens
def text_pipeline(text, vocab):
    return [vocab[token] for token in custom_tokenizer(text)]

# Classe para o dataset
class MyData(Dataset):
    def __init__(self, dataframe, text_column, vocab, max_length=100):
        self.texts = [text_pipeline(text, vocab) for text in dataframe[text_column]]
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        # Truncar se o texto for maior que max_length
        if len(text) > self.max_length:
            text = text[:self.max_length]
        return torch.tensor(text, dtype=torch.long)

# Função de colagem de lotes para o DataLoader
def collate_fn(batch):
    # Pega o índice do token de padding do vocabulário
    padding_value = vocab['<pad>'] if '<pad>' in vocab.stoi else 0
    return nn.utils.rnn.pad_sequence(batch, padding_value=padding_value)

# Modelo Transformer
class MyTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_encoder_layers, num_decoder_layers, dim_feedforward, max_seq_length, dropout=0.1):
        super(MyTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_seq_length, embed_dim))
        self.transformer = nn.Transformer(
            d_model=embed_dim,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def forward(self, src, tgt):
        src_emb = self.embedding(src) + self.positional_encoding[:, :src.size(1), :]
        tgt_emb = self.embedding(tgt) + self.positional_encoding[:, :tgt.size(1), :]
        out = self.transformer(src_emb, tgt_emb)
        return self.fc_out(out)

# Função de treinamento
def trainNormal(model, criterion, opti, train_dataloader, epochs, device):
    model.to(device)
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch in train_dataloader:
            src = batch[:, :-1].to(device)
            tgt = batch[:, 1:].to(device)

            opti.zero_grad()
            output = model(src, tgt)

            # Ajustar as dimensões para a função de perda
            output = output.view(-1, vocab_size)
            tgt = tgt.view(-1)

            loss = criterion(output, tgt)
            loss.backward()
            opti.step()

            total_loss += loss.item()

        print(f'Epoch: {epoch + 1} | Loss: {total_loss / len(train_dataloader):.4f}')

# Função principal
def main():
    parser = argparse.ArgumentParser('MyTransformer')
    parser.add_argument('--path', type=str, required=True)
    parser.add_argument('--text_column', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--out_path', type=str, default='model.pth')
    args = parser.parse_args()

    # Carregar o CSV
    df = pd.read_csv(args.path)
    
    # Construir o vocabulário
    counter = Counter()
    for question in df[args.text_column]:
        counter.update(custom_tokenizer(question))

    global vocab
    vocab = Vocab(counter, min_freq=1)
    vocab_size = len(vocab)

    # Exemplo de texto para checar o vocabulário
    print(f"Vocab size: {vocab_size}")
    print(f"Example tokenization: {text_pipeline('EAi meu amigo como voce está?', vocab)}")

    # Preparar o Dataset e DataLoader
    dataset = MyData(df, args.text_column, vocab)
    train_dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=True)

    # Parâmetros do modelo
    embed_dim = 512
    num_heads = 8
    num_encoder_layers = 6
    num_decoder_layers = 6
    dim_feedforward = 2048
    max_seq_length = 100

    # Instanciar o modelo
    model = MyTransformer(vocab_size, embed_dim, num_heads, num_encoder_layers, num_decoder_layers, dim_feedforward, max_seq_length)

    # Critério e otimizador
    criterion = nn.CrossEntropyLoss()
    opti = optim.Adam(model.parameters(), lr=0.001)

    # Verificação de dispositivo (CPU ou GPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Treinar o modelo
    trainNormal(model, criterion, opti, train_dataloader, epochs=10, device=device)

    # Salvar o modelo treinado
    torch.save(model.state_dict(), args.out_path)
    print("Modelo treinado e salvo com sucesso!")

# Executa a função principal
if __name__ == "__main__":
    main()
