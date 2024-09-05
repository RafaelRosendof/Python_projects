import torch
import streamlit as st
from pathlib import Path
from model import Transformer

#não, melhor é fazer isso com o llama.cpp
model = Transformer()
def load_model(model_path):
    # Carregar o modelo salvo como um dicionário de estado (state_dict)
    model = torch.load(model_path)
    model.eval()
    return model

def load_tokenizer(tokenizer_path):
    # Carregar o tokenizer diretamente do arquivo (essa parte depende do formato do tokenizer)
    # Aqui, estou considerando que o tokenizer é um objeto salvável. Pode ser necessário
    # personalizar isso dependendo de como a Meta armazenou o tokenizer.
    tokenizer = torch.load(tokenizer_path)
    return tokenizer

def evaluate(model, tokenizer, prompt, max_length=1024):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=max_length,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            repetition_penalty=1.5,
            length_penalty=1.0,
            early_stopping=True
        )
    generated_text = tokenizer.decode(outputs[0].cpu(), skip_special_tokens=True)
    return generated_text

def main():
    model_path = 'Meta-Llama-3-8B/consolidated.00.pth'
    tokenizer_path = 'Meta-Llama-3-8B/tokenizer.model'
    
    st.title("Inferência com Llama")
    st.write("## Menu de Inferência")
    
    prompt = st.text_area("Digite o texto de entrada:")
    max_length = st.number_input("Digite o tamanho máximo do texto de saída:", value=1024, min_value=1)
    
    if st.button("Gerar Texto"):
        with st.spinner('Carregando modelo...'):
            model = load_model(model_path)
            tokenizer = load_tokenizer(tokenizer_path)
        
        with st.spinner('Gerando texto...'):
            generated_text = evaluate(model, tokenizer, prompt, max_length)
            st.write("### Texto Gerado:")
            st.write(generated_text)

if __name__ == "__main__":
    main()