import torch 
import streamlit as st 
from transformers import LlamaForCasuallLm, LlamaTokenizer

def evaluate(model, tokenizer, prompt, max_length=1024):
    inputs = tokenizer(prompt , return_tensors="pt")
    outputs = model.generate(inputs.input_ids ,
                             max_length=max_length,
                             num_return_sequences=1,
                             no_repeat_ngram_size=2,
                             repetition_penalty=1.5,
                             length_penalty=1.0,
                             early_stopping=True,)
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():

    model_path = 'Meta-Llama-3-8B/consolidated.00.pth'
    tokenizer_path = 'Meta-Llama-3-8B/tokenizer.model'
    
    st.title("Inferência com Llama")
    st.write("## Menu de Inferência")
    prompt = st.text_area("Digite o texto de entrada:")
    max_length = st.number_input("Digite o tamanho máximo do texto de saída:", value=1024, min_value=1)
    
    if st.button("Gerar Prompt"):
        with st.spinner("Carregando modelo"):
            model = LlamaForCasuallLm.from_pretrained(model_path)
            tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path)

        with st.spinner("Gerando texto"):
            generated_text = evaluate(model, tokenizer, prompt, max_length)
            st.write("## Texto Gerado")
            st.write(generated_text)

if __name__ == "__main__":
    main()