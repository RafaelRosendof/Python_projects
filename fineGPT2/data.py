import pandas as pd

def preprocess_csv(file_path, output_path):
    # Ler o CSV
    df = pd.read_csv(file_path)
    
    # Exibir colunas do DataFrame para verificação
    print("Colunas disponíveis no DataFrame:")
    print(df.columns)
    
    # Definir as colunas a serem removidas
    cols_to_drop = ['reviewId', 'userName', 'score', 'thumbsUpCount', 'reviewCreatedVersion', 'at', 'appVersion']
    
    # Verificar se todas as colunas a serem removidas estão presentes
    cols_to_drop = [col for col in cols_to_drop if col in df.columns]
    
    # Remover colunas se houver alguma para remover
    if cols_to_drop:
        df.drop(columns=cols_to_drop, inplace=True)
    else:
        print("Nenhuma das colunas a serem removidas foi encontrada.")
    
    # Salvar o novo CSV com as colunas restantes
    df.to_csv(output_path, index=False)
    print(f"Arquivo processado salvo em {output_path}")

if __name__ == "__main__":
    preprocess_csv('chatgpt_reviews.csv', 'arquivo_preprocessado.csv')