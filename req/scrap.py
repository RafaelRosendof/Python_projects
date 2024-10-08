import requests as req 
import json , os 
from bs4 import BeautifulSoup as bs

url = ' https://kitcia.com.br/produto/alavanca-de-cambio-completa-5-marchas-7/'
url1 = 'https://www.gb.com.br/orcamento-online/alavanca-cambio-completa-com-rotula-gol-parati-saveiro-06-5-marchas/'

def extrair(url_path):
    
    reqs = req.get(url_path)

    if reqs.status_code == 200:
        soup = bs(reqs.content , 'html.parser')

        #Pegando o principal da página 
        texto = soup.get_text(separator='\n' , strip=True)

        return texto

    else:
        print('Erro ao extrair dados: ', reqs.status_code)
        return None

    
    
def processaHTML(dado , dado_saida):

    if dado:
        with open(dado_saida , 'w' , encoding='utf-8') as f:
            f.write(dado)
        print(f'Dados extraidos com sucesso {dado_saida}')
    else:
        print('Erro ao extrair dados')
    
    
    
def main():
    print('Extraindo dados do site: ', url)
    dado1 = extrair(url)
    
    print('Extraindo dados do site: ', url1)
    dado2 = extrair(url1)

    # Salvando em arquivos .txt
    processaHTML(dado1, 'saida1.txt')
    processaHTML(dado2, 'saida2.txt')


if __name__ == '__main__':
    main()