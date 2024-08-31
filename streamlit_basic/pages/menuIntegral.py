import streamlit as st 
import numpy as np
from parallelTrapz import parallel_trapezoidal

def func(x):
    return np.log10(np.sqrt(x * x + x)+1)

def main(a , b , n , tol , n_workers):
    st.title("Calculadora de Integrais")
    st.write("## Menu de Integrais")
    #st.write("Seja bem-vindo à calculadora de integrais. Aqui você pode calcular integrais definidas de funções reais de uma variável real. Para começar, selecione uma das opções abaixo:")
    st.write("Regra do trapézio paralelizada")

    #a = st.number_input("Digite o valor de a:", value=0.0)
    #b = st.number_input("Digite o valor de b:", value=1.0)
    #n = st.number_input("Digite o valor de n:", value=100 , min_value=1)
    #tol = st.number_input("Digite o valor de tolerância:", value=1e-10)
    #n_workers = st.number_input("Digite o número de workers:", value=4 , min_value=1)
    

    #if st.button("Shumppert"):
    result = parallel_trapezoidal(func, a, b, n, tol, n_workers)
    st.write(f"O resultado da integral é: {result}")


