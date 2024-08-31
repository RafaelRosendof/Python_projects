import streamlit as st 
from pages import menuIntegral

def show_main_menu():
    st.write("## Menu de Integrais")
    a = st.number_input("Digite o valor de A:", value=0.0)
    b = st.number_input("Digite o valor de B:", value=1.0)
    n = st.number_input("Digite o valor de N:", value=100, min_value=1)
    #tol = st.number_input("Digite o valor de TOL:", value=0.00000001)
    n_workers = st.number_input("Digite o número de trabalhadores (threads):", value=4, min_value=1)

    tol = 1e-8
    if st.button("Calcular"):
        st.write(f"A: {a}, B: {b}, N: {n}, TOL: {tol}, Workers: {n_workers}")
        menuIntegral.main(a, b, n, tol, n_workers)

if 'page' not in st.session_state:
    st.session_state.page = "main_menu"

if st.session_state.page == "main_menu":
    show_main_menu()