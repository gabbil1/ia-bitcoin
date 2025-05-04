import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
from datetime import datetime
import requests
import subprocess

st.set_page_config(page_title="Painel IA Bitcoin", layout="wide")
st.title("📊 Painel de Acompanhamento - IA Bitcoin")

# === Função para carregar os dados ===
def carregar_dados():
    if not os.path.exists("historico_operacoes.csv"):
        return pd.DataFrame()
    df = pd.read_csv("historico_operacoes.csv")
    if not df.empty:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

# === Carrega os dados em tempo real ===
st_autorefresh = st.experimental_rerun if st.session_state.get("auto_atualizar", False) else lambda: None

df = carregar_dados()

if df.empty:
    st.warning("O arquivo 'historico_operacoes.csv' está vazio ou ainda não foi gerado pela IA.")
    st.stop()

# === Informações principais ===
st.subheader("📌 Informações Gerais")
col1, col2, col3 = st.columns(3)
col1.metric("Capital Atual (USDT)", f"{df.iloc[-1]['capital']:.2f}")
col2.metric("Preço de Entrada", f"{df[df['acao'] == 'compra']['preco'].iloc[-1]:.2f}" if 'compra' in df['acao'].values else "-")
col3.metric("Total de Operações", len(df))

# === Gráfico do capital ao longo do tempo ===
st.subheader("📈 Evolução do Capital")
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(df['timestamp'], df['capital'], marker='o')
ax.set_title("Capital em USDT ao longo do tempo")
ax.set_xlabel("Data")
ax.set_ylabel("Capital (USDT)")
ax.grid(True)
st.pyplot(fig)

# === Gráfico de lucros e perdas ===
if 'lucro' in df.columns:
    st.subheader("💰 Lucros e Prejuízos por Venda")
    df_lucro = df[df['acao'] == 'venda']
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    colors = ['green' if val > 0 else 'red' for val in df_lucro['lucro']]
    ax2.bar(df_lucro['timestamp'], df_lucro['lucro'], color=colors)
    ax2.set_title("Lucro/Prejuízo por Venda")
    ax2.set_xlabel("Data")
    ax2.set_ylabel("Lucro/Prejuízo (USDT)")
    ax2.grid(True)
    st.pyplot(fig2)

# === Controle da IA ===
st.subheader("⚙️ Controle da IA")

if "ia_rodando" not in st.session_state:
    st.session_state.ia_rodando = False

if st.button("▶️ Iniciar IA"):
    try:
        subprocess.Popen(["python", "ia_bitcoin_main.py"])
        st.session_state.ia_rodando = True
        st.success("IA iniciada com sucesso (modo automático).")
    except Exception as e:
        st.error(f"Erro ao iniciar IA: {e}")

if st.button("⏹️ Parar IA"):
    try:
        response = requests.post("http://localhost:5000/parar")
        if response.status_code == 200:
            st.session_state.ia_rodando = False
            st.warning("IA parada.")
        else:
            st.error("Erro ao parar IA.")
    except Exception as e:
        st.error(f"Erro ao conectar à API: {e}")

st.markdown(f"**Status da IA:** {'🟢 Rodando' if st.session_state.ia_rodando else '🔴 Parada'}")

# === Atualização automática ===
st.sidebar.subheader("🔄 Atualização Automática")
auto_atualizar = st.sidebar.checkbox("Atualizar automaticamente a cada 30s", value=False)
st.session_state.auto_atualizar = auto_atualizar
if auto_atualizar:
    time.sleep(30)
    st.experimental_rerun()
