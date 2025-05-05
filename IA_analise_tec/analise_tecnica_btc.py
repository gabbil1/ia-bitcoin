# === BIBLIOTECAS NECESSÁRIAS ===
import ccxt
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator
from zoneinfo import ZoneInfo
from datetime import datetime
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.multiclass import unique_labels
import joblib

# === 1. COLETA DE DADOS COM AJUSTE DE TIMEZONE ===
def buscar_dados(par='BTC/USDT', timeframe='5m', limite=500):
    print(f"Coletando dados de {par} no timeframe {timeframe}...")
    exchange = ccxt.binance()
    ohlcv = exchange.fetch_ohlcv(par, timeframe=timeframe, limit=limite)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True).dt.tz_convert("America/Sao_Paulo")
    return df

# === 2. CÁLCULO DE INDICADORES TÉCNICOS ===
def adicionar_indicadores(df):
    print("Calculando indicadores técnicos (RSI, MACD, EMA)...")
    df['rsi'] = RSIIndicator(close=df['close']).rsi()
    df['macd'] = MACD(close=df['close']).macd()
    df['ema_20'] = EMAIndicator(close=df['close'], window=20).ema_indicator()
    return df

# === 3. GERAÇÃO DE SINAIS COMPOSTOS ===
def gerar_sinais_compostos(df):
    print("Gerando sinais compostos com base em MACD, RSI e EMA...")
    sinais = []
    for i in range(1, len(df)):
        if df['macd'][i] > 0 and df['rsi'][i] < 70 and df['close'][i] > df['ema_20'][i]:
            sinais.append("compra")
        elif df['macd'][i] < 0 and df['rsi'][i] > 30 and df['close'][i] < df['ema_20'][i]:
            sinais.append("venda")
        else:
            sinais.append("neutro")
    sinais.insert(0, "neutro")
    df['sinal'] = sinais
    return df

# === 4. BACKTEST SIMPLES ===
def backtest_simples(df):
    capital = 1000
    btc = 0
    for i in range(1, len(df)):
        if df['sinal'][i] == "compra" and capital > 0:
            btc = capital / df['close'][i]
            capital = 0
        elif df['sinal'][i] == "venda" and btc > 0:
            capital = btc * df['close'][i]
            btc = 0
    final = capital if capital > 0 else btc * df['close'].iloc[-1]
    retorno = (final - 1000) / 1000 * 100
    return round(retorno, 2)

# === 5. GRÁFICO INTERATIVO COM PLOTLY ===
def plotar_grafico_interativo(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['close'], mode='lines', name='Preço Fechamento'))
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['ema_20'], mode='lines', name='EMA 20', line=dict(dash='dash')))

    df_compra = df[df['sinal'] == 'compra']
    fig.add_trace(go.Scatter(x=df_compra['timestamp'], y=df_compra['close'], mode='markers',
                             name='Compra', marker=dict(symbol='triangle-up', color='green', size=10)))

    df_venda = df[df['sinal'] == 'venda']
    fig.add_trace(go.Scatter(x=df_venda['timestamp'], y=df_venda['close'], mode='markers',
                             name='Venda', marker=dict(symbol='triangle-down', color='red', size=10)))

    fig.update_layout(title='Análise Técnica BTC/USDT (Gráfico Interativo)',
                      xaxis_title='Data', yaxis_title='Preço (USDT)',
                      legend=dict(x=0, y=1.1, orientation='h'), height=600)
    fig.show()

# === 6. MACHINE LEARNING - TREINAMENTO E AVALIAÇÃO ===
def treinar_modelo_machine_learning(df):
    df = df.dropna()
    le = LabelEncoder()
    df['sinal_num'] = le.fit_transform(df['sinal'])

    X = df[['rsi', 'macd', 'ema_20', 'close', 'volume']]
    y = df['sinal_num']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    modelo = RandomForestClassifier(n_estimators=100, random_state=42)
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    labels = unique_labels(y, y_pred)
    relatorio = classification_report(y_test, y_pred, labels=labels, target_names=le.inverse_transform(labels), zero_division=0)

    joblib.dump(modelo, 'modelo_btc.pkl')
    joblib.dump(le, 'label_encoder_btc.pkl')

    return relatorio

# === 7. EXECUÇÃO COMPLETA ===
def executar_analise(par='BTC/USDT', timeframe='5m'):
    df = buscar_dados(par, timeframe)
    df = adicionar_indicadores(df)
    df = gerar_sinais_compostos(df)
    retorno = backtest_simples(df)

    print("\n=== Resultado da Análise ===")
    print(f"Timestamp (Horário São Paulo): {df.iloc[-1]['timestamp']}")
    print(f"Preço de Fechamento: {df.iloc[-1]['close']:.2f}")
    print(f"RSI: {df.iloc[-1]['rsi']:.2f}")
    print(f"MACD: {df.iloc[-1]['macd']:.4f}")
    print(f"EMA(20): {df.iloc[-1]['ema_20']:.2f}")
    print(f"Sinal Gerado: {df.iloc[-1]['sinal']}")
    print(f"Retorno Simulado da Estratégia: {retorno:.2f}%")

    plotar_grafico_interativo(df)

    print("\n=== Treinamento de IA ===")
    relatorio = treinar_modelo_machine_learning(df)
    print(relatorio)

    return df

# === CHAMADA DO SCRIPT ===
if __name__ == "__main__":
    executar_analise(par='BTC/USDT', timeframe='5m')
