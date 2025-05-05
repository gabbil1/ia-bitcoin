import ccxt
import pandas as pd
import numpy as np
import streamlit as st
from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator
from ta.trend import MACD, EMAIndicator, ADXIndicator, CCIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator
from zoneinfo import ZoneInfo
from datetime import datetime, timedelta
import time
import os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import matplotlib.pyplot as plt

# === CONTROLE DE CAPITAL E POSIÇÃO ===
historico_resultados = []
capital = 800.0
btc_posicao = 0.0
preco_entrada = 0.0
ultima_atualizacao_modelo = None

# === RESTAURAR ESTADO ===
def restaurar_estado():
    global capital, btc_posicao, preco_entrada
    try:
        df = pd.read_csv("historico_operacoes.csv")
        if df.empty:
            return
        ultima = df.iloc[-1]
        capital = ultima['capital']
        if ultima['acao'] == 'compra':
            btc_posicao = capital / ultima['preco'] if ultima['preco'] > 0 else 0
            preco_entrada = ultima['preco']
    except:
        pass

# === COLETA HISTÓRICA DE DADOS ===
def coletar_historico(par='BTC/USDT', timeframe='1h', dias=365, arquivo='historico_btc.csv'):
    exchange = ccxt.binance()
    limite = 1000
    ms_por_candle = exchange.parse_timeframe(timeframe) * 1000
    agora = exchange.milliseconds()
    desde = agora - (dias * 24 * 60 * 60 * 1000)
    todos_candles = []

    while desde < agora:
        candles = exchange.fetch_ohlcv(par, timeframe=timeframe, since=desde, limit=limite)
        if not candles:
            break
        todos_candles += candles
        desde = candles[-1][0] + ms_por_candle
        time.sleep(0.1)

    df = pd.DataFrame(todos_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True).dt.tz_convert("America/Sao_Paulo")
    df['rsi'] = RSIIndicator(close=df['close']).rsi()
    df['macd'] = MACD(close=df['close']).macd()
    df['ema_20'] = EMAIndicator(close=df['close'], window=20).ema_indicator()
    df['adx'] = ADXIndicator(high=df['high'], low=df['low'], close=df['close']).adx()
    df['obv'] = OnBalanceVolumeIndicator(close=df['close'], volume=df['volume']).on_balance_volume()
    df['cci'] = CCIIndicator(high=df['high'], low=df['low'], close=df['close'], window=20).cci()
    df['atr'] = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14).average_true_range()
    df['stoch'] = StochasticOscillator(high=df['high'], low=df['low'], close=df['close']).stoch()
    df['williams_r'] = WilliamsRIndicator(high=df['high'], low=df['low'], close=df['close']).williams_r()
    bb = BollingerBands(close=df['close'])
    df['bb_high'] = bb.bollinger_hband()
    df['bb_low'] = bb.bollinger_lband()
    df = df.dropna()
    df.to_csv(arquivo, index=False)

# === ROTULAGEM E TREINAMENTO ===
def rotular_dados(arquivo='historico_btc.csv'):
    df = pd.read_csv(arquivo)
    df['retorno_futuro'] = df['close'].shift(-3) / df['close'] - 1
    df['sinal'] = 'neutro'
    df.loc[df['retorno_futuro'] > 0.01, 'sinal'] = 'compra'
    df.loc[df['retorno_futuro'] < -0.01, 'sinal'] = 'venda'
    df = df.dropna()
    df.to_csv("dados_rotulados.csv", index=False)


def treinar_modelo_rotulado():
    df = pd.read_csv("dados_rotulados.csv")
    X = df.drop(columns=['timestamp', 'open', 'high', 'low', 'close', 'retorno_futuro', 'sinal'])
    y = df['sinal']
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    X_train, _, y_train, _ = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    modelo = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=4)
    modelo.fit(X_train, y_train)
    joblib.dump(modelo, "modelo_ia_btc.pkl")
    joblib.dump(le, "rotulador_btc.pkl")

# === PREVISÃO ===
def prever_sinal_atual():
    exchange = ccxt.binance()
    ohlcv = exchange.fetch_ohlcv('BTC/USDT', timeframe='1h', limit=50)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['rsi'] = RSIIndicator(close=df['close']).rsi()
    df['macd'] = MACD(close=df['close']).macd()
    df['ema_20'] = EMAIndicator(close=df['close'], window=20).ema_indicator()
    df['adx'] = ADXIndicator(high=df['high'], low=df['low'], close=df['close']).adx()
    df['obv'] = OnBalanceVolumeIndicator(close=df['close'], volume=df['volume']).on_balance_volume()
    df['cci'] = CCIIndicator(high=df['high'], low=df['low'], close=df['close'], window=20).cci()
    df['atr'] = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14).average_true_range()
    df['stoch'] = StochasticOscillator(high=df['high'], low=df['low'], close=df['close']).stoch()
    df['williams_r'] = WilliamsRIndicator(high=df['high'], low=df['low'], close=df['close']).williams_r()
    bb = BollingerBands(close=df['close'])
    df['bb_high'] = bb.bollinger_hband()
    df['bb_low'] = bb.bollinger_lband()
    df = df.dropna().reset_index(drop=True)
    modelo = joblib.load("modelo_ia_btc.pkl")
    le = joblib.load("rotulador_btc.pkl")
    X_novo = df.iloc[-1:][modelo.feature_names_in_]
    pred = modelo.predict(X_novo)
    return le.inverse_transform(pred)[0], df.iloc[-1]['close']

# === INTERFACE STREAMLIT ===
st.set_page_config(page_title="IA Bitcoin", layout="centered")
st.title("🔍 IA de Análise Técnica para Bitcoin")

if not os.path.exists("modelo_ia_btc.pkl") or not os.path.exists("rotulador_btc.pkl"):
    st.warning("Modelo não encontrado. Treinando agora...")
    coletar_historico()
    rotular_dados()
    treinar_modelo_rotulado()

restaurar_estado()
sinal, preco = prever_sinal_atual()

st.metric("Sinal Atual da IA", sinal.upper())
st.metric("Preço Atual do BTC", f"${preco:,.2f}")
st.metric("Capital Atual (USDT)", f"{capital:.2f}")
st.metric("BTC em Carteira", f"{btc_posicao:.6f}")

if st.button("Atualizar Sinal Agora"):
    sinal, preco = prever_sinal_atual()
    st.success(f"Novo Sinal: {sinal.upper()} | Preço BTC: ${preco:,.2f}")
