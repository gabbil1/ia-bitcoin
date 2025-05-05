# === IMPORTS INICIAIS ===
import ccxt
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator, ADXIndicator
from ta.volatility import BollingerBands
from ta.volume import OnBalanceVolumeIndicator
from zoneinfo import ZoneInfo
from datetime import datetime, timedelta
import time
import os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import joblib

# === CONTROLE DE CAPITAL E POSIÇÃO ===
historico_resultados = []
capital = 800.0
btc_posicao = 0.0
preco_entrada = 0.0

# === RESTAURAR ESTADO A PARTIR DO CSV ===
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
        print(f"[✔] Estado restaurado | Capital: {capital:.2f} | BTC: {btc_posicao:.6f} | Entrada: {preco_entrada:.2f}")
    except Exception as e:
        print(f"[ERRO] ao restaurar estado: {e}")
historico_resultados = []
capital = 800.0
btc_posicao = 0.0
preco_entrada = 0.0

# === FUNÇÃO PARA COLETA HISTÓRICA DE DADOS ===
def coletar_historico(par='BTC/USDT', timeframe='1h', dias=365, arquivo='historico_btc.csv'):
    print(f"[COLETA HISTÓRICA] Iniciando download de {dias} dias de dados ({timeframe}) do par {par}...")
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
    bb = BollingerBands(close=df['close'])
    df['bb_high'] = bb.bollinger_hband()
    df['bb_low'] = bb.bollinger_lband()

    df = df.dropna()
    df.to_csv(arquivo, index=False)
    print(f"[✔] Histórico salvo em {arquivo} com {len(df)} registros.")

# === FUNÇÃO PARA ROTULAR DADOS ===
def rotular_dados(arquivo='historico_btc.csv'):
    df = pd.read_csv(arquivo)
    df['retorno_futuro'] = df['close'].shift(-3) / df['close'] - 1
    df['sinal'] = 'neutro'
    df.loc[df['retorno_futuro'] > 0.01, 'sinal'] = 'compra'
    df.loc[df['retorno_futuro'] < -0.01, 'sinal'] = 'venda'
    df = df.dropna()
    df.to_csv("dados_rotulados.csv", index=False)
    print(f"[✔] Dados rotulados e salvos em dados_rotulados.csv ({len(df)} registros).")
    return df

# === TREINAMENTO DO MODELO AVANÇADO ===
def treinar_modelo_rotulado():
    df = pd.read_csv("dados_rotulados.csv")
    X = df[['rsi', 'macd', 'ema_20', 'adx', 'obv', 'bb_high', 'bb_low', 'volume']]
    y = df['sinal']
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    modelo = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=4)
    modelo.fit(X_train, y_train)
    joblib.dump(modelo, "modelo_ia_btc.pkl")
    joblib.dump(le, "rotulador_btc.pkl")
    print("[✔] Modelo treinado e salvo como modelo_ia_btc.pkl")

# === PREVISÃO EM TEMPO REAL E SIMULAÇÃO DE CAPITAL ===
def prever_sinal_1h():
    print("[PREVISÃO DE TENDÊNCIA 1H]")
    exchange = ccxt.binance()
    ohlcv = exchange.fetch_ohlcv('BTC/USDT', timeframe='1h', limit=50)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['rsi'] = RSIIndicator(close=df['close']).rsi()
    df['macd'] = MACD(close=df['close']).macd()
    df['ema_20'] = EMAIndicator(close=df['close'], window=20).ema_indicator()
    df['adx'] = ADXIndicator(high=df['high'], low=df['low'], close=df['close']).adx()
    df['obv'] = OnBalanceVolumeIndicator(close=df['close'], volume=df['volume']).on_balance_volume()
    bb = BollingerBands(close=df['close'])
    df['bb_high'] = bb.bollinger_hband()
    df['bb_low'] = bb.bollinger_lband()
    df = df.dropna().reset_index(drop=True)

    modelo = joblib.load("modelo_ia_btc.pkl")
    le = joblib.load("rotulador_btc.pkl")
    ultima_linha = df.iloc[-1:]
    X_novo = ultima_linha[['rsi', 'macd', 'ema_20', 'adx', 'obv', 'bb_high', 'bb_low', 'volume']]
    pred = modelo.predict(X_novo)
    sinal = le.inverse_transform(pred)[0]
    timestamp = pd.to_datetime(ultima_linha['timestamp'].values[0], unit='ms', utc=True).tz_convert("America/Sao_Paulo")
    print(f"[1H] Sinal previsto: {sinal.upper()} | Timestamp: {timestamp}")
    return sinal


def prever_sinal_em_tempo_real():
    global capital, btc_posicao, preco_entrada
    print("\n[PREVISÃO EM TEMPO REAL]")
    exchange = ccxt.binance()
    ohlcv = exchange.fetch_ohlcv('BTC/USDT', timeframe='1h', limit=50)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['rsi'] = RSIIndicator(close=df['close']).rsi()
    df['macd'] = MACD(close=df['close']).macd()
    df['ema_20'] = EMAIndicator(close=df['close'], window=20).ema_indicator()
    df['adx'] = ADXIndicator(high=df['high'], low=df['low'], close=df['close']).adx()
    df['obv'] = OnBalanceVolumeIndicator(close=df['close'], volume=df['volume']).on_balance_volume()
    bb = BollingerBands(close=df['close'])
    df['bb_high'] = bb.bollinger_hband()
    df['bb_low'] = bb.bollinger_lband()
    df = df.dropna().reset_index(drop=True)

    modelo = joblib.load("modelo_ia_btc.pkl")
    le = joblib.load("rotulador_btc.pkl")
    ultima_linha = df.iloc[-1:]
    X_novo = ultima_linha[['rsi', 'macd', 'ema_20', 'adx', 'obv', 'bb_high', 'bb_low', 'volume']]
    pred = modelo.predict(X_novo)
    sinal = le.inverse_transform(pred)[0]
    preco_atual = ultima_linha['close'].values[0]
    timestamp = pd.to_datetime(ultima_linha['timestamp'].values[0], unit='ms', utc=True).tz_convert("America/Sao_Paulo")

    print(f"Sinal previsto: {sinal.upper()} | Timestamp: {timestamp}")

    if sinal == "compra" and capital > 0:
        btc_posicao = capital / preco_atual
        preco_entrada = preco_atual
        capital = 0
        print(f"[COMPRA] Entrada a {preco_atual:.2f} | BTC comprado: {btc_posicao:.6f}")
        historico_resultados.append({
            'timestamp': timestamp,
            'acao': 'compra',
            'preco': preco_atual,
            'capital': round(capital, 2)
        })

    elif sinal == "venda" and btc_posicao > 0:
        capital = btc_posicao * preco_atual
        lucro = capital - (btc_posicao * preco_entrada)
        historico_resultados.append({
            'timestamp': timestamp,
            'acao': 'venda',
            'preco': preco_atual,
            'lucro': round(lucro, 2),
            'capital': round(capital, 2)
        })
        print(f"[VENDA] Saída a {preco_atual:.2f} | Lucro/prejuízo: {lucro:.2f} USDT")
        btc_posicao = 0
        preco_entrada = 0

    print(f"Capital atual: {capital:.2f} | BTC em carteira: {btc_posicao:.6f}\n")
    return sinal

# === GERAÇÃO DE GRÁFICOS DE PERFORMANCE ===
def gerar_graficos_performance():
    try:
        df = pd.read_csv("historico_operacoes.csv")
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')

        import matplotlib.pyplot as plt

        # Gráfico de evolução do capital
        plt.figure(figsize=(10, 5))
        plt.plot(df['timestamp'], df['capital'], marker='o', label='Capital USDT')
        plt.title("Evolução do Capital")
        plt.xlabel("Data")
        plt.ylabel("USDT")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig("grafico_capital.png")
        plt.close()

        # Gráfico de lucros e perdas por operação
        if 'lucro' in df.columns:
            plt.figure(figsize=(10, 5))
            df_lucros = df[df['acao'] == 'venda']
            plt.bar(df_lucros['timestamp'], df_lucros['lucro'], color=['green' if x > 0 else 'red' for x in df_lucros['lucro']])
            plt.title("Lucros e Prejuízos por Venda")
            plt.xlabel("Data")
            plt.ylabel("Lucro/Prejuízo (USDT)")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig("grafico_lucros.png")
            plt.close()

        print("[✔] Gráficos salvos como grafico_capital.png e grafico_lucros.png")

    except Exception as e:
        print(f"[ERRO] Não foi possível gerar os gráficos: {e}")

# === ATUALIZAÇÃO DO MODELO COM APRENDIZADO CONTÍNUO ===
def atualizar_modelo_automaticamente():
    try:
        df = pd.read_csv("historico_operacoes.csv")
        if len(df) < 10:
            return  # só reentreina se tiver dados suficientes
        df_rotulado = rotular_dados("historico_btc.csv")
        X = df_rotulado[['rsi', 'macd', 'ema_20', 'adx', 'obv', 'bb_high', 'bb_low', 'volume']]
        y = df_rotulado['sinal']

        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        modelo = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=4)
        modelo.fit(X, y_encoded)

        joblib.dump(modelo, "modelo_ia_btc.pkl")
        joblib.dump(le, "rotulador_btc.pkl")
        print("[✔] Modelo reentreinado com novos dados.")
    except Exception as e:
        print(f"[ERRO] ao atualizar modelo: {e}")

# === FLUXO COMPLETO ===

# === VERIFICA SE MODELO EXISTE E GERA CASO CONTRÁRIO ===
if not os.path.exists("modelo_ia_btc.pkl") or not os.path.exists("rotulador_btc.pkl"):
    print("[INFO] Modelo ou encoder não encontrado. Iniciando treinamento automático...")
    coletar_historico()
    rotular_dados()
    treinar_modelo_rotulado()

restaurar_estado()

while True:
    try:
        prever_sinal_em_tempo_real()
        prever_sinal_1h()

        if len(historico_resultados) > 0:
            df_resultados = pd.DataFrame(historico_resultados)
            df_resultados.to_csv("historico_operacoes.csv", index=False)
            print("[✔] Histórico atualizado em historico_operacoes.csv")
            gerar_graficos_performance()

            agora = datetime.now()
            if ultima_atualizacao_modelo is None or (agora - ultima_atualizacao_modelo).total_seconds() > 86400:
                atualizar_modelo_automaticamente()
                ultima_atualizacao_modelo = agora

    except Exception as e:
        print(f"Erro ao prever sinal: {e}")

    time.sleep(300)  # Espera 5 minutos

