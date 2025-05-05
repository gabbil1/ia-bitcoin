import ccxt
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator

# === 1. COLETA DE DADOS DA BINANCE ===
def buscar_dados(par='BTC/USDT', timeframe='1h', limite=500):
    print(f"Coletando dados de {par} no timeframe {timeframe}...")
    exchange = ccxt.binance()
    ohlcv = exchange.fetch_ohlcv(par, timeframe=timeframe, limit=limite)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

# === 2. CÁLCULO DE INDICADORES TÉCNICOS ===
def adicionar_indicadores(df):
    print("Calculando indicadores técnicos (RSI, MACD, EMA)...")
    df['rsi'] = RSIIndicator(close=df['close']).rsi()
    df['macd'] = MACD(close=df['close']).macd()
    df['ema_20'] = EMAIndicator(close=df['close'], window=20).ema_indicator()
    return df

# === 3. ESTRATÉGIA BASEADA EM MACD ===
def gerar_sinais_macd(df):
    print("Gerando sinais com base no MACD...")
    sinais = []
    for i in range(1, len(df)):
        if df['macd'][i] > 0 and df['macd'][i-1] <= 0:
            sinais.append("compra")
        elif df['macd'][i] < 0 and df['macd'][i-1] >= 0:
            sinais.append("venda")
        else:
            sinais.append("neutro")
    sinais.insert(0, "neutro")  # O primeiro não tem sinal anterior para comparação
    df['sinal'] = sinais
    return df

# === 4. EXECUÇÃO GERAL ===
def executar_analise(par='BTC/USDT', timeframe='1h'):
    df = buscar_dados(par, timeframe)
    df = adicionar_indicadores(df)
    df = gerar_sinais_macd(df)

    ultimo_sinal = df.iloc[-1]
    print("\n=== Resultado da Análise ===")
    print(f"Timestamp: {ultimo_sinal['timestamp']}")
    print(f"Preço de Fechamento: {ultimo_sinal['close']:.2f}")
    print(f"RSI: {ultimo_sinal['rsi']:.2f}")
    print(f"MACD: {ultimo_sinal['macd']:.4f}")
    print(f"EMA(20): {ultimo_sinal['ema_20']:.2f}")
    print(f"Sinal Gerado: {ultimo_sinal['sinal']}")

    return df

# === EXECUÇÃO DO SCRIPT ===
if __name__ == "__main__":
    # Altere o timeframe conforme necessário: '1m', '15m', '1h', '4h', '1d', '1w', '1M'
    executar_analise(par='BTC/USDT', timeframe='1h')

