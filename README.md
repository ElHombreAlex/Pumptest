# Pump.fun AI Trading Agent

An autonomous trading agent for [Pump.fun](https://pump.fun) that uses **Claude AI** to analyse newly launched tokens in real-time and execute buy/sell orders on the Solana blockchain.

## Architecture

```
PumpPortal WebSocket
        │
        ▼
 PumpPortalClient          ← streams new tokens + real-time trades
        │
   ┌────┴────┐
   │         │
   ▼         ▼
TokenEvent  TradeEvent
   │             │
   ▼             ▼
TokenAnalyzer  RiskManager  ← monitors TP/SL for open positions
(Claude AI)         │
   │          TradeExecutor ◄─────────────────┐
   │               │                          │
   └──────►  buy() │                     sell()
                   │
             Solana RPC (via PumpPortal local tx API)
```

### Key components

| File | Role |
|------|------|
| `main.py` | Entry point, logging setup, config validation |
| `agent.py` | Orchestrator — wires everything together |
| `pumpportal_client.py` | WebSocket client for PumpPortal real-time data |
| `analyzer.py` | Claude AI token scorer |
| `trader.py` | Trade builder, signer, and broadcaster |
| `risk_manager.py` | Position tracker, TP/SL exit logic |
| `models.py` | Shared dataclasses |
| `config.py` | All settings via environment variables |

## Quick Start

### 1. Clone and install

```bash
git clone <repo>
cd Pumptest
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env — at minimum set ANTHROPIC_API_KEY
```

### 3. Run in dry-run mode (no real trades)

```bash
python main.py
```

The agent will:
- Connect to PumpPortal WebSocket
- Stream new token launches
- Collect initial trade data
- Ask Claude to score each token
- Log BUY/SKIP decisions without sending real transactions

### 4. Enable live trading

Set in `.env`:
```
DRY_RUN=false
WALLET_PRIVATE_KEY=<your base58 private key>
WALLET_PUBLIC_KEY=<your public key>
SOLANA_RPC_URL=https://mainnet.helius-rpc.com/?api-key=<your key>
```

> **Warning:** Live trading involves real financial risk. Start with `MAX_BUY_SOL=0.001` and monitor carefully.

## Configuration Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | — | Claude AI key (required) |
| `WALLET_PRIVATE_KEY` | — | Base58 Solana private key |
| `WALLET_PUBLIC_KEY` | — | Solana wallet address |
| `SOLANA_RPC_URL` | mainnet-beta | RPC endpoint |
| `PUMPPORTAL_API_KEY` | — | Optional; reduces fees |
| `MAX_BUY_SOL` | 0.01 | Max SOL per trade |
| `MIN_CONFIDENCE_SCORE` | 70 | Claude score threshold (0-100) |
| `TAKE_PROFIT_MULTIPLIER` | 1.5 | Exit at 1.5× entry market cap |
| `STOP_LOSS_MULTIPLIER` | 0.7 | Exit at 0.7× entry market cap |
| `MAX_POSITIONS` | 3 | Max concurrent positions |
| `SLIPPAGE` | 10 | Slippage tolerance % |
| `PRIORITY_FEE` | 0.00001 | Solana priority fee in SOL |
| `DRY_RUN` | true | Paper-trade mode |

## How the AI Analysis Works

For each new token Claude receives:
- Token metadata (name, symbol, description, socials)
- Creator wallet address
- Initial developer buy amount
- First N trades with SOL amounts and market cap movement

Claude returns a structured JSON with:
- **confidence_score** (0-100)
- **recommendation** (BUY / WATCH / SKIP)
- **reasoning** (2-4 sentences)
- **risk_flags** (e.g. `rug_risk`, `no_socials`, `pump_and_dump`)
- **suggested_buy_sol** (size recommendation)

The agent only buys if `score >= MIN_CONFIDENCE_SCORE` and a position slot is free.

## Data Source

Real-time data is provided by [PumpPortal](https://pumpportal.fun) — a free third-party API for Pump.fun.

WebSocket: `wss://pumpportal.fun/api/data`
Trade API: `https://pumpportal.fun/api/trade-local`

## Disclaimer

This software is for educational purposes. Cryptocurrency trading carries significant financial risk. Never trade with funds you cannot afford to lose.
