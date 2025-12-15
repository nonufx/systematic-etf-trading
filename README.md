# Quantitative ETF Trading Strategy

This repository contains a Python-based quantitative trading research framework focused on liquid U.S. ETFs.

## Overview
The strategy dynamically allocates capital across SPY, QQQ, IWM, XLF, and XLI using a composite signal framework.

Key components:
- Log-return based feature engineering
- Momentum and mean-reversion signals
- Volatility scaling
- Rolling Sharpe ratio filtering
- AR(1) regime detection
- Turnover-aware portfolio construction
- Transaction cost modeling

## Structure
- `quant_route.py` — main strategy pipeline
- `metrics.py` — signal and feature engineering
- `scoring.py` — composite scoring logic
- `portfolio.py` — portfolio construction and execution logic

## Disclaimer
This project is for educational and research purposes only.
