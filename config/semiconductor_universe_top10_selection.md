# Semiconductor Top 10 Universe Selection

Generated for the 2026-05-22 top-10 multi-symbol research universe.

## Selected Symbols

| Rank | Symbol | Role |
|---:|---|---|
| 1 | NVDA | AI accelerator / fabless |
| 2 | TSM | Foundry / original anchor |
| 3 | AVGO | Networking / custom silicon |
| 4 | AMD | CPU/GPU fabless |
| 5 | INTC | IDM / foundry transition |
| 6 | MU | Memory |
| 7 | TXN | Analog |
| 8 | LRCX | Wafer equipment |
| 9 | AMAT | Wafer equipment |
| 10 | QCOM | Mobile / edge fabless |

## Quant Selection Rules

- Keep the universe small enough for daily automation and pooled model iteration.
- Use only US-traded symbols with existing Yahoo/Stooq support and sufficient daily history.
- Prefer names that appear with high weight in broad semiconductor benchmarks, especially SMH and SOX.
- Preserve value-chain coverage instead of selecting only AI/fabless names.
- Keep TSM because the system still maintains TSM-compatible root outputs.

## Web Sources Checked

- VanEck SMH holdings page: https://www.vaneck.com/us/en/investments/semiconductor-etf-smh/holdings/
- SMH current holdings mirror checked on 2026-05-22: https://stockanalysis.com/etf/smh/holdings/
- Nasdaq PHLX Semiconductor Sector Index fact sheet: https://www.nasdaq.com/docs/2026/05/05/SOX.pdf
- Nasdaq SOX index fact sheet: https://indexes.nasdaq.com/docs/FS_SOX.pdf

## Excluded Near-Misses

- ADI: strong analog name, but TXN was retained as the analog representative.
- KLAC: strong equipment name, but AMAT and LRCX were retained for equipment coverage.
- ASML: strategically important, but ranked behind selected names in this 10-name execution universe.
- MRVL: appears in SOX top weights, but was excluded to avoid overloading fabless/networking exposure.
