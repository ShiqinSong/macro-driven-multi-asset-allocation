# Macro-Driven Multi-Asset Allocation

This project examines how macroeconomic signals can inform tactical asset allocation around a diversified strategic portfolio.

The portfolio covers seven asset classes: U.S. equities, international equities, U.S. Treasuries, investment-grade credit, high-yield credit, commodities, and gold.

## Approach

- Construct strategic portfolio weights using rolling Equal Risk Contribution (ERC).
- Estimate relationships between macroeconomic factors and asset excess returns using a five-factor rolling regression model.
- Use model signals to adjust portfolio weights through tactical tilts.
- Compare portfolio performance, drawdowns, and turnover, including a 10-basis-point transaction cost assumption.

## Project Files

| File | Purpose |
|---|---|
| Part 1 - Weight selection & Final output.ipynb | Strategic portfolio construction and allocation analysis |
| Part 2 - Macro factor.py | Macroeconomic factor modeling and return prediction |
| Part 3 - Tactical allocation.ipynb | Tactical allocation and backtesting |

## Tools

Python, pandas, NumPy, and statistical modeling.

## Author

Shiqin Song
