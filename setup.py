from setuptools import setup, find_packages

setup(
    name="binance-trading-bot",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "ccxt>=4.0.0",
        "pandas>=1.5.0",
        "numpy>=1.24.0",
        "ta>=0.10.0",
        "rich>=13.0.0",
        "python-binance>=1.0.0",
    ],
    author="Trading Bot",
    description="Advanced Binance Trading Bot with CLI Dashboard",
    python_requires=">=3.8",
)