from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="kalman-pairs-trading",
    version="1.0.0",
    author="Octavio de Godoy",
    author_email="your.email@example.com",
    description="Production-ready Kalman Filter pairs trading system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/octaviodegodoy/kalman-pairs-trading",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience ::  Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial ::  Investment",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "pytest-cov>=2.12.0",
            "black>=21.0",
            "flake8>=3.9.0",
            "mypy>=0.910",
        ],
        "ml": [
            "tensorflow>=2.6.0",
            "keras>=2.6.0",
            "torch>=1.9.0",
        ],
        "dashboard": [
            "streamlit>=1.10.0",
            "dash>=2.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "kalman-backtest=src.cli:run_backtest",
            "kalman-optimize=src.cli:optimize_parameters",
            "kalman-trade=src.cli:start_trading",
        ],
    },
)