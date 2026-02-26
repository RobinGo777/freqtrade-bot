FROM freqtradeorg/freqtrade:stable

# Copy strategy
COPY user_data/strategies/ /freqtrade/user_data/strategies/

# Copy config
COPY user_data/config.json /freqtrade/user_data/config.json

# Create required directories
RUN mkdir -p /freqtrade/user_data/logs \
             /freqtrade/user_data/data \
             /freqtrade/user_data/backtest_results

# Run freqtrade in trade mode
CMD ["freqtrade", "trade", \
     "--config", "user_data/config.json", \
     "--strategy", "RSIDivergence12H_Breakeven_v4", \
     "--logfile", "user_data/logs/freqtrade.log"]
