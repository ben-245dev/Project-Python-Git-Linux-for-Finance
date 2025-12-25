from abc import ABC, abstractmethod

class TradingStrategy(ABC):
    name = "undefined"
    @abstractmethod
    def generate_signals(self, df, price_col):
        pass
    @abstractmethod
    def compute_equity_curve(self, df, price_col):
        pass
