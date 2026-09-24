"""StockAI: semnale BUY/SELL/HOLD din date tehnice, fundamentale și știri, cu Claude ca a doua opinie."""

from .analyzer import Analyzer, Recommendation
from .config import Settings

__all__ = ["Analyzer", "Recommendation", "Settings"]
