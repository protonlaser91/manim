__all__ = ["Interpolator"]

from abc import abstractmethod


class Interpolator:
    @abstractmethod
    def interpolate(self,alpha):
        pass
