from .BaseModel import BaseModel as BaseModelClass
from .GeometricBrownianMotion import GeometricBrownianMotion as GBMClass

def BaseModel(**params):
    return BaseModelClass(params)

def GeometricBrownianMotion(**params):
    return GBMClass(params)