from .model_base import BaseModel
from .model_brownian_motion import BrownianMotionModel 

def model_base(**params):
    return BaseModel(params)

def model_brownian_motion(**params):
    return BrownianMotionModel(params)