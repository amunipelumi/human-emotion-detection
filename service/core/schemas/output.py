from pydantic import BaseModel

class ApiOutput(BaseModel):
    emotion: str
    # time4loading:str
    total_time: str