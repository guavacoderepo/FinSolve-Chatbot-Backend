from pydantic import BaseModel
from typing import Optional
from enum import Enum

class Roles(Enum):
    Engineering = "engineering"
    Employee = "general"
    Finance = "finance"
    Executives = "executives"
    Marketing = "marketing"
    HR = "hr"

class User(BaseModel):
    id:int
    username:str
    role: Roles

class ConversationModel(BaseModel):
    userId: int
    prompt: str
    response: Optional[str]
    date: str

class ChatModel(BaseModel):
    prompt:str

class RegisterModel(BaseModel):
    username:str
    role: Roles
    password: str

class AuthResponseModel(BaseModel):
    status: bool
    user: User
    token: Optional[str] = None
    
class LoginModel(BaseModel):
    username:str
    password: str
    api_key: str
