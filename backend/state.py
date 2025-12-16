from typing import TypedDict, Annotated, List, Optional
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
import pandas as pd

# 1. Define the Global Store (Simple version for single user)
class GlobalStore:
    def __init__(self):
        self.df: Optional[pd.DataFrame] = None
    
    def set_df(self, df: pd.DataFrame):
        self.df = df
    
    def get_df(self) -> Optional[pd.DataFrame]:
        return self.df
    
    def clear(self):
        self.df = None

# Create a single instance to be imported by other files
STORE = GlobalStore()

# 2. Define AppState
class AppState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    youtube_url: Optional[str]
    is_processed: bool
    image_suggestions: Optional[List[str]]
    transcript_structure: Optional[List[dict]]