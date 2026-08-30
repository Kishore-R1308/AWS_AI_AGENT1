from datetime import datetime

from sqlalchemy import Column, DateTime, Integer, String, Text

from database import Base


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, index=True)
    account_id= Column(String(50), index=True, nullable=False)
    session_id = Column(String(100), index=True, nullable=False)
    user_message = Column(Text, nullable=False)
    assistant_message = Column(Text, nullable=False)
    intent = Column(String(50), nullable=False)
    service = Column(String(50), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
