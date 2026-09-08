from datetime import datetime

from sqlalchemy import (
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
)

from sqlalchemy.orm import relationship

from backend.database import Base


# ============================================================
# LEGACY CHAT MESSAGE
# ============================================================

class ChatMessage(Base):
    """
    Legacy chat history table.

    Kept for backward compatibility with the original
    /history/{account_id} endpoint and previously stored data.
    """

    __tablename__ = "chat_messages"

    id = Column(
        Integer,
        primary_key=True,
        index=True,
    )

    account_id = Column(
        String(50),
        nullable=False,
        index=True,
    )

    session_id = Column(
        String(100),
        nullable=False,
        index=True,
    )

    user_message = Column(
        Text,
        nullable=False,
    )

    assistant_message = Column(
        Text,
        nullable=False,
    )

    intent = Column(
        String(50),
        nullable=True,
    )

    service = Column(
        String(255),
        nullable=True,
    )

    rca = Column(
        Text,
        nullable=True,
    )

    recommendations = Column(
        Text,
        nullable=True,
    )

    created_at = Column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
    )


# ============================================================
# CONVERSATION
# ============================================================

class Conversation(Base):
    """
    Represents one independent user conversation/chat.
    """

    __tablename__ = "conversations"

    id = Column(
        String(100),
        primary_key=True,
        index=True,
    )

    account_id = Column(
        String(50),
        nullable=False,
        index=True,
    )

    title = Column(
        String(255),
        nullable=False,
        default="New Chat",
    )

    created_at = Column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
    )

    updated_at = Column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )

    # --------------------------------------------------------
    # Relationship
    # --------------------------------------------------------

    messages = relationship(
        "ConversationMessage",
        back_populates="conversation",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="ConversationMessage.created_at",
    )


# ============================================================
# CONVERSATION MESSAGE
# ============================================================

class ConversationMessage(Base):
    """
    Individual user or assistant message inside a conversation.

    RCA and recommendation fields are stored here so that
    incident-analysis conversations retain their complete result.
    """

    __tablename__ = "conversation_messages"

    id = Column(
        Integer,
        primary_key=True,
        index=True,
    )

    conversation_id = Column(
        String(100),
        ForeignKey(
            "conversations.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    role = Column(
        String(20),
        nullable=False,
    )

    content = Column(
        Text,
        nullable=False,
    )

    intent = Column(
        String(50),
        nullable=True,
    )

    service = Column(
        String(255),
        nullable=True,
    )

    rca = Column(
        Text,
        nullable=True,
    )

    recommendations = Column(
        Text,
        nullable=True,
    )

    created_at = Column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
    )

    # --------------------------------------------------------
    # Relationship
    # --------------------------------------------------------

    conversation = relationship(
        "Conversation",
        back_populates="messages",
    )
