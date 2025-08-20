"""Message Bus for inter-agent communication using SQLite."""

import sqlite3
import json
import threading
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable
from enum import Enum


class MessagePriority(Enum):
    """Message priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


class MessageStatus(Enum):
    """Message status types."""
    PENDING = "pending"
    DELIVERED = "delivered"
    READ = "read"
    ACKNOWLEDGED = "acknowledged"
    FAILED = "failed"


class MessageBus:
    """Thread-safe message bus for agent communication."""

    def __init__(self, db_path: str = "message_bus.db"):
        self.db_path = db_path
        self._lock = threading.RLock()
        self._subscribers: Dict[str, List[Callable]] = {}
        self._initialize_database()

    def _initialize_database(self):
        """Initialize the SQLite database for message storage."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    sender_id TEXT NOT NULL,
                    receiver_id TEXT,
                    channel TEXT,
                    subject TEXT,
                    content TEXT NOT NULL,
                    message_type TEXT,
                    priority INTEGER DEFAULT 2,
                    status TEXT DEFAULT 'pending',
                    created_at TEXT,
                    delivered_at TEXT,
                    read_at TEXT,
                    acknowledged_at TEXT,
                    expires_at TEXT,
                    metadata TEXT
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS subscriptions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent_id TEXT NOT NULL,
                    channel TEXT NOT NULL,
                    created_at TEXT,
                    UNIQUE(agent_id, channel)
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS message_acknowledgments (
                    message_id TEXT,
                    agent_id TEXT,
                    acknowledged_at TEXT,
                    response TEXT,
                    PRIMARY KEY (message_id, agent_id)
                )
            """)

            # Create indexes for better performance
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_messages_receiver ON messages(receiver_id)")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_messages_channel ON messages(channel)")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_messages_status ON messages(status)")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_messages_priority ON messages(priority)")

            conn.commit()

    def send_message(
        self,
        sender_id: str,
        content: Any,
        receiver_id: str = None,
        channel: str = None,
        subject: str = None,
        message_type: str = "general",
        priority: MessagePriority = MessagePriority.NORMAL,
        expires_in_hours: int = None,
        metadata: Dict[str, Any] = None
    ) -> str:
        """Send a message to a specific agent or channel."""
        if not receiver_id and not channel:
            raise ValueError("Either receiver_id or channel must be specified")

        message_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        expires_at = None

        if expires_in_hours:
            expires_at = (datetime.now() +
                          timedelta(hours=expires_in_hours)).isoformat()

        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("""
                        INSERT INTO messages 
                        (id, sender_id, receiver_id, channel, subject, content, message_type, 
                         priority, status, created_at, expires_at, metadata)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        message_id, sender_id, receiver_id, channel, subject,
                        json.dumps(content), message_type, priority.value,
                        MessageStatus.PENDING.value, now, expires_at,
                        json.dumps(metadata) if metadata else None
                    ))
                    conn.commit()

                # Notify subscribers if it's a channel message
                if channel and channel in self._subscribers:
                    self._notify_subscribers(channel, message_id)

                return message_id
            except Exception as e:
                print(f"Error sending message: {e}")
                return None

    def get_messages(
        self,
        agent_id: str,
        status: MessageStatus = None,
        channel: str = None,
        limit: int = 50,
        include_expired: bool = False
    ) -> List[Dict[str, Any]]:
        """Get messages for a specific agent."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = """
                    SELECT id, sender_id, receiver_id, channel, subject, content, 
                           message_type, priority, status, created_at, delivered_at,
                           read_at, acknowledged_at, expires_at, metadata
                    FROM messages 
                    WHERE (receiver_id = ? OR channel IN (
                        SELECT channel FROM subscriptions WHERE agent_id = ?
                    ))
                """
                params = [agent_id, agent_id]

                if status:
                    query += " AND status = ?"
                    params.append(status.value)

                if channel:
                    query += " AND channel = ?"
                    params.append(channel)

                if not include_expired:
                    query += " AND (expires_at IS NULL OR expires_at > ?)"
                    params.append(datetime.now().isoformat())

                query += " ORDER BY priority DESC, created_at DESC LIMIT ?"
                params.append(limit)

                cursor = conn.execute(query, params)
                messages = []

                for row in cursor.fetchall():
                    message = {
                        'id': row[0],
                        'sender_id': row[1],
                        'receiver_id': row[2],
                        'channel': row[3],
                        'subject': row[4],
                        'content': json.loads(row[5]) if row[5] else None,
                        'message_type': row[6],
                        'priority': row[7],
                        'status': row[8],
                        'created_at': row[9],
                        'delivered_at': row[10],
                        'read_at': row[11],
                        'acknowledged_at': row[12],
                        'expires_at': row[13],
                        'metadata': json.loads(row[14]) if row[14] else None
                    }
                    messages.append(message)

                return messages
        except Exception as e:
            print(f"Error getting messages: {e}")
            return []

    def mark_message_status(
        self,
        message_id: str,
        status: MessageStatus,
        agent_id: str = None
    ) -> bool:
        """Mark a message with a specific status."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                now = datetime.now().isoformat()

                if status == MessageStatus.DELIVERED:
                    conn.execute("""
                        UPDATE messages SET status = ?, delivered_at = ? WHERE id = ?
                    """, (status.value, now, message_id))
                elif status == MessageStatus.READ:
                    conn.execute("""
                        UPDATE messages SET status = ?, read_at = ? WHERE id = ?
                    """, (status.value, now, message_id))
                elif status == MessageStatus.ACKNOWLEDGED:
                    conn.execute("""
                        UPDATE messages SET status = ?, acknowledged_at = ? WHERE id = ?
                    """, (status.value, now, message_id))
                else:
                    conn.execute("""
                        UPDATE messages SET status = ? WHERE id = ?
                    """, (status.value, message_id))

                conn.commit()
                return True
        except Exception as e:
            print(f"Error marking message status: {e}")
            return False

    def acknowledge_message(
        self,
        message_id: str,
        agent_id: str,
        response: str = None
    ) -> bool:
        """Acknowledge receipt and processing of a message."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                now = datetime.now().isoformat()

                # Add acknowledgment record
                conn.execute("""
                    INSERT OR REPLACE INTO message_acknowledgments 
                    (message_id, agent_id, acknowledged_at, response)
                    VALUES (?, ?, ?, ?)
                """, (message_id, agent_id, now, response))

                # Update message status
                conn.execute("""
                    UPDATE messages SET status = ?, acknowledged_at = ? WHERE id = ?
                """, (MessageStatus.ACKNOWLEDGED.value, now, message_id))

                conn.commit()
                return True
        except Exception as e:
            print(f"Error acknowledging message: {e}")
            return False

    def subscribe_to_channel(self, agent_id: str, channel: str) -> bool:
        """Subscribe an agent to a communication channel."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR IGNORE INTO subscriptions (agent_id, channel, created_at)
                    VALUES (?, ?, ?)
                """, (agent_id, channel, datetime.now().isoformat()))
                conn.commit()
                return True
        except Exception as e:
            print(f"Error subscribing to channel: {e}")
            return False

    def unsubscribe_from_channel(self, agent_id: str, channel: str) -> bool:
        """Unsubscribe an agent from a communication channel."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    DELETE FROM subscriptions WHERE agent_id = ? AND channel = ?
                """, (agent_id, channel))
                conn.commit()
                return True
        except Exception as e:
            print(f"Error unsubscribing from channel: {e}")
            return False

    def get_channels(self, agent_id: str = None) -> List[str]:
        """Get all channels (optionally filtered by agent subscriptions)."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                if agent_id:
                    cursor = conn.execute("""
                        SELECT DISTINCT channel FROM subscriptions WHERE agent_id = ?
                    """, (agent_id,))
                else:
                    cursor = conn.execute(
                        "SELECT DISTINCT channel FROM messages WHERE channel IS NOT NULL")

                return [row[0] for row in cursor.fetchall()]
        except Exception as e:
            print(f"Error getting channels: {e}")
            return []

    def broadcast_message(
        self,
        sender_id: str,
        channel: str,
        content: Any,
        subject: str = None,
        message_type: str = "broadcast",
        priority: MessagePriority = MessagePriority.NORMAL,
        metadata: Dict[str, Any] = None
    ) -> str:
        """Broadcast a message to all subscribers of a channel."""
        return self.send_message(
            sender_id=sender_id,
            content=content,
            channel=channel,
            subject=subject,
            message_type=message_type,
            priority=priority,
            metadata=metadata
        )

    def get_conversation(
        self,
        agent1_id: str,
        agent2_id: str,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get conversation history between two agents."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT id, sender_id, receiver_id, subject, content, message_type,
                           priority, status, created_at, metadata
                    FROM messages 
                    WHERE (sender_id = ? AND receiver_id = ?) 
                       OR (sender_id = ? AND receiver_id = ?)
                    ORDER BY created_at ASC LIMIT ?
                """, (agent1_id, agent2_id, agent2_id, agent1_id, limit))

                messages = []
                for row in cursor.fetchall():
                    message = {
                        'id': row[0],
                        'sender_id': row[1],
                        'receiver_id': row[2],
                        'subject': row[3],
                        'content': json.loads(row[4]) if row[4] else None,
                        'message_type': row[5],
                        'priority': row[6],
                        'status': row[7],
                        'created_at': row[8],
                        'metadata': json.loads(row[9]) if row[9] else None
                    }
                    messages.append(message)

                return messages
        except Exception as e:
            print(f"Error getting conversation: {e}")
            return []

    def _notify_subscribers(self, channel: str, message_id: str):
        """Notify in-memory subscribers about new messages."""
        if channel in self._subscribers:
            for callback in self._subscribers[channel]:
                try:
                    callback(message_id)
                except Exception as e:
                    print(f"Error notifying subscriber: {e}")

    def add_subscriber_callback(self, channel: str, callback: Callable[[str], None]):
        """Add an in-memory callback for real-time notifications."""
        with self._lock:
            if channel not in self._subscribers:
                self._subscribers[channel] = []
            self._subscribers[channel].append(callback)

    def remove_subscriber_callback(self, channel: str, callback: Callable[[str], None]):
        """Remove an in-memory callback."""
        with self._lock:
            if channel in self._subscribers and callback in self._subscribers[channel]:
                self._subscribers[channel].remove(callback)

    def cleanup_expired_messages(self) -> int:
        """Clean up expired messages and return count of deleted messages."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    DELETE FROM messages 
                    WHERE expires_at IS NOT NULL AND expires_at < ?
                """, (datetime.now().isoformat(),))
                deleted_count = cursor.rowcount
                conn.commit()
                return deleted_count
        except Exception as e:
            print(f"Error cleaning up expired messages: {e}")
            return 0

    def get_message_stats(self) -> Dict[str, Any]:
        """Get statistics about message usage."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Total messages
                cursor = conn.execute("SELECT COUNT(*) FROM messages")
                total_messages = cursor.fetchone()[0]

                # Messages by status
                cursor = conn.execute("""
                    SELECT status, COUNT(*) FROM messages GROUP BY status
                """)
                status_counts = dict(cursor.fetchall())

                # Messages by priority
                cursor = conn.execute("""
                    SELECT priority, COUNT(*) FROM messages GROUP BY priority
                """)
                priority_counts = dict(cursor.fetchall())

                # Top channels
                cursor = conn.execute("""
                    SELECT channel, COUNT(*) FROM messages 
                    WHERE channel IS NOT NULL 
                    GROUP BY channel 
                    ORDER BY COUNT(*) DESC LIMIT 5
                """)
                top_channels = dict(cursor.fetchall())

                return {
                    'total_messages': total_messages,
                    'status_counts': status_counts,
                    'priority_counts': priority_counts,
                    'top_channels': top_channels
                }
        except Exception as e:
            print(f"Error getting message stats: {e}")
            return {}
