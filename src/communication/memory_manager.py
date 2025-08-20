"""Shared Memory Manager for agent communication using SQLite."""

import sqlite3
import json
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from pathlib import Path


class SharedMemoryManager:
    """Thread-safe shared memory manager using SQLite backend."""

    def __init__(self, db_path: str = "shared_memory.db"):
        self.db_path = db_path
        self._lock = threading.RLock()
        self._initialize_database()

    def _initialize_database(self):
        """Initialize the SQLite database with required tables."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS shared_memory (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    data_type TEXT,
                    created_at TEXT,
                    updated_at TEXT,
                    created_by TEXT,
                    access_count INTEGER DEFAULT 0
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS memory_locks (
                    key TEXT PRIMARY KEY,
                    locked_by TEXT,
                    locked_at TEXT
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS memory_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    key TEXT,
                    old_value TEXT,
                    new_value TEXT,
                    changed_by TEXT,
                    changed_at TEXT
                )
            """)

            conn.commit()

    def set(self, key: str, value: Any, agent_id: str = "unknown") -> bool:
        """Set a value in shared memory."""
        with self._lock:
            try:
                serialized_value = json.dumps(value)
                data_type = type(value).__name__
                now = datetime.now().isoformat()

                with sqlite3.connect(self.db_path) as conn:
                    # Check if key exists for history tracking
                    cursor = conn.execute(
                        "SELECT value FROM shared_memory WHERE key = ?", (key,))
                    old_value = cursor.fetchone()

                    # Update or insert
                    conn.execute("""
                        INSERT OR REPLACE INTO shared_memory 
                        (key, value, data_type, created_at, updated_at, created_by, access_count)
                        VALUES (?, ?, ?, 
                               COALESCE((SELECT created_at FROM shared_memory WHERE key = ?), ?),
                               ?, ?, 
                               COALESCE((SELECT access_count FROM shared_memory WHERE key = ?), 0))
                    """, (key, serialized_value, data_type, key, now, now, agent_id, key))

                    # Add to history
                    if old_value:
                        conn.execute("""
                            INSERT INTO memory_history (key, old_value, new_value, changed_by, changed_at)
                            VALUES (?, ?, ?, ?, ?)
                        """, (key, old_value[0], serialized_value, agent_id, now))

                    conn.commit()
                return True
            except Exception as e:
                print(f"Error setting shared memory key {key}: {e}")
                return False

    def get(self, key: str, agent_id: str = "unknown") -> Optional[Any]:
        """Get a value from shared memory."""
        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute("""
                        SELECT value, data_type FROM shared_memory WHERE key = ?
                    """, (key,))
                    result = cursor.fetchone()

                    if result:
                        # Increment access count
                        conn.execute("""
                            UPDATE shared_memory SET access_count = access_count + 1 
                            WHERE key = ?
                        """, (key,))
                        conn.commit()

                        return json.loads(result[0])
                    return None
            except Exception as e:
                print(f"Error getting shared memory key {key}: {e}")
                return None

    def delete(self, key: str, agent_id: str = "unknown") -> bool:
        """Delete a key from shared memory."""
        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute(
                        "DELETE FROM shared_memory WHERE key = ?", (key,))
                    conn.execute(
                        "DELETE FROM memory_locks WHERE key = ?", (key,))
                    conn.commit()
                return True
            except Exception as e:
                print(f"Error deleting shared memory key {key}: {e}")
                return False

    def list_keys(self, pattern: str = None) -> List[str]:
        """List all keys in shared memory, optionally filtered by pattern."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                if pattern:
                    cursor = conn.execute("""
                        SELECT key FROM shared_memory WHERE key LIKE ?
                    """, (f"%{pattern}%",))
                else:
                    cursor = conn.execute("SELECT key FROM shared_memory")

                return [row[0] for row in cursor.fetchall()]
        except Exception as e:
            print(f"Error listing shared memory keys: {e}")
            return []

    def get_metadata(self, key: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific key."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT created_at, updated_at, created_by, access_count, data_type
                    FROM shared_memory WHERE key = ?
                """, (key,))
                result = cursor.fetchone()

                if result:
                    return {
                        'created_at': result[0],
                        'updated_at': result[1],
                        'created_by': result[2],
                        'access_count': result[3],
                        'data_type': result[4]
                    }
                return None
        except Exception as e:
            print(f"Error getting metadata for key {key}: {e}")
            return None

    def acquire_lock(self, key: str, agent_id: str) -> bool:
        """Acquire an exclusive lock on a memory key."""
        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    # Check if already locked
                    cursor = conn.execute(
                        "SELECT locked_by FROM memory_locks WHERE key = ?", (key,))
                    if cursor.fetchone():
                        return False

                    # Acquire lock
                    conn.execute("""
                        INSERT INTO memory_locks (key, locked_by, locked_at)
                        VALUES (?, ?, ?)
                    """, (key, agent_id, datetime.now().isoformat()))
                    conn.commit()
                    return True
            except Exception as e:
                print(f"Error acquiring lock for key {key}: {e}")
                return False

    def release_lock(self, key: str, agent_id: str) -> bool:
        """Release a lock on a memory key."""
        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("""
                        DELETE FROM memory_locks WHERE key = ? AND locked_by = ?
                    """, (key, agent_id))
                    conn.commit()
                    return True
            except Exception as e:
                print(f"Error releasing lock for key {key}: {e}")
                return False

    def get_history(self, key: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get change history for a specific key."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT old_value, new_value, changed_by, changed_at
                    FROM memory_history WHERE key = ?
                    ORDER BY changed_at DESC LIMIT ?
                """, (key, limit))

                return [
                    {
                        'old_value': json.loads(row[0]) if row[0] else None,
                        'new_value': json.loads(row[1]) if row[1] else None,
                        'changed_by': row[2],
                        'changed_at': row[3]
                    }
                    for row in cursor.fetchall()
                ]
        except Exception as e:
            print(f"Error getting history for key {key}: {e}")
            return []

    def clear_all(self) -> bool:
        """Clear all shared memory (use with caution)."""
        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("DELETE FROM shared_memory")
                    conn.execute("DELETE FROM memory_locks")
                    conn.execute("DELETE FROM memory_history")
                    conn.commit()
                return True
            except Exception as e:
                print(f"Error clearing shared memory: {e}")
                return False

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about shared memory usage."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM shared_memory")
                total_keys = cursor.fetchone()[0]

                cursor = conn.execute("SELECT COUNT(*) FROM memory_locks")
                locked_keys = cursor.fetchone()[0]

                cursor = conn.execute("""
                    SELECT key, access_count FROM shared_memory 
                    ORDER BY access_count DESC LIMIT 5
                """)
                top_accessed = cursor.fetchall()

                return {
                    'total_keys': total_keys,
                    'locked_keys': locked_keys,
                    'top_accessed': [{'key': row[0], 'count': row[1]} for row in top_accessed]
                }
        except Exception as e:
            print(f"Error getting shared memory stats: {e}")
            return {}
