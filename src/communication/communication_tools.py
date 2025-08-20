"""Communication tools for CrewAI agents."""

from crewai.tools import BaseTool
from typing import Type, Any, Dict, List, Optional
from pydantic import BaseModel, Field
import json

from .memory_manager import SharedMemoryManager
from .message_bus import MessageBus, MessagePriority, MessageStatus
from .knowledge_store import KnowledgeStore


# Shared Memory Tools

class SharedMemoryInput(BaseModel):
    """Input schema for shared memory operations."""
    action: str = Field(
        ..., description="Action: set, get, delete, list, lock, unlock, history, stats")
    key: str = Field(
        None, description="Memory key for get/set/delete operations")
    value: str = Field(None, description="Value to store (JSON string)")
    agent_id: str = Field("unknown", description="Agent identifier")
    pattern: str = Field(None, description="Pattern for listing keys")


class SharedMemoryTool(BaseTool):
    name: str = "shared_memory"
    description: str = (
        "Access shared memory for inter-agent data sharing. Supports set/get/delete operations with locking."
    )
    args_schema: Type[BaseModel] = SharedMemoryInput

    def _run(self, action: str, key: str = None, value: str = None,
             agent_id: str = "unknown", pattern: str = None) -> str:
        # Initialize memory manager inside _run to avoid Pydantic conflicts
        memory_manager = SharedMemoryManager("shared_memory.db")

        try:
            if action == "set":
                if not key or value is None:
                    return "❌ **Error**: Both key and value required for set operation"

                try:
                    parsed_value = json.loads(value)
                except json.JSONDecodeError:
                    parsed_value = value  # Store as string if not valid JSON

                success = memory_manager.set(key, parsed_value, agent_id)
                return f"✅ **Memory Set**: {key} = {value}" if success else f"❌ **Failed** to set {key}"

            elif action == "get":
                if not key:
                    return "❌ **Error**: Key required for get operation"

                value = memory_manager.get(key, agent_id)
                if value is not None:
                    return f"📋 **Memory Value**: {key} = {json.dumps(value)}"
                return f"❌ **Not Found**: No value for key '{key}'"

            elif action == "delete":
                if not key:
                    return "❌ **Error**: Key required for delete operation"

                success = memory_manager.delete(key, agent_id)
                return f"🗑️ **Deleted**: {key}" if success else f"❌ **Failed** to delete {key}"

            elif action == "list":
                keys = memory_manager.list_keys(pattern)
                if keys:
                    return f"📋 **Memory Keys**:\n" + "\n".join([f"- {k}" for k in keys])
                return "📋 **No keys found**" + (f" matching '{pattern}'" if pattern else "")

            elif action == "lock":
                if not key:
                    return "❌ **Error**: Key required for lock operation"
                
                success = memory_manager.acquire_lock(key, agent_id)
                return f"🔒 **Locked**: {key}" if success else f"❌ **Lock Failed**: {key} (already locked?)"

            elif action == "unlock":
                if not key:
                    return "❌ **Error**: Key required for unlock operation"
                
                success = memory_manager.release_lock(key, agent_id)
                return f"🔓 **Unlocked**: {key}" if success else f"❌ **Unlock Failed**: {key}"

            elif action == "history":
                if not key:
                    return "❌ **Error**: Key required for history operation"
                
                history = memory_manager.get_history(key)
                if history:
                    result = f"📊 **History for {key}**:\n"
                    for entry in history:
                        result += f"- {entry['changed_at']}: {entry['changed_by']} changed value\n"
                    return result
                return f"📊 **No history** found for {key}"

            elif action == "stats":
                stats = memory_manager.get_stats()
                return f"""📊 **Memory Statistics**:
- Total Keys: {stats.get('total_keys', 0)}
- Locked Keys: {stats.get('locked_keys', 0)}
- Top Accessed: {', '.join([f"{item['key']} ({item['count']})" for item in stats.get('top_accessed', [])])}"""

            else:
                return f"❌ **Invalid Action**: {action}. Use: set, get, delete, list, lock, unlock, history, stats"

        except Exception as e:
            return f"❌ **Memory Error**: {str(e)}"


# Message Passing Tools

class MessagePassingInput(BaseModel):
    """Input schema for message passing operations."""
    action: str = Field(..., description="Action: send, get, ack, subscribe, unsubscribe, broadcast, stats")
    sender_id: str = Field("unknown", description="Sender agent ID")
    receiver_id: str = Field(None, description="Receiver agent ID")
    channel: str = Field(None, description="Communication channel")
    content: str = Field(None, description="Message content (JSON string)")
    subject: str = Field(None, description="Message subject")
    priority: str = Field("normal", description="Priority: low, normal, high, urgent")
    message_id: str = Field(None, description="Message ID for acknowledgment")
    limit: int = Field(10, description="Limit for message retrieval")


class MessagePassingTool(BaseTool):
    name: str = "message_passing"
    description: str = (
        "Send and receive messages between agents with channels, priorities, and acknowledgments."
    )
    args_schema: Type[BaseModel] = MessagePassingInput

    def _run(self, action: str, sender_id: str = "unknown", receiver_id: str = None,
             channel: str = None, content: str = None, subject: str = None,
             priority: str = "normal", message_id: str = None, limit: int = 10) -> str:
        # Initialize message bus inside _run to avoid Pydantic conflicts
        message_bus = MessageBus("message_bus.db")
        
        try:
            if action == "send":
                if not content:
                    return "❌ **Error**: Content required for send operation"
                if not receiver_id and not channel:
                    return "❌ **Error**: Either receiver_id or channel required"

                try:
                    parsed_content = json.loads(content)
                except json.JSONDecodeError:
                    parsed_content = content

                priority_map = {
                    "low": MessagePriority.LOW,
                    "normal": MessagePriority.NORMAL,
                    "high": MessagePriority.HIGH,
                    "urgent": MessagePriority.URGENT
                }

                msg_id = message_bus.send_message(
                    sender_id=sender_id,
                    content=parsed_content,
                    receiver_id=receiver_id,
                    channel=channel,
                    subject=subject,
                    priority=priority_map.get(priority, MessagePriority.NORMAL)
                )

                target = receiver_id or f"#{channel}"
                return f"📨 **Message Sent** to {target} (ID: {msg_id})"

            elif action == "get":
                messages = message_bus.get_messages(
                    agent_id=sender_id,
                    channel=channel,
                    limit=limit
                )

                if messages:
                    result = f"📬 **Messages for {sender_id}**:\n"
                    for msg in messages[:5]:  # Show only first 5
                        content_preview = str(msg['content'])[:50] + "..." if len(str(msg['content'])) > 50 else str(msg['content'])
                        result += f"- [{msg['priority']}] {msg['sender_id']}: {content_preview}\n"
                    if len(messages) > 5:
                        result += f"... and {len(messages) - 5} more messages"
                    return result
                return f"📬 **No messages** for {sender_id}"

            elif action == "ack":
                if not message_id:
                    return "❌ **Error**: message_id required for acknowledgment"

                success = message_bus.acknowledge_message(message_id, sender_id)
                return f"✅ **Acknowledged** message {message_id}" if success else f"❌ **Failed** to acknowledge {message_id}"

            elif action == "subscribe":
                if not channel:
                    return "❌ **Error**: Channel required for subscription"

                success = message_bus.subscribe_to_channel(sender_id, channel)
                return f"🔔 **Subscribed** to #{channel}" if success else f"❌ **Failed** to subscribe to #{channel}"

            elif action == "unsubscribe":
                if not channel:
                    return "❌ **Error**: Channel required for unsubscription"

                success = message_bus.unsubscribe_from_channel(sender_id, channel)
                return f"🔕 **Unsubscribed** from #{channel}" if success else f"❌ **Failed** to unsubscribe from #{channel}"

            elif action == "broadcast":
                if not channel or not content:
                    return "❌ **Error**: Channel and content required for broadcast"

                try:
                    parsed_content = json.loads(content)
                except json.JSONDecodeError:
                    parsed_content = content

                msg_id = message_bus.broadcast_message(
                    sender_id=sender_id,
                    channel=channel,
                    content=parsed_content,
                    subject=subject
                )
                return f"📢 **Broadcast** sent to #{channel} (ID: {msg_id})"

            elif action == "stats":
                stats = message_bus.get_message_stats()
                return f"""📊 **Message Statistics**:
- Total Messages: {stats.get('total_messages', 0)}
- Status Counts: {stats.get('status_counts', {})}
- Priority Counts: {stats.get('priority_counts', {})}
- Top Channels: {stats.get('top_channels', {})}"""

            else:
                return f"❌ **Invalid Action**: {action}. Use: send, get, ack, subscribe, unsubscribe, broadcast, stats"

        except Exception as e:
            return f"❌ **Message Error**: {str(e)}"


# Knowledge Store Tools

class KnowledgeStoreInput(BaseModel):
    """Input schema for knowledge store operations."""
    action: str = Field(..., description="Action: add, search, get, update, delete, categories, tags, stats")
    content: str = Field(None, description="Knowledge content")
    query: str = Field(None, description="Search query")
    knowledge_id: str = Field(None, description="Knowledge ID")
    agent_id: str = Field("unknown", description="Agent identifier")
    category: str = Field("general", description="Knowledge category")
    tags: str = Field(None, description="Comma-separated tags")
    n_results: int = Field(10, description="Number of search results")
    metadata: str = Field(None, description="Additional metadata (JSON string)")


class KnowledgeStoreTool(BaseTool):
    name: str = "knowledge_store"
    description: str = (
        "Store and search knowledge using semantic similarity. Supports categorization and tagging."
    )
    args_schema: Type[BaseModel] = KnowledgeStoreInput

    def _run(self, action: str, content: str = None, query: str = None,
             knowledge_id: str = None, agent_id: str = "unknown",
             category: str = "general", tags: str = None, n_results: int = 10,
             metadata: str = None) -> str:
        
        # Initialize knowledge store inside _run to avoid Pydantic conflicts
        try:
            knowledge_store = KnowledgeStore("./chroma_db")
        except ImportError:
            return "❌ **Error**: ChromaDB not available. Install with: pip install chromadb"

        try:
            if action == "add":
                if not content:
                    return "❌ **Error**: Content required for add operation"

                parsed_tags = [tag.strip() for tag in tags.split(",")] if tags else None
                parsed_metadata = json.loads(metadata) if metadata else None

                k_id = knowledge_store.add_knowledge(
                    content=content,
                    agent_id=agent_id,
                    category=category,
                    tags=parsed_tags,
                    metadata=parsed_metadata
                )

                return f"📚 **Knowledge Added** (ID: {k_id})" if k_id else "❌ **Failed** to add knowledge"

            elif action == "search":
                if not query:
                    return "❌ **Error**: Query required for search operation"

                results = knowledge_store.search_knowledge(
                    query=query,
                    n_results=n_results,
                    agent_id=agent_id if agent_id != "unknown" else None,
                    category=category if category != "general" else None
                )

                if results:
                    result_text = f"🔍 **Search Results for '{query}'**:\n"
                    for i, item in enumerate(results[:3]):  # Show top 3
                        content_preview = item['content'][:100] + "..." if len(item['content']) > 100 else item['content']
                        similarity = f"{item['similarity']:.2f}"
                        result_text += f"{i+1}. [{similarity}] {content_preview}\n"
                    if len(results) > 3:
                        result_text += f"... and {len(results) - 3} more results"
                    return result_text
                return f"🔍 **No results** found for '{query}'"

            elif action == "get":
                if not knowledge_id:
                    return "❌ **Error**: knowledge_id required for get operation"

                knowledge = knowledge_store.get_knowledge(knowledge_id)
                if knowledge:
                    return f"📖 **Knowledge ({knowledge_id})**:\n{knowledge['content']}\nCategory: {knowledge['metadata'].get('category', 'N/A')}"
                return f"❌ **Not Found**: Knowledge ID {knowledge_id}"

            elif action == "categories":
                categories = knowledge_store.list_categories()
                return f"📂 **Categories**: {', '.join(categories)}" if categories else "📂 **No categories** found"

            elif action == "tags":
                all_tags = knowledge_store.list_tags()
                return f"🏷️ **Tags**: {', '.join(all_tags)}" if all_tags else "🏷️ **No tags** found"

            elif action == "stats":
                stats = knowledge_store.get_stats()
                return f"""📊 **Knowledge Statistics**:
- Total Items: {stats.get('total_knowledge_items', 0)}
- Categories: {stats.get('categories', {})}
- Agents: {stats.get('agents', {})}"""

            else:
                return f"❌ **Invalid Action**: {action}. Use: add, search, get, categories, tags, stats"

        except Exception as e:
            return f"❌ **Knowledge Error**: {str(e)}"


# Team Communication Tool (High-level orchestrator)

class TeamCommunicationInput(BaseModel):
    """Input schema for team communication operations."""
    action: str = Field(..., description="Action: announce, status_update, coordination, decision, summary")
    agent_id: str = Field("unknown", description="Agent identifier")
    message: str = Field(..., description="Communication message")
    target: str = Field(None, description="Target agent or 'all' for team-wide")
    priority: str = Field("normal", description="Priority: low, normal, high, urgent")
    context: str = Field(None, description="Additional context or metadata")


class TeamCommunicationTool(BaseTool):
    name: str = "team_communication"
    description: str = (
        "High-level team communication for announcements, status updates, coordination, and decisions."
    )
    args_schema: Type[BaseModel] = TeamCommunicationInput

    def _run(self, action: str, agent_id: str, message: str, target: str = None,
             priority: str = "normal", context: str = None) -> str:
        
        # Initialize components inside _run to avoid Pydantic conflicts
        memory_manager = SharedMemoryManager("shared_memory.db")
        message_bus = MessageBus("message_bus.db")
        
        try:
            current_time = json.dumps({"timestamp": str(__import__('datetime').datetime.now()), "agent": agent_id})
            
            priority_map = {
                "low": MessagePriority.LOW,
                "normal": MessagePriority.NORMAL,
                "high": MessagePriority.HIGH,
                "urgent": MessagePriority.URGENT
            }

            if action == "announce":
                # Store in shared memory and broadcast
                memory_manager.set(f"announcement_{agent_id}", {
                    "message": message,
                    "context": context,
                    "timestamp": current_time
                }, agent_id)

                msg_id = message_bus.broadcast_message(
                    sender_id=agent_id,
                    channel="team_announcements",
                    content={"announcement": message, "context": context},
                    subject="Team Announcement",
                    priority=priority_map.get(priority, MessagePriority.NORMAL)
                )

                return f"📢 **Team Announcement** posted by {agent_id} (Message ID: {msg_id})"

            elif action == "status_update":
                # Update agent status in shared memory
                memory_manager.set(f"status_{agent_id}", {
                    "status": message,
                    "context": context,
                    "last_update": current_time
                }, agent_id)

                # Notify team if target is 'all'
                if target == "all":
                    msg_id = message_bus.broadcast_message(
                        sender_id=agent_id,
                        channel="team_status",
                        content={"status_update": message, "agent": agent_id},
                        subject=f"Status Update: {agent_id}",
                        priority=priority_map.get(priority, MessagePriority.NORMAL)
                    )
                    return f"📊 **Status Updated** and broadcast to team (Message ID: {msg_id})"
                else:
                    return f"📊 **Status Updated** for {agent_id}"

            elif action == "coordination":
                # Send coordination message
                if target and target != "all":
                    msg_id = message_bus.send_message(
                        sender_id=agent_id,
                        receiver_id=target,
                        content={"coordination_request": message, "context": context},
                        subject="Coordination Request",
                        priority=priority_map.get(priority, MessagePriority.NORMAL)
                    )
                    return f"🤝 **Coordination Request** sent to {target} (Message ID: {msg_id})"
                else:
                    msg_id = message_bus.broadcast_message(
                        sender_id=agent_id,
                        channel="team_coordination",
                        content={"coordination_request": message, "context": context},
                        subject="Team Coordination",
                        priority=priority_map.get(priority, MessagePriority.NORMAL)
                    )
                    return f"🤝 **Team Coordination** request broadcast (Message ID: {msg_id})"

            elif action == "decision":
                # Record decision in shared memory and notify team
                decision_id = f"decision_{agent_id}_{hash(message) % 10000}"
                memory_manager.set(decision_id, {
                    "decision": message,
                    "context": context,
                    "agent": agent_id,
                    "timestamp": current_time
                }, agent_id)

                msg_id = message_bus.broadcast_message(
                    sender_id=agent_id,
                    channel="team_decisions",
                    content={"decision": message, "context": context, "decision_id": decision_id},
                    subject="Team Decision",
                    priority=priority_map.get(priority, MessagePriority.HIGH)
                )

                return f"⚖️ **Decision Recorded** and broadcast (ID: {decision_id}, Message ID: {msg_id})"

            elif action == "summary":
                # Generate team summary from recent communications
                recent_messages = message_bus.get_messages(agent_id, limit=10)
                recent_count = len(recent_messages)
                
                # Get team status from shared memory
                status_keys = memory_manager.list_keys("status_")
                status_count = len(status_keys)

                summary_content = f"""## Team Communication Summary
- Recent Messages: {recent_count}
- Active Agent Statuses: {status_count}
- Summary generated by: {agent_id}
- Context: {context or 'General summary'}
"""

                # Store summary in shared memory
                memory_manager.set(f"summary_{agent_id}", {
                    "summary": summary_content,
                    "timestamp": current_time
                }, agent_id)

                return f"📋 **Team Summary Generated**:\n{summary_content}"

            else:
                return f"❌ **Invalid Action**: {action}. Use: announce, status_update, coordination, decision, summary"

        except Exception as e:
            return f"❌ **Team Communication Error**: {str(e)}"