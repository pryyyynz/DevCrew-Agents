"""Knowledge Store using ChromaDB for semantic search and retrieval."""

import json
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    print("ChromaDB not available. Install with: pip install chromadb")


class KnowledgeStore:
    """Vector-based knowledge store for agent memory and context."""

    def __init__(self, persist_directory: str = "./chroma_db", collection_name: str = "agent_knowledge"):
        if not CHROMADB_AVAILABLE:
            raise ImportError(
                "ChromaDB is required but not installed. Run: pip install chromadb")

        self.persist_directory = persist_directory
        self.collection_name = collection_name

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        # Get or create collection - fix the error handling
        try:
            self.collection = self.client.get_collection(name=collection_name)
        # Catch any exception (including InvalidCollectionException)
        except Exception:
            # Collection doesn't exist, create it
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"description": "Agent knowledge and context storage"}
            )

    def add_knowledge(
        self,
        content: str,
        metadata: Dict[str, Any] = None,
        knowledge_id: str = None,
        agent_id: str = "unknown",
        category: str = "general",
        tags: List[str] = None
    ) -> str:
        """Add knowledge to the store."""
        if not knowledge_id:
            knowledge_id = str(uuid.uuid4())

        # Prepare metadata
        full_metadata = {
            "agent_id": agent_id,
            "category": category,
            "created_at": datetime.now().isoformat(),
            "tags": json.dumps(tags or []),
            **(metadata or {})
        }

        try:
            self.collection.add(
                documents=[content],
                metadatas=[full_metadata],
                ids=[knowledge_id]
            )
            return knowledge_id
        except Exception as e:
            print(f"Error adding knowledge: {e}")
            return None

    def search_knowledge(
        self,
        query: str,
        n_results: int = 10,
        agent_id: str = None,
        category: str = None,
        tags: List[str] = None,
        min_similarity: float = 0.0
    ) -> List[Dict[str, Any]]:
        """Search knowledge using semantic similarity."""
        try:
            # Build where clause for filtering
            where_clause = {}
            if agent_id:
                where_clause["agent_id"] = agent_id
            if category:
                where_clause["category"] = category

            # Perform search
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where_clause if where_clause else None
            )

            # Format results
            knowledge_items = []
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    metadata = results['metadatas'][0][i] if results['metadatas'] else {
                    }
                    distance = results['distances'][0][i] if results['distances'] else 0
                    similarity = 1 - distance  # Convert distance to similarity

                    if similarity >= min_similarity:
                        # Parse tags back from JSON
                        tags_json = metadata.get('tags', '[]')
                        try:
                            parsed_tags = json.loads(tags_json)
                        except:
                            parsed_tags = []

                        # Filter by tags if specified
                        if tags and not any(tag in parsed_tags for tag in tags):
                            continue

                        knowledge_items.append({
                            'id': results['ids'][0][i],
                            'content': doc,
                            'metadata': metadata,
                            'similarity': similarity,
                            'tags': parsed_tags
                        })

            return knowledge_items
        except Exception as e:
            print(f"Error searching knowledge: {e}")
            return []

    def get_knowledge(self, knowledge_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve specific knowledge by ID."""
        try:
            results = self.collection.get(ids=[knowledge_id])

            if results['documents'] and results['documents'][0]:
                metadata = results['metadatas'][0] if results['metadatas'] else {
                }
                tags_json = metadata.get('tags', '[]')
                try:
                    parsed_tags = json.loads(tags_json)
                except:
                    parsed_tags = []

                return {
                    'id': knowledge_id,
                    'content': results['documents'][0],
                    'metadata': metadata,
                    'tags': parsed_tags
                }
            return None
        except Exception as e:
            print(f"Error getting knowledge: {e}")
            return None

    def update_knowledge(
        self,
        knowledge_id: str,
        content: str = None,
        metadata: Dict[str, Any] = None,
        tags: List[str] = None
    ) -> bool:
        """Update existing knowledge."""
        try:
            # Get current knowledge
            current = self.get_knowledge(knowledge_id)
            if not current:
                return False

            # Prepare updates
            new_content = content if content is not None else current['content']
            new_metadata = current['metadata'].copy()
            if metadata:
                new_metadata.update(metadata)
            if tags is not None:
                new_metadata['tags'] = json.dumps(tags)
            new_metadata['updated_at'] = datetime.now().isoformat()

            # Update in collection
            self.collection.update(
                ids=[knowledge_id],
                documents=[new_content],
                metadatas=[new_metadata]
            )
            return True
        except Exception as e:
            print(f"Error updating knowledge: {e}")
            return False

    def delete_knowledge(self, knowledge_id: str) -> bool:
        """Delete knowledge by ID."""
        try:
            self.collection.delete(ids=[knowledge_id])
            return True
        except Exception as e:
            print(f"Error deleting knowledge: {e}")
            return False

    def list_categories(self) -> List[str]:
        """Get all unique categories in the knowledge store."""
        try:
            results = self.collection.get()
            categories = set()
            if results['metadatas']:
                for metadata in results['metadatas']:
                    category = metadata.get('category', 'general')
                    categories.add(category)
            return list(categories)
        except Exception as e:
            print(f"Error listing categories: {e}")
            return []

    def list_tags(self) -> List[str]:
        """Get all unique tags in the knowledge store."""
        try:
            results = self.collection.get()
            all_tags = set()
            if results['metadatas']:
                for metadata in results['metadatas']:
                    tags_json = metadata.get('tags', '[]')
                    try:
                        tags = json.loads(tags_json)
                        all_tags.update(tags)
                    except:
                        continue
            return list(all_tags)
        except Exception as e:
            print(f"Error listing tags: {e}")
            return []

    def get_knowledge_by_agent(
        self,
        agent_id: str,
        category: str = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get all knowledge created by a specific agent."""
        try:
            where_clause = {"agent_id": agent_id}
            if category:
                where_clause["category"] = category

            results = self.collection.get(
                where=where_clause,
                limit=limit
            )

            knowledge_items = []
            if results['documents']:
                for i, doc in enumerate(results['documents']):
                    metadata = results['metadatas'][i] if results['metadatas'] else {
                    }
                    tags_json = metadata.get('tags', '[]')
                    try:
                        parsed_tags = json.loads(tags_json)
                    except:
                        parsed_tags = []

                    knowledge_items.append({
                        'id': results['ids'][i],
                        'content': doc,
                        'metadata': metadata,
                        'tags': parsed_tags
                    })

            return knowledge_items
        except Exception as e:
            print(f"Error getting knowledge by agent: {e}")
            return []

    def add_conversation_context(
        self,
        conversation_id: str,
        agent_id: str,
        context: str,
        turn_number: int = 0,
        metadata: Dict[str, Any] = None
    ) -> str:
        """Add conversation context to knowledge store."""
        full_metadata = {
            "conversation_id": conversation_id,
            "turn_number": turn_number,
            "context_type": "conversation",
            **(metadata or {})
        }

        return self.add_knowledge(
            content=context,
            metadata=full_metadata,
            agent_id=agent_id,
            category="conversation"
        )

    def search_conversation_context(
        self,
        query: str,
        conversation_id: str = None,
        n_results: int = 5
    ) -> List[Dict[str, Any]]:
        """Search conversation context."""
        where_clause = {"context_type": "conversation"}
        if conversation_id:
            where_clause["conversation_id"] = conversation_id

        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where_clause
            )

            context_items = []
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    metadata = results['metadatas'][0][i] if results['metadatas'] else {
                    }
                    distance = results['distances'][0][i] if results['distances'] else 0

                    context_items.append({
                        'id': results['ids'][0][i],
                        'content': doc,
                        'metadata': metadata,
                        'similarity': 1 - distance,
                        'conversation_id': metadata.get('conversation_id'),
                        'turn_number': metadata.get('turn_number', 0)
                    })

            return context_items
        except Exception as e:
            print(f"Error searching conversation context: {e}")
            return []

    def add_decision_context(
        self,
        decision: str,
        rationale: str,
        agent_id: str,
        outcome: str = None,
        metadata: Dict[str, Any] = None
    ) -> str:
        """Add decision-making context to knowledge store."""
        content = f"Decision: {decision}\nRationale: {rationale}"
        if outcome:
            content += f"\nOutcome: {outcome}"

        full_metadata = {
            "decision": decision,
            "rationale": rationale,
            "outcome": outcome,
            "context_type": "decision",
            **(metadata or {})
        }

        return self.add_knowledge(
            content=content,
            metadata=full_metadata,
            agent_id=agent_id,
            category="decisions"
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge store."""
        try:
            results = self.collection.get()
            total_items = len(results['ids']) if results['ids'] else 0

            # Count by category
            category_counts = {}
            # Count by agent
            agent_counts = {}

            if results['metadatas']:
                for metadata in results['metadatas']:
                    category = metadata.get('category', 'general')
                    agent_id = metadata.get('agent_id', 'unknown')

                    category_counts[category] = category_counts.get(
                        category, 0) + 1
                    agent_counts[agent_id] = agent_counts.get(agent_id, 0) + 1

            return {
                'total_knowledge_items': total_items,
                'categories': category_counts,
                'agents': agent_counts,
                'collection_name': self.collection_name
            }
        except Exception as e:
            print(f"Error getting knowledge store stats: {e}")
            return {}

    def clear_all(self, confirm: bool = False) -> bool:
        """Clear all knowledge (use with extreme caution)."""
        if not confirm:
            raise ValueError("Must set confirm=True to clear all knowledge")

        try:
            # Delete the collection and recreate it
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "Agent knowledge and context storage"}
            )
            return True
        except Exception as e:
            print(f"Error clearing knowledge store: {e}")
            return False
