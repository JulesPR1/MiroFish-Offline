"""
Graph building service.
Uses GraphStorage (Neo4j) to replace Zep Cloud API.
"""

import time
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass

from ..config import Config
from ..models.task import TaskManager, TaskStatus
from ..storage import GraphStorage
from .text_processor import TextProcessor

logger = logging.getLogger('mirofish.graph_builder')


@dataclass
class GraphInfo:
    """Graph information"""
    graph_id: str
    node_count: int
    edge_count: int
    entity_types: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "graph_id": self.graph_id,
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "entity_types": self.entity_types,
        }


class GraphBuilderService:
    """
    Graph building service
    Build knowledge graph through GraphStorage interface
    """

    def __init__(self, storage: GraphStorage):
        self.storage = storage
        self.task_manager = TaskManager()
        self._parallel_batches = Config.NER_PARALLEL_BATCHES

    def build_graph_async(
        self,
        text: str,
        ontology: Dict[str, Any],
        graph_name: str = "MiroFish Graph",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        batch_size: int = None
    ) -> str:
        """
        Build graph asynchronously

        Args:
            text: Input text to process
            ontology: Ontology definition (from ontology generator output)
            graph_name: Name for the graph
            chunk_size: Text chunk size
            chunk_overlap: Chunk overlap size
            batch_size: Chunks per NER LLM call (defaults to Config.NER_BATCH_SIZE)

        Returns:
            Task ID
        """
        if batch_size is None:
            batch_size = Config.NER_BATCH_SIZE

        # Create task
        task_id = self.task_manager.create_task(
            task_type="graph_build",
            metadata={
                "graph_name": graph_name,
                "chunk_size": chunk_size,
                "text_length": len(text),
            }
        )

        # Execute build in background thread
        thread = threading.Thread(
            target=self._build_graph_worker,
            args=(task_id, text, ontology, graph_name, chunk_size, chunk_overlap, batch_size)
        )
        thread.daemon = True
        thread.start()

        return task_id

    def _build_graph_worker(
        self,
        task_id: str,
        text: str,
        ontology: Dict[str, Any],
        graph_name: str,
        chunk_size: int,
        chunk_overlap: int,
        batch_size: int
    ):
        """Graph build worker thread"""
        try:
            self.task_manager.update_task(
                task_id,
                status=TaskStatus.PROCESSING,
                progress=5,
                message="Starting graph building..."
            )

            # 1. Create graph
            graph_id = self.create_graph(graph_name)
            self.task_manager.update_task(
                task_id,
                progress=10,
                message=f"Graph created: {graph_id}"
            )

            # 2. Set ontology
            self.set_ontology(graph_id, ontology)
            self.task_manager.update_task(
                task_id,
                progress=15,
                message="Ontology set"
            )

            # 3. Text chunking
            chunks = TextProcessor.split_text(text, chunk_size, chunk_overlap)
            total_chunks = len(chunks)
            self.task_manager.update_task(
                task_id,
                progress=20,
                message=f"Text split into {total_chunks} chunks"
            )

            # 4. Send data in batches (NER + embedding + Neo4j insert — synchronous)
            episode_uuids = self.add_text_batches(
                graph_id, chunks, batch_size,
                lambda msg, prog: self.task_manager.update_task(
                    task_id,
                    progress=20 + int(prog * 0.6),  # 20-80%
                    message=msg
                )
            )

            # 5. Wait for processing (no-op for Neo4j — already synchronous)
            self.storage.wait_for_processing(episode_uuids)

            self.task_manager.update_task(
                task_id,
                progress=85,
                message="Data processing completed, getting graph information..."
            )

            # 6. Get graph information
            graph_info = self._get_graph_info(graph_id)

            # Completed
            self.task_manager.complete_task(task_id, {
                "graph_id": graph_id,
                "graph_info": graph_info.to_dict(),
                "chunks_processed": total_chunks,
            })

        except Exception as e:
            import traceback
            error_msg = f"{str(e)}\n{traceback.format_exc()}"
            self.task_manager.fail_task(task_id, error_msg)

    def create_graph(self, name: str) -> str:
        """Create graph"""
        return self.storage.create_graph(
            name=name,
            description="MiroFish Social Simulation Graph"
        )

    def set_ontology(self, graph_id: str, ontology: Dict[str, Any]):
        """
        SetGraphOntology

        Simply stores ontology as JSON in the Graph node.
        No more dynamic Pydantic class creation (was Zep-specific).
        The NER extractor reads this ontology to guide extraction.
        """
        self.storage.set_ontology(graph_id, ontology)

    def _process_one_batch(
        self,
        graph_id: str,
        batch_chunks: List[str],
        batch_num: int,
        total_batches: int,
    ) -> List[str]:
        """Process a single NER batch. Safe to call from multiple threads."""
        t0 = time.time()
        ids = self.storage.add_text_batch(
            graph_id,
            batch_chunks,
            batch_size=len(batch_chunks),
        )
        elapsed = time.time() - t0
        logger.info(
            f"[graph_build] Batch {batch_num}/{total_batches} "
            f"({len(batch_chunks)} chunks, {len(ids)} episodes) done in {elapsed:.1f}s"
        )
        return ids

    def add_text_batches(
        self,
        graph_id: str,
        chunks: List[str],
        batch_size: int = None,
        progress_callback: Optional[Callable] = None
    ) -> List[str]:
        """
        Add text to graph using batched NER extraction + optional parallelism.

        batch_size controls how many chunks go into a single NER LLM call.
        NER_PARALLEL_BATCHES controls how many batches are submitted concurrently.
        """
        if batch_size is None:
            batch_size = Config.NER_BATCH_SIZE

        total_chunks = len(chunks)
        batches = [chunks[i:i + batch_size] for i in range(0, total_chunks, batch_size)]
        total_batches = len(batches)

        logger.info(
            f"[graph_build] Starting: {total_chunks} chunks → {total_batches} NER batches "
            f"(batch_size={batch_size}, parallel={self._parallel_batches})"
        )

        results: Dict[int, List[str]] = {}
        lock = threading.Lock()
        completed = [0]

        def _submit(batch_num: int, batch_chunks: List[str]) -> tuple:
            ids = self._process_one_batch(graph_id, batch_chunks, batch_num, total_batches)
            with lock:
                completed[0] += 1
                done = completed[0]
            if progress_callback:
                progress_callback(
                    f"Processed batch {done}/{total_batches} ({len(batch_chunks)} chunks)...",
                    done / total_batches,
                )
            return batch_num, ids

        try:
            with ThreadPoolExecutor(max_workers=self._parallel_batches) as executor:
                futures = {
                    executor.submit(_submit, batch_num, batch_chunks): batch_num
                    for batch_num, batch_chunks in enumerate(batches, start=1)
                }
                for future in as_completed(futures):
                    batch_num, ids = future.result()
                    results[batch_num] = ids
        except Exception as e:
            logger.error(f"[graph_build] Batch processing FAILED: {e}")
            if progress_callback:
                progress_callback(f"Batch processing failed: {str(e)}", 0)
            raise

        # Reassemble in original order
        episode_uuids = []
        for batch_num in sorted(results.keys()):
            episode_uuids.extend(results[batch_num])

        logger.info(f"[graph_build] All {total_chunks} chunks processed, {len(episode_uuids)} episodes created")
        return episode_uuids

    def _get_graph_info(self, graph_id: str) -> GraphInfo:
        """Get graph information"""
        info = self.storage.get_graph_info(graph_id)
        return GraphInfo(
            graph_id=info["graph_id"],
            node_count=info["node_count"],
            edge_count=info["edge_count"],
            entity_types=info.get("entity_types", []),
        )

    def get_graph_data(self, graph_id: str) -> Dict[str, Any]:
        """Get complete graph data (including details)"""
        return self.storage.get_graph_data(graph_id)

    def delete_graph(self, graph_id: str):
        """Delete graph"""
        self.storage.delete_graph(graph_id)
