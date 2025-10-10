from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from azure.storage.blob import BlobServiceClient

from agent_framework._workflows._checkpoint import WorkflowCheckpoint

logger = logging.getLogger(__name__)


class AzureBlobCheckpointStorage:
    """Checkpoint storage backed by Azure Blob Storage.

    This storage implements the same async interface as FileCheckpointStorage, so it
    can be used interchangeably by the workflow sample.
    """

    def __init__(self, connection_string: str, container_name: str = "checkpoints") -> None:
        self._client = BlobServiceClient.from_connection_string(connection_string)
        self._container_name = container_name
        self._container_client = self._client.get_container_client(container_name)
        try:
            self._container_client.create_container()
            logger.info(f"Created container: {container_name}")
        except Exception:
            # Container may already exist
            pass

    async def save_checkpoint(self, checkpoint: WorkflowCheckpoint) -> str:
        blob_name = f"{checkpoint.checkpoint_id}.json"
        data = json.dumps(checkpoint.to_dict(), ensure_ascii=False, indent=2).encode("utf-8")

        def _upload():
            blob_client = self._container_client.get_blob_client(blob_name)
            blob_client.upload_blob(data, overwrite=True)

        await asyncio.to_thread(_upload)
        logger.info(f"Saved checkpoint {checkpoint.checkpoint_id} to blob {blob_name}")
        return checkpoint.checkpoint_id

    async def load_checkpoint(self, checkpoint_id: str) -> WorkflowCheckpoint | None:
        blob_name = f"{checkpoint_id}.json"

        def _download() -> bytes | None:
            try:
                blob_client = self._container_client.get_blob_client(blob_name)
                stream = blob_client.download_blob()
                return stream.readall()
            except Exception:
                return None

        data = await asyncio.to_thread(_download)
        if data is None:
            return None
        checkpoint_dict = json.loads(data.decode("utf-8"))
        checkpoint = WorkflowCheckpoint.from_dict(checkpoint_dict)
        logger.info(f"Loaded checkpoint {checkpoint_id} from blob {blob_name}")
        return checkpoint

    async def list_checkpoint_ids(self, workflow_id: str | None = None) -> list[str]:
        def _list() -> list[str]:
            ids: list[str] = []
            for b in self._container_client.list_blobs(name_starts_with=""):
                if not b.name.endswith(".json"):
                    continue
                try:
                    blob_client = self._container_client.get_blob_client(b.name)
                    raw = blob_client.download_blob().readall()
                    obj = json.loads(raw.decode("utf-8"))
                    if workflow_id is None or obj.get("workflow_id") == workflow_id:
                        ids.append(obj.get("checkpoint_id", b.name[:-5]))
                except Exception:
                    logger.warning(f"Failed to read blob: {b.name}")
            return ids

        return await asyncio.to_thread(_list)

    async def list_checkpoints(self, workflow_id: str | None = None) -> list[WorkflowCheckpoint]:
        def _list() -> list[WorkflowCheckpoint]:
            cps: list[WorkflowCheckpoint] = []
            for b in self._container_client.list_blobs(name_starts_with=""):
                if not b.name.endswith(".json"):
                    continue
                try:
                    blob_client = self._container_client.get_blob_client(b.name)
                    raw = blob_client.download_blob().readall()
                    obj = json.loads(raw.decode("utf-8"))
                    if workflow_id is None or obj.get("workflow_id") == workflow_id:
                        cps.append(WorkflowCheckpoint.from_dict(obj))
                except Exception:
                    logger.warning(f"Failed to read blob: {b.name}")
            return cps

        return await asyncio.to_thread(_list)

    async def delete_checkpoint(self, checkpoint_id: str) -> bool:
        blob_name = f"{checkpoint_id}.json"

        def _delete() -> bool:
            try:
                blob_client = self._container_client.get_blob_client(blob_name)
                blob_client.delete_blob()
                return True
            except Exception:
                return False

        return await asyncio.to_thread(_delete)
