"""
Checkpoint workflow module for Azure Functions.
Refactored from checkpoint_with_resume_retries.py to be callable as a function.
"""

import asyncio
import os
import random
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List

from dotenv import load_dotenv

from agent_framework import (
    AgentExecutor,
    AgentExecutorRequest,
    AgentExecutorResponse,
    ChatMessage,
    Executor,
    FileCheckpointStorage,
    RequestInfoExecutor,
    Role,
    WorkflowBuilder,
    WorkflowContext,
    handler,
)
from agent_framework.azure import AzureOpenAIChatClient

# Import Azure Blob checkpoint storage
try:
    from azure_blob_checkpoint_storage import AzureBlobCheckpointStorage
except ImportError:
    AzureBlobCheckpointStorage = None


class SimulatedTransientError(Exception):
    """Custom exception to simulate transient network/service failures."""
    pass


class UpperCaseExecutor(Executor):
    """Uppercases the input text and persists both local and shared state."""

    @handler
    async def to_upper_case(self, text: str, ctx: WorkflowContext[str]) -> None:
        result = text.upper()
        print(f"UpperCaseExecutor: '{text}' -> '{result}'")

        # Persist executor-local state so it is captured in checkpoints
        prev = await ctx.get_state() or {}
        count = int(prev.get("count", 0)) + 1
        await ctx.set_state({
            "count": count,
            "last_input": text,
            "last_output": result,
        })

        # Write to shared_state so downstream executors and any resumed runs can read it.
        await ctx.set_shared_state("original_input", text)
        await ctx.set_shared_state("upper_output", result)

        # Send transformed text to the next executor.
        await ctx.send_message(result)


class SubmitToLowerAgent(Executor):
    """Builds an AgentExecutorRequest to send to the lowercasing agent."""

    def __init__(self, id: str, agent_id: str):
        super().__init__(id=id)
        self._agent_id = agent_id

    @handler
    async def submit(self, text: str, ctx: WorkflowContext[AgentExecutorRequest]) -> None:
        # Demonstrate reading shared_state written by UpperCaseExecutor.
        orig = await ctx.get_shared_state("original_input")
        upper = await ctx.get_shared_state("upper_output")
        print(f"LowerAgent (shared_state): original_input='{orig}', upper_output='{upper}'")

        # Build a minimal, deterministic prompt for the AgentExecutor.
        prompt = f"Convert the following text to lowercase. Return ONLY the transformed text.\n\nText: {text}"

        # Send to the AgentExecutor.
        await ctx.send_message(
            AgentExecutorRequest(messages=[ChatMessage(Role.USER, text=prompt)], should_respond=True),
            target_id=self._agent_id,
        )


class FinalizeFromAgent(Executor):
    """Consumes the AgentExecutorResponse and yields the final result."""

    @handler
    async def finalize(self, response: AgentExecutorResponse, ctx: WorkflowContext[Any, str]) -> None:
        result = response.agent_run_response.text or ""

        # Persist executor-local state for auditability when inspecting checkpoints.
        prev = await ctx.get_state() or {}
        count = int(prev.get("count", 0)) + 1
        await ctx.set_state({
            "count": count,
            "last_output": result,
            "final": True,
        })

        # Yield the final result so external consumers see the final value.
        await ctx.yield_output(result)


class ReverseTextExecutor(Executor):
    """Reverses the input text and persists local state. Simulates transient failures."""

    def __init__(self, id: str, failure_probability: float = 0.6):
        super().__init__(id=id)
        self._failure_probability = failure_probability

    @handler
    async def reverse_text(self, text: str, ctx: WorkflowContext[str]) -> None:
        # Simulate transient failure based on probability
        if random.random() < self._failure_probability:
            print(f"ReverseTextExecutor: SIMULATED FAILURE processing '{text}' (transient network error)")
            raise SimulatedTransientError(f"Transient failure in ReverseTextExecutor for input: {text}")
        
        result = text[::-1]
        print(f"ReverseTextExecutor: '{text}' -> '{result}' (SUCCESS)")

        # Persist executor-local state so checkpoint inspection can reveal progress.
        prev = await ctx.get_state() or {}
        count = int(prev.get("count", 0)) + 1
        await ctx.set_state({
            "count": count,
            "last_input": text,
            "last_output": result,
        })

        # Forward the reversed string to the next stage.
        await ctx.send_message(result)


def generate_execution_id() -> str:
    """Generate a unique execution identifier combining timestamp, process ID, and UUID."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    process_id = os.getpid()
    unique_id = str(uuid.uuid4())[:8]
    return f"{timestamp}_pid{process_id}_{unique_id}"


def create_process_safe_checkpoint_storage(execution_id: str) -> FileCheckpointStorage:
    """Create a process-safe checkpoint storage with the given execution identifier."""
    import tempfile
    
    # Use temp directory for Azure Functions (since we don't have write access to app directory)
    temp_dir = tempfile.gettempdir()
    checkpoint_base = os.path.join(temp_dir, "agent_framework_checkpoints")
    process_checkpoint_dir = os.path.join(checkpoint_base, execution_id)
    
    # Ensure directory exists
    os.makedirs(process_checkpoint_dir, exist_ok=True)
    
    print(f"[SECURITY] Process-safe file checkpoint storage")
    print(f"[FILE] Checkpoint directory: {process_checkpoint_dir}")
    
    return FileCheckpointStorage(storage_path=process_checkpoint_dir)


def create_workflow(checkpoint_storage, failure_probability: float = 0.6) -> Any:
    """Create the workflow with checkpointing enabled."""
    # Instantiate the pipeline executors.
    upper_case_executor = UpperCaseExecutor(id="upper-case")
    reverse_text_executor = ReverseTextExecutor(id="reverse-text", failure_probability=failure_probability)

    # Configure the agent stage that lowercases the text.
    api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    chat_client = AzureOpenAIChatClient(api_key=api_key)
    lower_agent = AgentExecutor(
        chat_client.create_agent(
            instructions=("You transform text to lowercase. Reply with ONLY the transformed text.")
        ),
        id="lower_agent",
    )

    # Bridge to the agent and terminalization stage.
    submit_lower = SubmitToLowerAgent(id="submit_lower", agent_id=lower_agent.id)
    finalize = FinalizeFromAgent(id="finalize")

    # Build the workflow with checkpointing enabled.
    return (
        WorkflowBuilder(max_iterations=5)
        .add_edge(upper_case_executor, reverse_text_executor)  # Uppercase -> Reverse
        .add_edge(reverse_text_executor, submit_lower)  # Reverse -> Build Agent request
        .add_edge(submit_lower, lower_agent)  # Submit to AgentExecutor
        .add_edge(lower_agent, finalize)  # Agent output -> Finalize
        .set_start_executor(upper_case_executor)  # Entry point
        .with_checkpointing(checkpoint_storage=checkpoint_storage)  # Enable persistence
        .build()
    )


async def run_workflow_with_retries(
    checkpoint_storage, 
    initial_message: str = "hello world",
    max_retries: int = 3,
    failure_probability: float = 0.6
) -> Dict[str, Any]:
    """Run workflow with automatic retry logic using checkpoints."""
    retry_backoff_seconds = 2.0  # Fixed backoff for Azure Functions
    
    workflow_outputs = []
    
    for attempt in range(1, max_retries + 1):
        print(f"\n{'='*50}")
        print(f"[RETRY] RETRY {attempt}/{max_retries}: Starting workflow execution")
        
        try:
            workflow = create_workflow(checkpoint_storage=checkpoint_storage, failure_probability=failure_probability)
            
            # Get existing checkpoints to potentially resume from
            existing_checkpoints = await checkpoint_storage.list_checkpoints()
            
            if attempt == 1 or not existing_checkpoints:
                # First attempt or no checkpoints - start fresh
                print(f"[START] Starting fresh workflow with message: '{initial_message}'")
                
                async for event in workflow.run_stream(message=initial_message):
                    print(f"Event: {event}")
                    if hasattr(event, 'data') and event.data is not None:
                        # Capture workflow outputs
                        if str(type(event).__name__) == 'WorkflowOutputEvent':
                            workflow_outputs.append(str(event.data))
            else:
                # Retry attempt - find the latest checkpoint to resume from
                latest_checkpoint = max(existing_checkpoints, key=lambda cp: cp.timestamp)
                print(f"[RESUME] Resuming from checkpoint: {latest_checkpoint.checkpoint_id[:8]}...")
                
                async for event in workflow.run_stream_from_checkpoint(
                    latest_checkpoint.checkpoint_id, 
                    checkpoint_storage=checkpoint_storage
                ):
                    print(f"Resume Event: {event}")
                    if hasattr(event, 'data') and event.data is not None:
                        # Capture workflow outputs
                        if str(type(event).__name__) == 'WorkflowOutputEvent':
                            workflow_outputs.append(str(event.data))
            
            # If we reach here, workflow completed successfully
            print(f"[SUCCESS] SUCCESS after {attempt} attempt{'s' if attempt > 1 else ''}!")
            final_checkpoints = await checkpoint_storage.list_checkpoints()
            
            return {
                "success": True,
                "attempts_used": attempt,
                "total_checkpoints": len(final_checkpoints),
                "final_output": workflow_outputs[-1] if workflow_outputs else None,
                "all_outputs": workflow_outputs,
                "checkpoints": [
                    {
                        "checkpoint_id": cp.checkpoint_id,
                        "timestamp": cp.timestamp,
                        "iteration_count": cp.iteration_count,
                        "workflow_id": cp.workflow_id
                    } 
                    for cp in final_checkpoints
                ]
            }
            
        except SimulatedTransientError as e:
            print(f"[WARNING] RETRY NEEDED ({attempt}/{max_retries}): {e}")
            
            if attempt < max_retries:
                print(f"[WAIT] Backing off for {retry_backoff_seconds}s before next attempt...")
                time.sleep(retry_backoff_seconds)
            else:
                print(f"[STOP] All {max_retries} attempts exhausted. Workflow permanently failed.")
                final_checkpoints = await checkpoint_storage.list_checkpoints()
                return {
                    "success": False,
                    "attempts_used": attempt,
                    "total_checkpoints": len(final_checkpoints),
                    "error": str(e),
                    "error_type": "SimulatedTransientError",
                    "checkpoints": [
                        {
                            "checkpoint_id": cp.checkpoint_id,
                            "timestamp": cp.timestamp,
                            "iteration_count": cp.iteration_count,
                            "workflow_id": cp.workflow_id
                        } 
                        for cp in final_checkpoints
                    ]
                }
        
        except Exception as e:
            print(f"[ERROR] Unexpected non-transient error on attempt {attempt}: {e}")
            final_checkpoints = await checkpoint_storage.list_checkpoints()
            return {
                "success": False,
                "attempts_used": attempt,
                "total_checkpoints": len(final_checkpoints),
                "error": str(e),
                "error_type": type(e).__name__,
                "checkpoints": [
                    {
                        "checkpoint_id": cp.checkpoint_id,
                        "timestamp": cp.timestamp,
                        "iteration_count": cp.iteration_count,
                        "workflow_id": cp.workflow_id
                    } 
                    for cp in final_checkpoints
                ]
            }
    
    # Should not reach here
    return {"success": False, "attempts_used": max_retries, "error": "Unknown error"}


async def run_checkpoint_workflow(
    initial_message: str = "hello world",
    use_azure_blob: bool = True,
    auto_cleanup: bool = False, 
    max_retries: int = 3,
    failure_probability: float = 0.6
) -> Dict[str, Any]:
    """
    Main function to run the checkpoint workflow.
    Returns a dictionary with execution results.
    """
    # Generate unique execution identifier for process isolation
    execution_id = generate_execution_id()
    print(f"[SECURITY] Execution ID: {execution_id}")
    
    # Create checkpoint storage based on parameters
    if use_azure_blob:
        if AzureBlobCheckpointStorage is None:
            raise RuntimeError("AzureBlobCheckpointStorage not available. Install azure-storage-blob package.")
        
        # Use AzureWebJobsStorage (standard Azure Functions storage connection)
        conn_str = os.getenv("AzureWebJobsStorage")
        if not conn_str:
            raise RuntimeError("AzureWebJobsStorage must be set to use Azure Blob storage")
        
        print(f"[CLOUD] Using Azure Blob Storage for checkpointing")
        checkpoint_storage = AzureBlobCheckpointStorage(conn_str, execution_prefix=execution_id)
    else:
        print(f"[FILE] Using local file storage for checkpointing")
        checkpoint_storage = create_process_safe_checkpoint_storage(execution_id)

    print(f"[TARGET] Multi-Process Safe Checkpoint-based Retry Demo (Failure Rate: {failure_probability*100:.0f}%)")
    
    # Run workflow with automatic retries
    result = await run_workflow_with_retries(
        checkpoint_storage, 
        initial_message, 
        max_retries=max_retries,
        failure_probability=failure_probability
    )
    
    # Add execution metadata
    result["execution_id"] = execution_id
    result["workflow_success"] = result.get("success", False)
    result["storage_type"] = "azure_blob" if use_azure_blob else "local_file"
    
    print(f"\n{'='*60}")
    
    if result["success"]:
        print("[SUCCESS] WORKFLOW COMPLETED SUCCESSFULLY!")
    else:
        print("[FAILED] WORKFLOW FAILED AFTER ALL RETRY ATTEMPTS")
    
    print(f"[STATS] Total checkpoints created: {result['total_checkpoints']}")
    print(f"[ATTEMPTS] Attempts used: {result['attempts_used']}")
    
    if result.get("final_output"):
        print(f"[OUTPUT] Final output: {result['final_output']}")
    
    # Handle cleanup if enabled
    if auto_cleanup and result["success"]:
        print("[CLEANUP] Auto-cleanup enabled for successful run...")
        # Note: Could implement cleanup here, but skipping for Functions to preserve debugging capability
    else:
        print("[INFO] Checkpoints preserved for debugging")
    
    print(f"{'='*60}")
    
    return result