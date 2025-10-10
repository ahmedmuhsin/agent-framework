# Copyright (c) Microsoft. All rights reserved.

import asyncio
import os
import random
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

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
from azure.identity import AzureCliCredential

# Optional: Azure Blob checkpoint storage
try:
    import sys
    from pathlib import Path
    # Add the checkpoint directory to sys.path to import the azure storage module
    checkpoint_dir = Path(__file__).parent
    if str(checkpoint_dir) not in sys.path:
        sys.path.insert(0, str(checkpoint_dir))
    from azure_blob_checkpoint_storage import AzureBlobCheckpointStorage  # type: ignore
except ImportError as e:
    print(f"Warning: Could not import AzureBlobCheckpointStorage: {e}")
    AzureBlobCheckpointStorage = None

from agent_framework.devui import serve
if TYPE_CHECKING:
    from agent_framework import Workflow
    from agent_framework._workflows._checkpoint import WorkflowCheckpoint

"""
Sample: Checkpointing and Resuming a Workflow with Automatic Retries

Purpose:
This sample demonstrates how to implement automatic retry logic using checkpoints when
workflow executors experience transient failures. It builds on the basic checkpointing
sample by adding:

1. Simulated intermittent failures in pipeline stages
2. Automatic failure detection and recovery
3. Intelligent checkpoint-based retry logic
4. Configurable retry attempts and backoff delays

Pipeline:
1) UpperCaseExecutor converts input to uppercase and records state.
2) ReverseTextExecutor reverses the string (WITH SIMULATED FAILURES).
3) SubmitToLowerAgent prepares an AgentExecutorRequest for the lowercasing agent.
4) lower_agent (AgentExecutor) converts text to lowercase via Azure OpenAI.
5) FinalizeFromAgent yields the final result.

What you learn:
- How to simulate and handle transient failures in workflow executors.
- How to implement automatic retry logic using existing checkpoints.
- How to configure retry behavior (max attempts, backoff delays).
- How to maintain workflow resilience without losing progress.
- How to log and track retry attempts for debugging and monitoring.

Prerequisites:
- Azure AI or Azure OpenAI available for AzureOpenAIChatClient.
- Authentication with azure-identity via AzureCliCredential. Run az login locally.
- Filesystem access for writing JSON checkpoint files in a temp directory.
"""

# Retry configuration
MAX_RETRY_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = 2.0
FAILURE_PROBABILITY = 0.6  # 60% chance of failure to demonstrate retry logic


def generate_execution_id() -> str:
    """
    Generate a unique execution identifier combining timestamp, process ID, and UUID.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    process_id = os.getpid()
    unique_id = str(uuid.uuid4())[:8]  # Short UUID for readability
    return f"{timestamp}_pid{process_id}_{unique_id}"


def create_process_safe_checkpoint_storage(execution_id: str) -> FileCheckpointStorage:
    """
    Create a process-safe checkpoint storage with the given execution identifier.
    """
    # Create process-specific checkpoint directory
    base_dir = os.path.dirname(__file__)
    checkpoint_base = os.path.join(base_dir, "tmp", "checkpoints")
    process_checkpoint_dir = os.path.join(checkpoint_base, execution_id)
    
    # Ensure directory exists
    os.makedirs(process_checkpoint_dir, exist_ok=True)
    
    print(f"🔐 Process-safe file checkpoint storage")
    print(f"📁 Checkpoint directory: {process_checkpoint_dir}")
    
    return FileCheckpointStorage(storage_path=process_checkpoint_dir)


def cleanup_checkpoint_storage(execution_id: str, success: bool) -> None:
    """
    Clean up checkpoint files after workflow completion.
    Optionally preserve files for failed runs for debugging.
    """
    base_dir = os.path.dirname(__file__)
    process_checkpoint_dir = os.path.join(base_dir, "tmp", "checkpoints", execution_id)
    
    if success:
        # Remove successful run checkpoints to save space
        try:
            import shutil
            if os.path.exists(process_checkpoint_dir):
                shutil.rmtree(process_checkpoint_dir)
                print(f"🧹 Cleaned up checkpoint directory: {execution_id}")
        except Exception as e:
            print(f"⚠️ Could not clean up checkpoints: {e}")
    else:
        # Preserve failed run checkpoints for debugging
        print(f"🔍 Preserving checkpoint directory for debugging: {execution_id}")
        print(f"   Location: {process_checkpoint_dir}")


async def cleanup_azure_checkpoint_storage(checkpoint_storage: Any, execution_id: str, success: bool) -> None:
    """
    Clean up Azure blob checkpoints after workflow completion.
    Optionally preserve blobs for failed runs for debugging.
    """
    if success:
        # Remove successful run checkpoints to save space
        try:
            checkpoints = await checkpoint_storage.list_checkpoints()
            deleted_count = 0
            for checkpoint in checkpoints:
                success = await checkpoint_storage.delete_checkpoint(checkpoint.checkpoint_id)
                if success:
                    deleted_count += 1
            print(f"🧹 Cleaned up {deleted_count} checkpoint blobs for execution: {execution_id}")
        except Exception as e:
            print(f"⚠️ Could not clean up checkpoint blobs: {e}")
    else:
        # Preserve failed run checkpoints for debugging
        checkpoints = await checkpoint_storage.list_checkpoints()
        print(f"🔍 Preserving {len(checkpoints)} checkpoint blobs for debugging: {execution_id}")
        print(f"   Blobs with prefix: {execution_id}/")


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
        # and available after resume for observability or logic.
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
    """Builds an AgentExecutorRequest to send to the lowercasing agent while keeping shared-state visibility."""

    def __init__(self, id: str, agent_id: str):
        super().__init__(id=id)
        self._agent_id = agent_id

    @handler
    async def submit(self, text: str, ctx: WorkflowContext[AgentExecutorRequest]) -> None:
        # Demonstrate reading shared_state written by UpperCaseExecutor.
        # Shared state survives across checkpoints and is visible to all executors.
        orig = await ctx.get_shared_state("original_input")
        upper = await ctx.get_shared_state("upper_output")
        print(f"LowerAgent (shared_state): original_input='{orig}', upper_output='{upper}'")

        # Build a minimal, deterministic prompt for the AgentExecutor.
        prompt = f"Convert the following text to lowercase. Return ONLY the transformed text.\n\nText: {text}"

        # Send to the AgentExecutor. should_respond=True instructs the agent to produce a reply.
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

    @handler
    async def reverse_text(self, text: str, ctx: WorkflowContext[str]) -> None:
        # Simulate transient failure based on probability
        if random.random() < FAILURE_PROBABILITY:
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


def create_workflow(checkpoint_storage: FileCheckpointStorage) -> "Workflow":
    # Instantiate the pipeline executors.
    upper_case_executor = UpperCaseExecutor(id="upper-case")
    reverse_text_executor = ReverseTextExecutor(id="reverse-text")

    # Configure the agent stage that lowercases the text.
    chat_client = AzureOpenAIChatClient(credential=AzureCliCredential())
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


def _render_checkpoint_summary(checkpoints: list["WorkflowCheckpoint"]) -> None:
    """Display human-friendly checkpoint metadata using framework summaries."""

    if not checkpoints:
        return

    print("\n📋 Checkpoint Summary:")
    for cp in sorted(checkpoints, key=lambda c: c.timestamp):
        summary = RequestInfoExecutor.checkpoint_summary(cp)
        msg_count = sum(len(v) for v in cp.messages.values())
        state_keys = sorted(cp.executor_states.keys())
        orig = cp.shared_state.get("original_input")
        upper = cp.shared_state.get("upper_output")

        line = (
            f"  🔸 {summary.checkpoint_id[:8]}... | iter={summary.iteration_count} | msg={msg_count} | states={len(state_keys)}"
        )
        if summary.status:
            line += f" | status={summary.status}"
        line += f"\n     shared_state: original='{orig}', upper='{upper}'"
        print(line)


def _log_retry_attempt(attempt: int, max_attempts: int, action: str) -> None:
    """Log retry attempt with clear visual indicators."""
    prefix = "🔄" if attempt < max_attempts else "🚨"
    print(f"{prefix} RETRY {attempt}/{max_attempts}: {action}")


def _log_failure(attempt: int, max_attempts: int, error: Exception) -> None:
    """Log failure with appropriate severity indicator."""
    if attempt < max_attempts:
        print(f"⚠️  RETRY NEEDED ({attempt}/{max_attempts}): {error}")
    else:
        print(f"💥 FINAL FAILURE ({attempt}/{max_attempts}): {error}")


def _log_success(attempt: int) -> None:
    """Log successful completion."""
    if attempt == 1:
        print("✅ SUCCESS on first attempt!")
    else:
        print(f"✅ SUCCESS after {attempt} attempts!")


async def run_workflow_with_retries(
    checkpoint_storage: FileCheckpointStorage, 
    initial_message: str = "hello world"
) -> tuple[bool, list["WorkflowCheckpoint"]]:
    """
    Run workflow with automatic retry logic using checkpoints.
    Returns (success, all_checkpoints).
    """
    
    for attempt in range(1, MAX_RETRY_ATTEMPTS + 1):
        print(f"\n{'='*50}")
        _log_retry_attempt(attempt, MAX_RETRY_ATTEMPTS, "Starting workflow execution")
        
        try:
            workflow = create_workflow(checkpoint_storage=checkpoint_storage)
            
            # Get existing checkpoints to potentially resume from
            existing_checkpoints = await checkpoint_storage.list_checkpoints()
            
            if attempt == 1 or not existing_checkpoints:
                # First attempt or no checkpoints - start fresh
                print(f"🚀 Starting fresh workflow with message: '{initial_message}'")
                
                async for event in workflow.run_stream(message=initial_message):
                    print(f"Event: {event}")
            else:
                # Retry attempt - find the latest checkpoint to resume from
                latest_checkpoint = max(existing_checkpoints, key=lambda cp: cp.timestamp)
                print(f"🔄 Resuming from checkpoint: {latest_checkpoint.checkpoint_id[:8]}...")
                
                async for event in workflow.run_stream_from_checkpoint(
                    latest_checkpoint.checkpoint_id, 
                    checkpoint_storage=checkpoint_storage
                ):
                    print(f"Resume Event: {event}")
            
            # If we reach here, workflow completed successfully
            _log_success(attempt)
            final_checkpoints = await checkpoint_storage.list_checkpoints()
            return True, final_checkpoints
            
        except SimulatedTransientError as e:
            _log_failure(attempt, MAX_RETRY_ATTEMPTS, e)
            
            if attempt < MAX_RETRY_ATTEMPTS:
                print(f"⏳ Backing off for {RETRY_BACKOFF_SECONDS}s before next attempt...")
                time.sleep(RETRY_BACKOFF_SECONDS)
            else:
                print(f"� All {MAX_RETRY_ATTEMPTS} attempts exhausted. Workflow permanently failed.")
                final_checkpoints = await checkpoint_storage.list_checkpoints()
                return False, final_checkpoints
        
        except Exception as e:
            print(f"💀 Unexpected non-transient error on attempt {attempt}: {e}")
            final_checkpoints = await checkpoint_storage.list_checkpoints()
            return False, final_checkpoints
    
    # Should not reach here
    final_checkpoints = await checkpoint_storage.list_checkpoints()
    return False, final_checkpoints


async def main():
    # Load environment variables from .env file
    load_dotenv()
    
    # Generate unique execution identifier for process isolation
    execution_id = generate_execution_id()
    print(f"🔐 Execution ID: {execution_id}")
    
    # Create checkpoint storage based on configuration
    use_azure = os.getenv("USE_AZURE_BLOB", "0") == "1"

    if use_azure:
        if AzureBlobCheckpointStorage is None:
            raise RuntimeError("AzureBlobCheckpointStorage not available. Install azure-storage-blob package.")
        
        conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        if not conn_str:
            raise RuntimeError("AZURE_STORAGE_CONNECTION_STRING must be set to use Azure Blob storage")
        
        print(f"� Using Azure Blob Storage for checkpointing")
        print(f" Connection: {conn_str[:50]}...")
        
        # Use execution_id as blob prefix to isolate checkpoints per process
        checkpoint_storage = AzureBlobCheckpointStorage(conn_str, execution_prefix=execution_id)
    else:
        print(f"📁 Using local file storage for checkpointing")
        checkpoint_storage = create_process_safe_checkpoint_storage(execution_id)

    print(f"🎯 Multi-Process Safe Checkpoint-based Retry Demo (Failure Rate: {FAILURE_PROBABILITY*100:.0f}%)")
    
    # Run workflow with automatic retries
    success, all_checkpoints = await run_workflow_with_retries(checkpoint_storage)

    # Display final results and checkpoint summary
    print(f"\n{'='*60}")
    
    if success:
        print("🎉 WORKFLOW COMPLETED SUCCESSFULLY!")
    else:
        print("💥 WORKFLOW FAILED AFTER ALL RETRY ATTEMPTS")
    
    if all_checkpoints:
        print(f"📊 Total checkpoints created: {len(all_checkpoints)}")
        _render_checkpoint_summary(all_checkpoints)
        
        # Demonstrate manual checkpoint inspection and resume capability
        workflow_id = all_checkpoints[0].workflow_id
        sorted_cps = sorted([cp for cp in all_checkpoints if cp.workflow_id == workflow_id], key=lambda c: c.timestamp)
        
        print(f"\n🔍 Available checkpoints for manual inspection:")
        for idx, cp in enumerate(sorted_cps):
            summary = RequestInfoExecutor.checkpoint_summary(cp)
            line = f"  [{idx}] {summary.checkpoint_id[:8]}... iter={summary.iteration_count}"
            if summary.status:
                line += f" status={summary.status}"
            msg_count = sum(len(v) for v in cp.messages.values())
            line += f" messages={msg_count}"
            print(line)
            
        print(f"\n💡 You can manually resume from any checkpoint using:")
        print(f"   workflow.run_stream_from_checkpoint(checkpoint_id, checkpoint_storage)")
    else:
        print("⚠️ No checkpoints were created")
    
    print(f"{'='*60}")
    
    # Clean up checkpoint storage based on success/failure (optional)
    auto_cleanup = os.getenv("AUTO_CLEANUP_CHECKPOINTS", "0") == "1"
    if auto_cleanup:
        print("🧹 Auto-cleanup enabled...")
        if use_azure:
            await cleanup_azure_checkpoint_storage(checkpoint_storage, execution_id, success)
        else:
            cleanup_checkpoint_storage(execution_id, success)
    else:
        print("ℹ️ Auto-cleanup disabled. Checkpoints preserved.")
        if use_azure:
            checkpoints = await checkpoint_storage.list_checkpoints()
            print(f"   Azure blobs with prefix '{execution_id}': {len(checkpoints)} checkpoints")
        else:
            print(f"   Local directory: {execution_id}")
        print("   Set AUTO_CLEANUP_CHECKPOINTS=1 to enable automatic cleanup.")

    """
    Sample Output:

    Running workflow with initial message...
    UpperCaseExecutor: 'hello world' -> 'HELLO WORLD'
    Event: ExecutorInvokeEvent(executor_id=upper_case_executor)
    Event: ExecutorCompletedEvent(executor_id=upper_case_executor)
    ReverseTextExecutor: 'HELLO WORLD' -> 'DLROW OLLEH'
    Event: ExecutorInvokeEvent(executor_id=reverse_text_executor)
    Event: ExecutorCompletedEvent(executor_id=reverse_text_executor)
    LowerAgent (shared_state): original_input='hello world', upper_output='HELLO WORLD'
    Event: ExecutorInvokeEvent(executor_id=submit_lower)
    Event: ExecutorInvokeEvent(executor_id=lower_agent)
    Event: ExecutorInvokeEvent(executor_id=finalize)

    Checkpoint summary:
    - dfc63e72-8e8d-454f-9b6d-0d740b9062e6 | label='after_initial_execution' | iter=0 | messages=1 | states=['upper_case_executor'] | shared_state: original_input='hello world', upper_output='HELLO WORLD'
    - a78c345a-e5d9-45ba-82c0-cb725452d91b | label='superstep_1' | iter=1 | messages=1 | states=['reverse_text_executor', 'upper_case_executor'] | shared_state: original_input='hello world', upper_output='HELLO WORLD'
    - 637c1dbd-a525-4404-9583-da03980537a2 | label='superstep_2' | iter=2 | messages=0 | states=['finalize', 'lower_agent', 'reverse_text_executor', 'submit_lower', 'upper_case_executor'] | shared_state: original_input='hello world', upper_output='HELLO WORLD'

    Available checkpoints to resume from:
        [0] id=dfc63e72-... iter=0 messages=1 label='after_initial_execution'
        [1] id=a78c345a-... iter=1 messages=1 label='superstep_1'
        [2] id=637c1dbd-... iter=2 messages=0 label='superstep_2'

    Enter checkpoint index (or paste checkpoint id) to resume from, or press Enter to skip resume: 1

    Resuming from checkpoint: a78c345a-e5d9-45ba-82c0-cb725452d91b
    LowerAgent (shared_state): original_input='hello world', upper_output='HELLO WORLD'
    Resumed Event: ExecutorInvokeEvent(executor_id=submit_lower)
    Resumed Event: ExecutorInvokeEvent(executor_id=lower_agent)
    Resumed Event: ExecutorInvokeEvent(executor_id=finalize)
    """  # noqa: E501


if __name__ == "__main__":
    asyncio.run(main())
