# Azure Functions Checkpoint Workflow

This directory contains an Azure Functions implementation of the checkpoint workflow sample using the Azure Functions v2 programming model.

## Overview

The Azure Functions app wraps the checkpoint workflow with HTTP endpoints, making it accessible as a serverless API. This enables:

- **Serverless Execution**: Run workflows on-demand without managing infrastructure
- **HTTP API**: Simple REST interface for triggering workflows
- **Multi-Process Safety**: Each execution gets a unique ID for isolated checkpoint storage
- **Azure Storage**: Uses Azure Blob Storage for checkpoint persistence (with Azurite for local development)
- **Retry Logic**: Automatic retry with exponential backoff on transient failures

## Files

- `function_app.py`: Main Azure Functions app with HTTP triggers
- `checkpoint_workflow.py`: Core workflow logic refactored for function calls
- `azure_blob_checkpoint_storage.py`: Custom checkpoint storage for Azure Blob Storage (copy from parent directory)
- `requirements.txt`: Python dependencies
- `setup_credentials.py`: Helper script for interactive credential setup
- `local.settings.json`: Local development configuration

## Authentication Setup

The Azure Functions app uses API key authentication for Azure OpenAI instead of Azure CLI credentials for better reliability in serverless environments.

### Option 1: Interactive Setup

Run the credential setup script:

```bash
python setup_credentials.py
```

This will prompt you for:
- Azure OpenAI endpoint URL
- Chat deployment name (e.g., gpt-4)
- API key

### Option 2: Manual Setup

Edit `local.settings.json` and update these values:

```json
{
  "Values": {
    "AZURE_OPENAI_ENDPOINT": "https://your-resource.openai.azure.com/",
    "AZURE_OPENAI_CHAT_DEPLOYMENT_NAME": "gpt-4",
    "AZURE_OPENAI_API_KEY": "your-actual-api-key"
  }
}
```

**Important**: Never commit your actual API key to version control!
- `host.json`: Azure Functions host configuration
- `local.settings.json`: Local development settings

## Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Environment**:
   Update `local.settings.json` with your Azure OpenAI settings:
   ```json
   {
     "Values": {
       "AZURE_OPENAI_ENDPOINT": "https://your-openai-resource.openai.azure.com/",
       "AZURE_OPENAI_DEPLOYMENT": "gpt-4"
     }
   }
   ```

4. **Start Azurite** (for local development):
   ```bash
   azurite --silent --location c:\azurite --debug c:\azurite\debug.log
   ```

5. **Run Locally**:
   ```bash
   func start
   ```

## API Endpoints

### POST /api/run_workflow

Executes the checkpoint workflow with retry logic.

**Request Body**:
```json
{
  "message": "hello world",
  "use_azure_blob": true,
  "auto_cleanup": false,
  "max_retries": 3,
  "failure_probability": 0.6
}
```

**Response**:
```json
{
  "success": true,
  "execution_id": "20240101_120000_pid1234_abcd1234",
  "attempts_used": 2,
  "total_checkpoints": 4,
  "final_output": "dlrow olleh",
  "storage_type": "azure_blob",
  "checkpoints": [
    {
      "checkpoint_id": "checkpoint_1",
      "timestamp": "2024-01-01T12:00:00.000Z",
      "iteration_count": 1,
      "workflow_id": "workflow_1"
    }
  ]
}
```

### GET /api/health

Health check endpoint.

**Response**:
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T12:00:00.000Z",
  "version": "1.0.0"
}
```

### GET /api/info

API documentation endpoint.

**Response**:
```json
{
  "name": "Checkpoint Workflow API",
  "version": "1.0.0",
  "description": "Azure Functions wrapper for checkpoint-enabled agent workflow with retry logic",
  "endpoints": {
    "/api/run_workflow": "POST - Execute workflow with retry logic",
    "/api/health": "GET - Health check",
    "/api/info": "GET - API information"
  }
}
```

## Configuration

Environment variables (set in `local.settings.json` for local development):

- `AzureWebJobsStorage`: Azure Storage connection string (default: "UseDevelopmentStorage=true" for Azurite)
- `AZURE_OPENAI_ENDPOINT`: Azure OpenAI service endpoint
- `AZURE_OPENAI_DEPLOYMENT`: Azure OpenAI deployment name

Parameters are passed via HTTP request body instead of environment variables:
- `use_azure_blob`: Use Azure Blob Storage (default: true)
- `failure_probability`: Simulated failure rate (default: 0.6)
- `max_retries`: Maximum retry attempts (default: 3)
- `auto_cleanup`: Auto-cleanup on success (default: false)

## Deployment

1. **Create Function App**:
   ```bash
   az functionapp create --resource-group myResourceGroup --consumption-plan-location eastus --runtime python --runtime-version 3.11 --functions-version 4 --name myCheckpointWorkflow --storage-account mystorageaccount
   ```

2. **Deploy**:
   ```bash
   func azure functionapp publish myCheckpointWorkflow
   ```

3. **Configure App Settings**:
   ```bash
   az functionapp config appsettings set --name myCheckpointWorkflow --resource-group myResourceGroup --settings AZURE_OPENAI_ENDPOINT="https://your-openai.openai.azure.com/" AZURE_OPENAI_DEPLOYMENT="gpt-4"
   ```

## Testing

Test the deployed function:

```bash
# Health check
curl https://myCheckpointWorkflow.azurewebsites.net/api/health

# Run workflow
curl -X POST https://myCheckpointWorkflow.azurewebsites.net/api/run_workflow \
  -H "Content-Type: application/json" \
  -d '{"message": "hello azure functions", "max_retries": 2}'
```

## Features

- **Process Isolation**: Each execution gets a unique ID (timestamp_pid_uuid)
- **Automatic Retries**: Configurable retry logic with exponential backoff
- **Checkpoint Resume**: Failed workflows resume from the last successful checkpoint
- **Azure Blob Storage**: Persistent, scalable checkpoint storage
- **Comprehensive Logging**: Detailed execution logs for debugging
- **Health Monitoring**: Built-in health check endpoint
- **API Documentation**: Self-documenting API with info endpoint

## Architecture

The workflow processes text through multiple stages:

1. **UpperCaseExecutor**: Converts input to uppercase
2. **ReverseTextExecutor**: Reverses the text (with simulated failures)
3. **SubmitToLowerAgent**: Builds request for AI agent
4. **AgentExecutor**: Uses Azure OpenAI to convert to lowercase
5. **FinalizeFromAgent**: Extracts and yields final result

Each stage maintains executor-local state and contributes to shared state, enabling precise checkpoint resume on failures.