import azure.functions as func
import logging
import json
import asyncio
import os
import sys
from pathlib import Path

# Add the parent directory to sys.path to import our workflow modules
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

# Test import to catch any import errors early
try:
    from checkpoint_workflow import run_checkpoint_workflow
    logging.info("Successfully imported checkpoint_workflow module")
except ImportError as e:
    logging.error(f"Failed to import checkpoint_workflow: {e}")
    # Don't raise here, just log it so other functions still work
    run_checkpoint_workflow = None

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

@app.route(route="test", methods=["GET", "POST"])
def test_function(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Test function processed a request.')
    
    return func.HttpResponse(
        json.dumps({
            "message": "Test function is working!",
            "method": req.method,
            "url": req.url
        }),
        status_code=200,
        mimetype="application/json"
    )


@app.route(route="run_workflow", methods=["GET", "POST"])
def run_workflow_http(req: func.HttpRequest) -> func.HttpResponse:
    """
    HTTP trigger to run the checkpoint workflow with retry logic.
    
    Expected JSON payload:
    {
        "message": "hello world",
        "use_azure_blob": true,
        "auto_cleanup": false,
        "max_retries": 3,
        "failure_probability": 0.6
    }
    """
    logging.info('Checkpoint workflow HTTP trigger function processed a request.')
    
    if run_checkpoint_workflow is None:
        logging.error("checkpoint_workflow module not available")
        return func.HttpResponse(
            json.dumps({"error": "Workflow module not available - import failed"}),
            status_code=500,
            mimetype="application/json"
        )
    
    try:
        # Handle both GET and POST requests
        if req.method == "GET":
            # For GET requests, use query parameters or defaults
            message = req.params.get("message", "hello world")
            use_azure_blob = req.params.get("use_azure_blob", "true").lower() == "true"
            auto_cleanup = req.params.get("auto_cleanup", "false").lower() == "true" 
            max_retries = int(req.params.get("max_retries", "3"))
            failure_probability = float(req.params.get("failure_probability", "0.3"))  # Lower default for GET testing
            
            logging.info(f'GET request - using query params and defaults')
        else:
            # For POST requests, parse JSON body
            try:
                req_body = req.get_json()
            except ValueError:
                logging.error("Invalid JSON in request body")
                return func.HttpResponse(
                    json.dumps({"error": "Invalid JSON in request body"}),
                    status_code=400,
                    mimetype="application/json"
                )
            
            if not req_body:
                logging.warning("Request body is empty, using defaults")
                req_body = {}
            
            # Extract parameters with defaults
            message = req_body.get("message", "hello world")
            use_azure_blob = req_body.get("use_azure_blob", True)
            auto_cleanup = req_body.get("auto_cleanup", False)
            max_retries = req_body.get("max_retries", 3)
            failure_probability = req_body.get("failure_probability", 0.6)
        
        logging.info(f'Extracted parameters: message="{message}", use_azure_blob={use_azure_blob}, max_retries={max_retries}')
        
        # Validate parameters
        if not isinstance(message, str) or not message.strip():
            return func.HttpResponse(
                json.dumps({"error": "Message must be a non-empty string"}),
                status_code=400,
                mimetype="application/json"
            )
        
        if not isinstance(max_retries, int) or max_retries < 1 or max_retries > 10:
            return func.HttpResponse(
                json.dumps({"error": "max_retries must be an integer between 1 and 10"}),
                status_code=400,
                mimetype="application/json"
            )
        
        if not isinstance(failure_probability, (int, float)) or failure_probability < 0 or failure_probability > 1:
            return func.HttpResponse(
                json.dumps({"error": "failure_probability must be a number between 0 and 1"}),
                status_code=400,
                mimetype="application/json"
            )
        
        # Run the async workflow function synchronously using asyncio.run()
        logging.info(f'About to run checkpoint workflow with message: "{message}"')
        
        try:
            # Use asyncio.run() to execute the async function in a sync context
            result = asyncio.run(run_checkpoint_workflow(
                initial_message=message,
                use_azure_blob=use_azure_blob,
                auto_cleanup=auto_cleanup,
                max_retries=max_retries,
                failure_probability=failure_probability
            ))
            logging.info(f'Workflow completed, got result: {result.keys() if result else "None"}')
        except Exception as workflow_error:
            logging.error(f'Error in run_checkpoint_workflow: {workflow_error}', exc_info=True)
            raise
        
        # Return success response
        response_data = {
            "success": True,
            "execution_id": result["execution_id"],
            "workflow_success": result["workflow_success"],
            "attempts_used": result["attempts_used"],
            "total_checkpoints": result["total_checkpoints"],
            "final_output": result.get("final_output"),
            "storage_type": "azure_blob" if use_azure_blob else "local_file",
            "parameters": {
                "message": message,
                "max_retries": max_retries,
                "failure_probability": failure_probability,
                "auto_cleanup": auto_cleanup
            },
            "checkpoints": result.get("checkpoints", [])
        }
        
        logging.info(f'Workflow completed successfully. Execution ID: {result["execution_id"]}')
        
        return func.HttpResponse(
            json.dumps(response_data, indent=2),
            status_code=200,
            mimetype="application/json"
        )
        
    except Exception as e:
        logging.error(f'Error running checkpoint workflow: {str(e)}', exc_info=True)
        
        return func.HttpResponse(
            json.dumps({
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__
            }),
            status_code=500,
            mimetype="application/json"
        )


@app.route(route="health", methods=["GET"])
def health_check(req: func.HttpRequest) -> func.HttpResponse:
    """Simple health check endpoint."""
    return func.HttpResponse(
        json.dumps({
            "status": "healthy",
            "service": "checkpoint-workflow-function",
            "version": "1.0.0"
        }),
        status_code=200,
        mimetype="application/json"
    )


@app.route(route="info", methods=["GET"])
def info(req: func.HttpRequest) -> func.HttpResponse:
    """Get information about the function and available parameters."""
    return func.HttpResponse(
        json.dumps({
            "service": "checkpoint-workflow-function",
            "version": "1.0.0",
            "endpoints": {
                "/api/run_workflow": {
                    "method": "POST",
                    "description": "Run the checkpoint workflow with retry logic",
                    "parameters": {
                        "message": {
                            "type": "string",
                            "default": "hello world",
                            "description": "Input message to process through the workflow"
                        },
                        "use_azure_blob": {
                            "type": "boolean", 
                            "default": True,
                            "description": "Use Azure Blob Storage for checkpoints (vs local file storage)"
                        },
                        "auto_cleanup": {
                            "type": "boolean",
                            "default": False,
                            "description": "Automatically clean up checkpoints after successful completion"
                        },
                        "max_retries": {
                            "type": "integer",
                            "default": 3,
                            "range": "1-10",
                            "description": "Maximum number of retry attempts on failure"
                        },
                        "failure_probability": {
                            "type": "number",
                            "default": 0.6,
                            "range": "0.0-1.0",
                            "description": "Probability of simulated transient failures (0=no failures, 1=always fail)"
                        }
                    }
                },
                "/api/health": {
                    "method": "GET",
                    "description": "Health check endpoint"
                },
                "/api/info": {
                    "method": "GET", 
                    "description": "Get API information and documentation"
                }
            },
            "example_request": {
                "url": "/api/run_workflow",
                "method": "POST",
                "body": {
                    "message": "test workflow",
                    "use_azure_blob": True,
                    "auto_cleanup": False,
                    "max_retries": 3,
                    "failure_probability": 0.4
                }
            }
        }, indent=2),
        status_code=200,
        mimetype="application/json"
    )