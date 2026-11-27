# Qwen3-30B-A3B-Instruct-2507-FP8 SageMaker Deployment

This guide provides comprehensive instructions for deploying the Qwen3-30B-A3B-Instruct-2507-FP8 model on AWS SageMaker using vLLM and Docker.

## Hardware Requirements

### GPU Requirements for FP8 Models

The FP8 version of Qwen3 requires specific GPU capabilities:

- **Recommended GPUs**: NVIDIA GPUs with compute capability > 8.9
  - Ada Lovelace architecture
  - Hopper architecture
  - Later GPU generations
  - These GPUs run FP8 models as **w8a8** (8-bit weights and activations)

- **Alternative GPUs**: Ampere cards (with vLLM v0.9.0+)
  - Supports FP8 Marlin block-wise quantization
  - Runs as **w8a16** (8-bit weights, 16-bit activations)

> **Note**: The FP8 models use block-wise quantization. For detailed GPU selection and limitations, see the [vLLM FP8 documentation](https://docs.vllm.ai/en/latest/features/quantization/fp8/).

### AWS Instance Types

#### Endpoint Instance
- **Instance Type**: `ml.g6.48xlarge`
- **GPU Count**: 8 GPUs
- **Purpose**: Model inference endpoint
- **Justification**:
    - Uses the Ada Lovelace GPU, meeting our FP8 architecture requirements
    - Qwen3 30B models use a GQA architecture with 4 KV heads. Using a GPU instance with 4 GPUs creates a 1:1 mapping, which can lead to OOM or memory fragmentation issues. We use an 8 GPU instance to counteract this

#### Notebook Instance (for deployment)
- **Instance Type**: `ml.t3.medium`
- **Environment**: SageMaker Notebook Instance (not Studio)

## Deployment Strategy

### Prerequisites

#### AWS IAM Permissions
Ensure your IAM role has the following permissions:
- `AmazonEC2ContainerRegistryFullAccess`
- `AmazonS3FullAccess`
- `AmazonSageMakerFullAccess`

#### Dependencies

Install required Python packages:
```bash
pip install -U sagemaker boto3 awscli
```

#### Docker Setup
- Update vLLM to the latest version in your Dockerfile
- Use the official vLLM SageMaker entrypoint script
  - Reference: [vLLM SageMaker Entrypoint](https://docs.vllm.ai/en/stable/examples/online_serving/sagemaker-entrypoint/)

Base Dockerfile:
```bash
FROM vllm/vllm-openai:v0.11.2

COPY ./sagemaker-entrypoint.sh /app/
RUN chmod +x /app/sagemaker-entrypoint.sh

ENTRYPOINT ["/app/sagemaker-entrypoint.sh"]
```

### Deployment Steps

#### 1. ECR Repository Setup
Create an Amazon ECR repository to store your Docker image:
```bash
aws ecr create-repository --repository-name <your-repo-name> --region <your-region-name>
```

#### 2. Build Docker Image
Build the Docker image with the latest vLLM version:
```bash
docker build --build-arg VERSION=latest -t <your-repo-name>:latest .
```

#### 3. Push Image to ECR
Authenticate and push the image to ECR:
```bash
aws ecr get-login-password --region <region> | docker login --username AWS --password-stdin <account-id>.dkr.ecr.<region>.amazonaws.com
docker tag <repo-name>:latest <account-id>.dkr.ecr.<region>.amazonaws.com/<repo-name>:latest
docker push <account-id>.dkr.ecr.<region>.amazonaws.com/<repo-name>:latest
```

#### 4. Configure vLLM Parameters

The deployment uses environment variables with the `SM_VLLM_` prefix to configure vLLM:

```python
VLLM_ENV = {
    'SM_VLLM_MODEL': "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8",
    'SM_VLLM_TENSOR_PARALLEL_SIZE': '8',
    'SM_VLLM_MAX_MODEL_LEN': '4096',
    'SM_VLLM_GPU_MEMORY_UTILIZATION': '0.8',
    'SM_VLLM_ENABLE_EXPERT_PARALLEL': 'true'
}
```

**Configuration Details**:
- **Tensor Parallel Size**: `8` (matches the 8 GPUs on ml.g6.48xlarge)
- **Max Model Length**: `4096` tokens (adjust lower if experiencing memory issues)
- **GPU Memory Utilization**: `0.8` (80% - adjust lower if needed)
- **Expert Parallel**: `true` (required for compatibility with FP8 block-wise quantization)

For full vLLM configuration options, see:
- [vLLM Environment Variables](https://docs.vllm.ai/en/stable/configuration/env_vars/)
- [vLLM Engine Arguments](https://docs.vllm.ai/en/v0.4.1/models/engine_args.html)

#### 5. Deploy to SageMaker

Create and deploy the SageMaker model:
```python
model = sagemaker.Model(
    name=model_name,
    image_uri=CONTAINER,
    sagemaker_session=sagemaker_session,
    role=iam_role,
    env=VLLM_ENV,
)

predictor = model.deploy(
    instance_type='ml.g6.48xlarge',
    initial_instance_count=1,
    endpoint_name=endpoint_name,
    container_startup_health_check_timeout=450
)
```

## Invocation

### Example Inference Request

Use the SageMaker Runtime client to invoke the endpoint:

```python
import json
import boto3

# Define the payload
payload = {
    "model": "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8",
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Hi, how are you doing?"
                }
            ]
        }
    ],
    "temperature": 0.7,
    "max_tokens": 100,
    "stream": False
}

# Invoke the endpoint
sagemaker_runtime = boto3.client('sagemaker-runtime', region_name='<your-region>')
response = sagemaker_runtime.invoke_endpoint(
    EndpointName=endpoint_name,
    ContentType='application/json',
    Body=json.dumps(payload)
)

# Parse response
response_body = json.loads(response['Body'].read().decode())
print(response_body)
```

## Resource Cleanup

To avoid ongoing charges, delete the deployed resources:

```python
sagemaker_client = boto3.client('sagemaker', region_name='<your-region>')

# Delete model
sagemaker_client.delete_model(ModelName=model_name)

# Delete endpoint
sagemaker_client.delete_endpoint(EndpointName=endpoint_name)

# Delete endpoint configuration
sagemaker_client.delete_endpoint_config(EndpointConfigName=endpoint_name)
```

## References

1. [Qwen3 vLLM Deployment Guide](https://qwen.readthedocs.io/en/latest/deployment/vllm.html)
2. [vLLM FP8 Quantization](https://docs.vllm.ai/en/latest/features/quantization/fp8/)

## Troubleshooting

### Memory Issues
If you encounter out-of-memory errors:
1. Reduce `SM_VLLM_MAX_MODEL_LEN` (e.g., from 4096 to 2048)
2. Lower `SM_VLLM_GPU_MEMORY_UTILIZATION` (e.g., from 0.8 to 0.7)

### Tensor Parallel Errors
If you see errors about weight quantization block size:
```
ValueError: The output_size of gate's and up's weight = 192 is not divisible by weight quantization block_n = 128.
```

Solutions:
- See if reduced tensor parallel size works: `'SM_VLLM_TENSOR_PARALLEL_SIZE': '4'`
- Ensure expert parallel is enabled: `'SM_VLLM_ENABLE_EXPERT_PARALLEL': 'true'`

### Container Startup Timeout
The `container_startup_health_check_timeout` is set to 450 seconds. If deployment fails due to timeout:
- Increase this value in the `model.deploy()` call
- Check CloudWatch logs for detailed error messages
