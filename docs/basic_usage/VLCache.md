# VLCache Usage

This is the official usage guide for VLCache. For details, please refer to the paper: 
**[VLCache: Computing 2% Vision Tokens and Reusing 98% for Vision-Language Inference](https://arxiv.org/abs/2512.12977)**  

VLCache enables efficient vision-language inference by reusing previously computed vision tokens KV caches across requests. This example shows how to run VLCache on Qwen3-VL-8B-Instruct model.

## 1. Launch Qwen3-VL model with SGLang

```bash
python3 -m sglang.launch_server \
  --model-path /models/Qwen3-VL-8B-Instruct/ \
  --mem-fraction-static 0.7 \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 30000 \
  --disable-cuda-graph \
  --disable-radix-cache \
  --chunked-prefill-size 8000
```

## 2. First Request: Generate KV Cache

Send a request with two images to generate the initial KV cache:

```bash
curl -X POST "http://localhost:30000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/models/Qwen3-VL-8B-Instruct",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "Describe these images."
          },
          {
            "type": "image_url",
            "image_url": {
              "url": "https://dashscope.oss-cn-beijing.aliyuncs.com/images/dog_and_girl.jpeg"
            }
          },
          {
            "type": "image_url",
            "image_url": {
              "url": "https://dashscope.oss-cn-beijing.aliyuncs.com/images/tiger.png"
            }
          }
        ]
      }
    ],
    "max_tokens": 30
  }'
```

## 3. Second Request: Reuse Position-independent KV Cache

Send a new request with the **same images in different order**. The server will reuse cached representations for the images, significantly reducing latency:

```bash
curl -X POST "http://localhost:30000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/models/Qwen3-VL-8B-Instruct",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "Describe these images."
          },
          {
            "type": "image_url",
            "image_url": {
              "url": "https://dashscope.oss-cn-beijing.aliyuncs.com/images/tiger.png"
            }
          },
          {
            "type": "image_url",
            "image_url": {
              "url": "https://dashscope.oss-cn-beijing.aliyuncs.com/images/dog_and_girl.jpeg"
            }
          }
        ]
      }
    ],
    "max_tokens": 30
  }'
```

### Tips
By default, **recompute ratio** is set to 0.3 per layer. You can customize this value by adding the `recompute_ratio_in_layer` field to the model’s `config.json`:

```json
{
  "recompute_ratio_in_layer": 0.2
}
```

- Value can be:
  - A **float** between `0.0` and `1.0` (applies uniformly to all layers), or  
  - A **list of floats** with length equal to the number of model layers (e.g., 36 for Qwen-VL-8B).
- If using a list, the values **must be non-increasing** (i.e., `ratio[i] >= ratio[i+1]`).


