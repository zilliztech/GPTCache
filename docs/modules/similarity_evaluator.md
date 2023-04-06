# Similarity Evaluator
 
This module collects data from both the **Cache Storage** and **Vector Store**, and uses various strategies to determine the similarity between the input request and the requests from the **Vector Store**. Based on this similarity, it determines whether a request matches the cache. GPTCache provides a standardized interface for integrating various strategies, along with a collection of implementations to use. The following similarity definitions are currently supported or will be supported in the future:
  - [x] The distance we obtain from the **Vector Store**.
  - [x] A model-based similarity determined using the GPTCache/albert-duplicate-onnx model from [ONNX](https://onnx.ai/).
  - [x] Exact matches between the input request and the requests obtained from the **Vector Store**.
  - [x] Distance represented by applying linalg.norm from numpy to the embeddings.
  - [ ] BM25 and other similarity measurements
  - [ ] Support other models
 
