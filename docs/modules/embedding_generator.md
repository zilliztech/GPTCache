# Embedding Generator

This module is created to extract embeddings from requests for similarity search. GPTCache offers a generic interface that supports multiple embedding APIs, and presents a range of solutions to choose from. 
  - [✓] Disable embedding. This will turn GPTCache into a keyword-matching cache.
  - [✓] Support OpenAI embedding API.
  - [✓] Support [ONNX](https://onnx.ai/) with the GPTCache/paraphrase-albert-onnx model.
  - [✓] Support [Hugging Face](https://huggingface.co/) embedding API.
  - [✓] Support [Cohere](https://docs.cohere.ai/reference/embed) embedding API.
  - [✓] Support [fastText](https://fasttext.cc) embedding API.
  - [✓] Support [SentenceTransformers](https://www.sbert.net) embedding API.
  - [ ] Support other embedding APIs.