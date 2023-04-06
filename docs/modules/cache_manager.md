# Cache Manager

The **Cache Manager** is responsible for controlling the operation of both the **Cache Storage** and **Vector Store**.
  - **Eviction Policy**:
  Currently, GPTCache makes decisions about evictions based solely on the number of lines. This approach can result in inaccurate resource evaluation and may cause out-of-memory (OOM) errors. We are actively investigating and developing a more sophisticated strategy.
    - [✓] LRU eviction policy
    - [✓] FIFO eviction policy
    - [ ] More complicated eviction policies
