# VQ-ANR

This is the official code for paper: VQ-ANR: Attention Neural Representation with Vector Quantization for Time-Varying Ensemble Data Exploration.

## How to use the code
Take the Nyx dataset as an example:

- To run training, use the following command:
  ```bash
  python main.py --config_file Nyx.yaml --mode train --device 0
  ```

- To run inference, use the following command:
- ```bash
  python main.py --config_file Nyx.yaml --mode inf --device 0
  ```
