# Sketch Generation via Diffusion Models using Sequential Strokes

This project documents an in-depth exploration of various diffusion model architectures for the task of generating hand-drawn sketches in a sequential, stroke-by-stroke manner using the Google Quick, Draw! dataset.

Throughout this assignment, three primary architectures were implemented and tested:
- An Unconditional Diffusion Transformer
- A Conditional Diffusion Transformer
- A Hierarchical Autoregressive model combining a StrokeHistoryEncoder with a cross-attention Diffusion Transformer

While the models encountered significant training challenges common in this domain, such as mode collapse, and did not ultimately produce high-fidelity sketches, this report details the complete implementation and rigorous experimentation process for each approach. There is no output for any provided classes.

## Installation
- Python 3.9+
- PyTorch 2.3.0
- `pip install -r requirements.txt`

## Reports & Analysis

For a comprehensive overview of the project, including the methodologies and findings, please see the two main reports:

* **Detailed Technical Notebook:** [`Technical_Assignment_Report.ipynb`](Technical_Assignment_Report.ipynb)
* **Final Summary Report:** [`general_report.pdf`](general_report.pdf)

## Project Structure

The repository is organized into the following key components:

* **`src/`**: Contains all the source code for the project.
    * **`src/models/`**: Core model architectures.
        * `dit.py`: The Diffusion Transformer (DiT) models.
        * `history_encoder.py`: The LSTM-based encoder for the store histroy encoding.
    * **`src/modules/`**: The main PyTorch `Dataset` object.
        * `dataset.py`: Handles loading and preprocessing of sketch data, contains dataset object for each approach.
    * **`src/utils/`**: Helper functions for data processing.
        * `dataset_utils.py`: Contains utility functions for data conversion and normalization.
* **`assets/`**: Contains architecture diagrams used in the reports.
* **`Technical_Assignment_Report.ipynb`**: A technically detailed Jupyter Notebook covering the entire development process, including data analysis, model implementation, training loops, and results.
* **`general_report.pdf`**: A formal report summarizing the project's objectives, methods, and conclusions.