# DNN-NSL-KDD-demo

A demonstration project implementing Deep Neural Networks (DNN) for intrusion detection using the NSL-KDD dataset. This repository contains code, scripts, and instructions for training, evaluating, and analyzing DNN-based models on the NSL-KDD intrusion detection benchmark.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Results](#results)
- [References](#references)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Overview

This project demonstrates the use of deep learning techniques, specifically Deep Neural Networks, for network intrusion detection. The NSL-KDD dataset is used as a standard benchmark to evaluate the effectiveness of these models. The repository is intended for students, researchers, and practitioners interested in applying deep learning to cybersecurity problems.

## Features

- Data preprocessing and feature engineering for NSL-KDD
- Implementation of DNN models for classification
- Model training, validation, and testing scripts
- Performance evaluation and visualization
- Easy-to-follow structure for reproducibility

## Dataset

The NSL-KDD dataset is an improved version of the KDD Cup 1999 dataset and is widely used for research in intrusion detection systems.

- [NSL-KDD Dataset Info](https://www.unb.ca/cic/datasets/nsl.html)

**Note:** Download the dataset and place it in the appropriate directory as described in the [Usage](#usage) section.

## Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/chealseainfo1/DNN-NSL-KDD-demo.git
    cd DNN-NSL-KDD-demo
    ```

2. **Create and activate a virtual environment (optional but recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. **Download the NSL-KDD dataset** from the [official website](https://www.unb.ca/cic/datasets/nsl.html) and place the files (e.g., `KDDTrain+.txt`, `KDDTest+.txt`) in the `data/` directory.

2. **Run the preprocessing script (if available):**
    ```bash
    python preprocess.py
    ```

3. **Train the DNN model:**
    ```bash
    python train.py
    ```

4. **Evaluate the model:**
    ```bash
    python evaluate.py
    ```

5. **Analyze results and visualize performance:**
    - Check the output logs, plots, and result files generated in the `results/` directory.

## Project Structure

```
DNN-NSL-KDD-demo/
│
├── data/                  # NSL-KDD dataset files
├── models/                # Saved models and checkpoints
├── results/               # Output results, plots, and logs
├── preprocess.py          # Data preprocessing script
├── train.py               # Model training script
├── evaluate.py            # Model evaluation script
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

## Results

- Model accuracy, precision, recall, F1-score, and confusion matrix are reported after training and evaluation.
- See the `results/` directory for detailed outputs and visualizations.

## References

1. Tavallaee, M., Bagheri, E., Lu, W., & Ghorbani, A. A. (2009). "A Detailed Analysis of the KDD CUP 99 Data Set". Proceedings of the 2009 IEEE Symposium on Computational Intelligence for Security and Defense Applications. [PDF](https://www.researchgate.net/publication/221618772_A_Detailed_Analysis_of_the_KDD_CUP_99_Data_Set)
2. NSL-KDD Dataset: https://www.unb.ca/cic/datasets/nsl.html
3. Hodo, E., Bellekens, X., Hamilton, A., Dubouilh, P. L., Iorkyase, E., Tachtatzis, C., & Atkinson, R. (2017). "Threat analysis of IoT networks Using Artificial Neural Network Intrusion Detection System." 2017 International Symposium on Networks, Computers and Communications (ISNCC).
4. Goodfellow, I., Bengio, Y., & Courville, A. (2016). "Deep Learning." MIT Press. [Book](https://www.deeplearningbook.org/)
5. Chollet, F. (2015). "Keras." https://keras.io/
6. Abadi, M., Agarwal, A., Barham, P., et al. (2016). "TensorFlow: Large-scale machine learning on heterogeneous systems." https://www.tensorflow.org/

## Contributing

Contributions are welcome! Please open issues or pull requests for improvements, bug fixes, or new features.

1. Fork the repository.
2. Create your feature branch (`git checkout -b feature/YourFeature`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature/YourFeature`).
5. Open a pull request.

## Acknowledgements

- NSL-KDD dataset by University of New Brunswick
- Open-source deep learning libraries: TensorFlow, Keras, PyTorch, etc.
- Community contributors and researchers

---

For questions or support, please create an issue in the repository.


