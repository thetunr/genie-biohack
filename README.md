# Genie in a Bottle

## Overview

This is our skin lesion classification project that utilizes the Fitzpatrick17k dataset and a Convolutional Neural Network (CNN) model to analyze images of skin lesions. The model that is provided by the [Fitzpatrick17k team](https://github.com/mattgroh/fitzpatrick17k) predicts a skin condition for a given image, contributing to early detection and informed decision-making in dermatological health. Our aim was to utilize their resources to adjust their model to predict whether a given skin lesion is **benign**, **malignant**, or **nonneoplastic**.

This project was developed as part of **[BioHack 2024](https://www.biohacknyc.com/)**.

## Group Members

- Shiraz Bheda
- Soohyeuk Choi
- Chloe Jung
- Sol Jung
- Tony Oh

## Dataset

We utilized the [Fitzpatrick17k dataset](https://github.com/mattgroh/fitzpatrick17k), a comprehensive dataset of skin images with expert-labeled annotations. This dataset was central to the development and evaluation of our project.

## How to Run

1. Clone this repository.
2. Adjust all directories to point to the correct locations for data and scripts.
3. Run `scraping.py` to retrieve the necessary images.
4. Train the model by running `train.py`.
5. Use `predict.py` to input an image and view the classification output.

## Status

This project requires further work to address various issues, including refining the workflow and improving the integration of scripts. The CNN model used in this project is derived directly from the Fitzpatrick17k GitHub repository. Future updates and fixes will be made to enhance usability and functionality.

## Current Results

Our model demonstrated promising results in classifying skin lesions with around 80% accuracy on the testing set, making it a valuable tool for dermatological analysis.

## Acknowledgments

This project was built upon the outstanding work of the Fitzpatrick17k team. This project was only possible through utilizing their extensive dataset and resources. For more details, visit their [GitHub repository](https://github.com/mattgroh/fitzpatrick17k).
