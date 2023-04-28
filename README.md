### "Deepfake-detection" with Explainable AI

This repo contains the code, results, models for Deep Fake detection with XAI.
Course Project for ECE 792 - Advanced Machine Learning


- To classify real and fake images, pretrained XceptionNet model is implemented.
- For model interpretability, algorithms such as LIME and Grad-CAM are implemented.

#### Datasets used:
- FaceForensic++
- Celeb-DF

#### Files Description
- ```models```: contains trained models
- ```outputs```: contains output results of predictions, lime and grad-cam
- ```plots```: contains plots of accuracy, loss against epochs
- ```CelebDF_XceptionNet_Code.ipynb```: contains code of xception on celebdf dataset
- ```FaceForensics_XceptionNet_Code.ipynb```: contains code of xception on faceforensics dataset
- ```face_extraction_code_from_vidoes.py```: contains code for extraction of faces from the videos
- ```grad_cam.ipynb```: contains the code for gradCAM algorithm
- ```lime_code.ipynb```: contains the code for LIME algorithms
- ```plot_curves.ipynb```: contains the code to plot the graphs
- ```test_predictions.ipynb```: continas the code for predicicion on test dataset
