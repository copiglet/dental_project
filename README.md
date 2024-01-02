
### README for Dental Diagnostic AI Project

#### Project Overview
This project leverages deep learning and advanced medical imaging techniques to enhance the precision of dental diagnostics. It focuses on developing software that utilizes cutting-edge medical imaging technology to create accurate 3D models, aiding in the early diagnosis of periodontal diseases.

#### Technical Stack
- **Programming Language:** Python
- **Libraries & Frameworks:** nnU-Net, TensorFlow, PyTorch, MONAI
- **Data Processing & Analysis:** NumPy, Pandas, OpenCV
- **Deployment:** Docker, Docker Compose

#### Project Structure
- `run.py`: Main script that orchestrates the entire pipeline, from processing DICOM series, converting them to NIfTI format, to various image processing operations.
- `models.py`: Defines deep learning models using MONAI, specifically UNETR and UNet, for image segmentation tasks relevant in medical imaging analysis.
- `utils.py`: Contains utility functions for data processing, image transformations, and other auxiliary operations that support the main pipeline.
- `predict_CBCTSeg.py`: Script for predicting or segmenting CBCT images, utilizing the models and utilities defined in `models.py` and `utils.py`.

#### How to Run
1. Ensure all dependencies are installed as per the technical stack.
2. Use `run.py` to execute the entire pipeline. This script will handle data preparation, model training, and inference.
3. For specific CBCT image segmentation, use `predict_CBCTSeg.py`, ensuring the necessary input data is correctly formatted and available.
4. Models can be accessed and modified in `models.py` for experimentation or optimization.
5. `utils.py` provides additional support and can be modified as per project requirements.

#### Achievements
- Development of a precise 3D segmentation model for visualizing dental structures and periodontal disease.
- The model provides critical information for dental diagnosis and treatment planning.
- Enhanced clinical diagnostics through early detection and monitoring of periodontitis.

#### Deployment
- Models are deployable using Docker, ensuring accessibility and scalability in clinical environments.

#### Model Output Examples
- The `merged.png` file showcases the results obtained from the nnU-Net deep learning model trained on datasets of maxilla, mandible, root canal, and tooth. The images display the segmented 3D structures with clear differentiation of each anatomical area.
- The `pdt.png` file presents the outcome of the nnU-Net model trained on datasets with manually labeled periodontitis disease. It illustrates the model's capability to identify and segment the disease-affected areas accurately.

These visualizations demonstrate the effectiveness of our deep learning models in providing valuable insights for dental diagnostics and disease identification.
