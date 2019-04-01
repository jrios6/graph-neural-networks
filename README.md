### Code for Graph Neural Network Experiments

Project Blog and Updates at https://jrios6.github.io/blog/.

### Instructions
1. Install required libraries with `source install_python_gpu_script.sh`. 
2. Run RGGCN experiments with `python train.py --dataset DATASET_NAME --save_path model_states/PATH_NAME`. Replace `DATASET_NAME` with `cora`, `citeseer` or `pubmed` for the respective datasets. Replace `PATH_NAME` with the desired directory path to save the test model at.
3. Additional command line parameters can be found in train.py. 
4. It is recommended to run the training script with `--no-cuda` for pubmed dataset due to memory requirements.
