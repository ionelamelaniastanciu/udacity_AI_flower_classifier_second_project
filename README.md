## AI Programming with Python Project

##### This project is made during AWS AI & ML Scholarship 2023

Project code for Udacity's AI Programming with Python Nanodegree program. 
In this project, students first develop code for an image classifier built with PyTorch, then convert it into a command line application.

### Files

#### Python files:

##### 1. train.py 
    ~ contains the main function with the algorthm
##### 2. get_input_args.py 
    ~ contains the possible arguments for the command line
##### 3. train_neural_network.py
    ~ trains and validate the neural network
##### 4. predict.py
     ~ predicts the class (or classes) of an image using a trained deep learning neural network
##### 5. test_image.py
    ~ test_image
        tests an image and returns the probabilities and labels resulted from the model
    ~ test_all_images 
            tests all the images and writes to a file
##### 6. checkpoint_saver.py
    ~ save_model saves the checkpoint of a model
    ~ load_model loads the checkpoint of a model

#### Other files:

##### 1. cat_to_name.json 
     ~ contains all the names of the flowers in JSON format
##### 2. train_results.txt
    ~  contains the output of the training
##### 3. test_all_images_<method>.txt
    ~  contains the output of the all images
##### 4. checkpoint<method>_command_line.txt
    ~  contains the checkpoints for the models

#### Jupyter Notebook files
##### Image_Classifier_Project<method>.ipynb
##### checkpoint<method>.txt

#### How to run
    1. using the train.py file directly on IDE

    2. using any commands:

    python train.py
    
    python train.py data_dir --save_dir save_directory 
    * Choose architecture: python train.py data_dir --arch "vgg13" 
    * Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20 
    * Use GPU for training: python train.py data_dir --gpu (by default is cuda)
