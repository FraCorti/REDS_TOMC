# REDS:  Resource-Efficient Deep Subnetworks for Dynamic Resource Constraints


In this work we introduce Resource-Efficient Deep Subnetworks (REDS) to tackle model adaptation to variable resources. In contrast to the state-of-the-art, REDS use structured sparsity constructively by exploiting permutation invariance of neurons, which allows for hardware-specific optimizations. Specifically, REDS achieve computational efficiency by (1) skipping sequential computational blocks identified by a novel iterative knapsack optimizer, and (2) leveraging simple math to re-arrange the order of operations in REDS computational graph to take advantage of the data cache. REDS support conventional deep networks frequently deployed on the edge and provide computational benefits even for small and simple networks. We evaluate REDS on seven benchmark architectures trained on the [Visual Wake Words](https://arxiv.org/pdf/1906.05721.pdf), [Google Speech Commands](https://arxiv.org/pdf/1804.03209.pdf), [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) and [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html) datasets, and test on four off-the-shelf mobile and embedded hardware platforms. We provide a theoretical result and empirical evidence for REDS outstanding performance in terms of submodels' test set accuracy, and demonstrate an adaptation time in response to dynamic resource constraints of under 40 microseconds of models deployed on [Arduino Nano 33 BLE Sense](https://docs.arduino.cc/hardware/nano-33-ble-sense) through [Tensorflow Lite for Microcontrollers](https://github.com/tensorflow/tflite-micro). 

## Software prerequisites

Install the software packages required for reproducing the experiment by running the
command: `pip3 install -r requirements.txt` inside the project folder.

Run the `setup.sh` script file to create the hierarchy of folders used to store the results of the experiments.

[Install](https://support.gurobi.com/hc/en-us/articles/360044290292-How-do-I-install-Gurobi-for-Python-)
the [GUROBI](https://www.gurobi.com/) solver and obtain
a license (see [free academic license](https://support.gurobi.com/hc/en-us/articles/360040541251-How-do-I-obtain-a-free-academic-license-)).
To link the license and the solver to the programs you have to pass the arguments: **--gurobi_home** and
**--gurobi_license_file** to each program. The former points to the absolute path of the installation of Gurobi and the
latter to its license.

```
python kws_ds_convolution.py --gurobi_license_file path/to/license/gurobi.lic --gurobi_home path/to/installation//gurobi/gurobi1002/linux64 
```

Change linux64 with your operating system version/type and remember to set the enviroment variables correctly, see [here](https://support.gurobi.com/hc/en-us/articles/13443862111761-How-do-I-set-system-environment-variables-for-Gurobi) for more information.  

## Fine-tuning REDS models

For each program, you can specify the usage of the GPU by passing an id number from the  `--cuda_device` argument. In the default configuration, all the experiments results are stored inside the /logs directory and printed to the screen.
For each program, you can specify the solver's maximum running time per iteration by passing the value in seconds to the
`--solver_time_limit` argument. For the [DS-CNN size L](https://arxiv.org/pdf/1711.07128.pdf), the suggested time is at least 3 hours (10800
seconds).

All the individual subnetwork architectures can be trained in isolation by running the `_full_training.py` files. 

**Fine-tune DS-CNN models**

```
python kws_ds_convolution.py --gurobi_license_file path/to/license/gurobi.lic --gurobi_home path/to/installation//gurobi/gurobi1002/linux64 
```

To train the REDS DS-CNN S models on *CIFAR10* or *Fashion-MNIST* run for the former the *vision_ds_convolution_fashion_mnist.py* file and for the latter *vision_ds_convolution_cifar10.py* file. The pre-trained models are stored in the models/ folder.

**Fine-tune DNN models**

```
python kws_dnn.py --gurobi_license_file path/to/license/gurobi.lic --gurobi_home path/to/installation//gurobi/gurobi1002/linux64 
```

**Fine-tune CNN models**

```
python kws_convolution_cnn.py --gurobi_license_file path/to/license/gurobi.lic --gurobi_home path/to/installation//gurobi/gurobi1002/linux64 
```


## Analysis results on Pixel 6 and Xiaomi Redmi Note 9 Pro 
The results obtained from the subnetworks configuration are obtained from the official Google
Tensorflow Lite benchmarking [tool](https://www.tensorflow.org/lite/performance/measurement). From left to right: number of model parameters, model accuracy and model inference
as a function of MAC percentage in each REDS subnetwork. 

**(1) Models size S**

<img src="result/plots/plotly_mobile_parameters_sizeS.png" width="270"/> <img src="result/plots/plotly_mobile_accuracy_sizeS.png" width="270"/> <img src="result/plots/plotly_mobile_inference_sizeS.png" width="270"/> 

**(2) Models size L**

<img src="result/plots/plotly_mobile_parameters_sizeL.png" width="270"/> <img src="result/plots/plotly_mobile_accuracy_sizeL.png" width="270"/> <img src="result/plots/plotly_mobile_inference_sizeL.png" width="270"/>

## Visual Wake Words 
To download the Visual Wake Words dataset please refer to [this](https://github.com/arpit6232/visualwakeup_aesd) github repository. The tf-record files need to be located inside the *visualwakeup_aesd/data*, you do not need to convert the Tensorflow Object Detection API proto files because they are already provided as python files in the *visualwakeup_aesd/lib* folder. 

## Fine-tune MobileNet v1 model 
After downloading and convert the Visual Wake Words dataset to tf-record files you can run the [MobileNet v1](https://arxiv.org/pdf/1704.04861.pdf) finetuning. Be careful that the GUROBI solver and Visual Wake Words RAM consumption can take up to 30 GB. Run the following command from the shell:  

```
python knapsack_mobilenetv1_leaky_relu.py --gurobi_license_file path/to/license/gurobi.lic --gurobi_home path/to/installation//gurobi/gurobi1002/linux64 
```

The peak memory usage analysis for the MobileNet v1 backbone is obtained from the [tflite tools](https://github.com/eliberis/tflite-tools). The knapsack OR-Tools formulation can be found in the *knapsack.py* file in the *ortools_knapsack_solver_mobilenetv1* function.  
<img src="result/plots/SRAM_MobileNetV1_vww_96_96_3_1constraint.png" width="270"/> <img src="result/plots/SRAM_MobileNetV1_vww_96_96_3_2constraints_200KB.png" width="270"/>

To run the solver *without* the peak memory usage constraint pass the *--peak_memory_constraint* flag to the python script. Be careful to set a timer to the solver when doing that by passing the flag *--solver_time_limit* followed by the number of seconds needed (ie. 40000 should be fine, in case is not increase it). Be careful that the GUROBI solver could be take up to 50 GB of RAM during the subnetwork architecture search process. 

In all the analysis we conduct the peak memory usage is considered the [maximum size](https://dl.acm.org/doi/pdf/10.1145/3569468) in bytes between all the activation maps produced by the model. 


## TensorFlow Lite for Microcontrollers Analysis 
TO BE PUBLISHED
