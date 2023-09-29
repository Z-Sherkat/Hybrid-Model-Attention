# Introduction
This is the official implementation of a Self-Attention-Based LSTM, Bi-LSTM, CNN-LSTM and CNN-Bi-LSTM Model for SOC estimation in "A Self-Attention-Based CNN-Bi-LSTM Model for Accurate State-of-Charge Estimation of Lithium-
Ion Batteries".

# Code Description:

   •	N10deg/->create 12 deep learning models (with and without attention
mechanism) for -10 degree.
 
   •	0deg/->create 12 deep learning models (with and without attention
mechanism) for zero degree.
 
   •	10deg/->create 12 deep learning models (with and without attention
mechanism) for 10 degree.
 
   •	25deg/->create 12 deep learning models (with and without attention
mechanism) for 25 degree.

# Images:

images/ -> images presented in our publication.


# Dataset

The McMaster data set includes data of a series of tests carried out at six ambient temperatures
(40deg, 25deg, 10deg, 0deg, −10deg, and −20deg). The dataset encompasses a series of tests temperatures is available 
here: https://data.mendeley.com/datasets/cp3473x7xv/3. The considered McMaster data set was collected from 
a single 3 Ah LG Chem INR18650HG2 NMC cell. Its specifications are listed in Table below. The data set includes
the information on four discharges of automotive industry standard driving cycles: US06, LA92, UDDS, and HWFET,
as well as eight additional mix drive cycles with associated charge cycles. In the original McMaster data set, discharge cycles were
collected at 1Hz, while charge cycles were collected at 0.01Hz. Among upsampling techniques, the linear interpo-
lation technique was employed to increase the number of samples in charge cycles to ensure equal sample rates of
charge/discharge cycles. To improve the prediction results, two additional features, including the average voltage and
the average current, over all previous k steps were added to the input data set. 


<img width="305" alt="Screenshot 2023-09-28 at 8 23 16 PM" src="https://github.com/Z-Sherkat/Hybrid-Model-Attention/assets/97856714/4eab9c5d-c8b9-4a57-933f-e31d51489430">

 # Performance Comparision

 Results of SOC estimation at room temperature reported for 12 competing deep learning models (with and without attention
mechanism).

<img width="532" alt="Screenshot 2023-09-29 at 12 29 08 PM" src="https://github.com/Z-Sherkat/Hybrid-Model-Attention/assets/97856714/e7731860-27dc-499c-baa1-23b2c34b9ca3">

 Results of SOC estimation at lower ambient temperatures, for 12 competing deep learning models
(with and without attention mechanism). 

 <img width="531" alt="Screenshot 2023-09-29 at 12 27 46 PM" src="https://github.com/Z-Sherkat/Hybrid-Model-Attention/assets/97856714/148f70a7-283c-4a6a-bd91-493435f10e04">



 
 # Reference Papers

