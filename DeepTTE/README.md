This project adapts the official implementation of the AAAI 2018 paper *When Will You Arrive? Estimating Travel Time Based on Deep Neural Networks* and extends it for our own rail travel time estimation task.

The original code was designed for taxi trajectory data. In our work, we retained the DeepTTE architecture as the baseline model, but we replaced the original sample data pipeline with our own data preprocessing workflow to support rail trajectory data and additional contextual features.

We provide our own data_prep.py script to convert raw CSV trip records into the JSONL format required by DeepTTE. Unlike the original implementation, our preprocessing groups records by trip_id and service date, so that each instance represents a single trip occurrence on a specific day. It also computes trip-level and trajectory-level features from row-level observations.

In addition to the original DeepTTE inputs, our adapted preprocessing includes extra rail- and context-related features such as:

- arrival delay

- departure delay

- speed

- peak-hour indicator

- temperature

- apparent temperature

- precipitation

- rain

- snowfall

- wind speed

- wind gust

- wind direction

These features are incorporated into the modified model pipeline through our updated preprocessing, configuration, and trajectory encoding components.

The original sample data in Chengdu can still be replaced with custom datasets, but in our case the repository has been adapted to use our own processed rail dataset instead of the original taxi data.

# **Data**

download jsonl files: https://drive.google.com/drive/u/0/folders/1KlkrNklfQaCP7O9vt5ybRMQOwu3_QQP4


# **Usage**

## **Model Training**

python main.py --task train --batch_size 64 --epochs 50 --pooling_method attention --kernel_size 3 --alpha 0.1 --data_ratio 1.0 --log_file deeptte_run

with iteration (until model testing):

python main.py --task train  --iterations 30 --batch_size 64 --epochs 50 --pooling_method attention --kernel_size 3 --alpha 0.1 --data_ratio 1.0 --log_file deeptte_run

### **Parameters**

- task: train/test

- batch_size: batch size for training

- epochs: number of training epochs

- kernel_size: kernel size of Geo-Conv

- pooling_method: attention/mean

- alpha: weight used in multi-task learning

- log_file: path of log file

- result_file: path to save prediction results

## **Model Evaluation**

python main.py --task test --weight_file ./saved_weights/best_model.pt --batch_size 64 --result_file ./result/deeptte.res --pooling_method attention --kernel_size 3 --alpha 0.1 --data_ratio 1.0 --log_file deeptte_test

### **Parameters**

- weight_file: path of saved model weight

- result_file: path to save evaluation results

## **Using Our Own Data**

Instead of directly using the original Chengdu sample trajectories, we preprocess our own dataset into the JSONL format expected by DeepTTE.

Each sample is stored as a JSON string containing:

- driverID

- tripID

- dateID: day in month, from 0 to 30

- weekID: day of week, from 0 to 6

- timeID: start time in minutes, from 0 to 1439

- dist: total trip distance in km

- time: total trip travel time in minutes

- lngs: sequence of longitude points

- lats: sequence of latitude points

- states: sequence of status values

- time_gap: elapsed time from the first point

- dist_gap: cumulative distance from the first point

Additional features in our adapted version include:

- arrival_delay

- departure_delay

- speed

- is_peak_hour

- temperature

- apparent_temperature

- precipitation

- rain

- snowfall

- windspeed

- windgust

- winddirection

After preprocessing, the normalization statistics in config.json must also be regenerated so they match the new dataset and features.

## **Data Preprocessing**

We use a custom data_prep.py script to transform raw CSV trip data into the DeepTTE JSONL format.

Key preprocessing steps:

- parse timestamps and service dates

- sort records by trip_id and timestamp

- group rows by trip_id and trip date

- keep only rows with valid latitude and longitude

- compute trip-level distance and time from row-level incremental values

- build trajectory sequences for longitude, latitude, state, time gap, and distance gap

- attach additional operational and weather-related features

===================================================================================================================================

This project is the code of AAAI 2018 paper ***When Will You Arrive? Estimating Travel Time Based on Deep Neural Networks***.

We provide the complete version of code and part of sample data in Chengdu. You can replace the sample data with your own data easily. See the samples in data/ for more details.
The complete data can be downloaded at https://duke.box.com/s/ni5ca8iktneq828fk5cul8afwkvszkdr , which is provided by the following competion http://www.dcjingsai.com/common/cmpt/%E4%BA%A4%E9%80%9A%E7%BA%BF%E8%B7%AF%E9%80%9A%E8%BE%BE%E6%97%B6%E9%97%B4%E9%A2%84%E6%B5%8B_%E8%B5%9B%E4%BD%93%E4%B8%8E%E6%95%B0%E6%8D%AE.html.

# Usage:

## Model Training
python train.py
### Parameters:

* task: train/test
* batch_size: the batch_size to train, default 400
* epochs: the epoch to train, default 100
* kernel_size: the kernel size of Geo-Conv, only used when the model contains the Geo-conv part
* pooling_method: attention/mean
* alpha: the weight of combination in multi-task learning
* log_file: the path of log file
* result_file: the path to save the predict result. By default, this switch is off during the training

Example:
```
python main.py --task train  --batch_size 10  --result_file ./result/deeptte.res --pooling_method attention --kernel_size 3 --alpha 0.1 --log_file run_log
```


## Model Evaluation

### Parameters:
* weight_file: the path of model weight
* result_file: the path to save the result

## Example:
```
python main.py --task test --weight_file ./saved_weights/weight --batch_size 10  --result_file ./result/deeptte.res --pooling_method attention --kernel_size 3 --alpha 0.1
```

## How to User Your Own Data
In the data folder we provide some sample data. You can use your own data with the corresponding format as in the data samples. The sampled data contains 1800 trajectories. To make the model performance close to our proposed result, make sure your dataset contains more than 5M trajectories.

### Format Instructions
Each sample is a json string. The key contains:
* driverID
* dateID: the date in a month, from 0 to 30
* weekID: the day of week, from 0 to 6 (Mon to Sun)
* timeID: the ID of the start time (in minute), from 0 to 1439
* dist: total distance of the path (KM)
* time: total travel time (min), i.e., the ground truth. You can set it as any value during the test phase
* lngs: the sequence of longitutes of all sampled GPS points
* lats: the sequence of latitudes of all sampled GPS points
* states: the sequence of taxi states (available/unavaible). You can remove this attributes if it is not available in your dataset. See models/base/Attr.py for details.
* time_gap: the same length as lngs. Each value indicates the time gap from current point to the firt point (set it as arbitrary values during the test)
* dist_gap: the same as time_gap

The GPS points in a path should be resampled with nearly equal distance.

Furthermore, repalce the config file according to your own data, including the dist_mean, time_mean, lngs_mean, etc.


