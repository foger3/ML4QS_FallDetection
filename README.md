# [ML4QS_FallDetection](https://github.com/foger3/ML4QS_FallDetection)

This project aim to distinguish the falling from the non-falling motion by using appropriate preprocessing steps and state-of-the-art machine learning techniques.

# Getting Started

## Prerequisite

- **Python3.10**

- **Clone the ML4Qs_FallDetection repository**

  ```bash
  git clone https://github.com/foger3/ML4QS_FallDetection.git
  ```

- **Install required python packages**

  ```bash
  pip3 install -r ML4QS_FallDetection/src/requirement.txt
  ```



# Usage

Run `python3 ML4QS_FallDetection/src/main.py`. There are several options can be set as shown below.

```bash
options:
  -h, --help            show this help message and exit
  
  -a, --aggregate       convert the raw data to an aggregated data format
  
  -g N, --granularity N
                        the granularity in milliseconds dataset (default: 10)
                        
  -s {temporal,non-temporal}, --split {temporal,non-temporal}
                        the train test set split method (default: temporal)
                        
  -m {like,binary}, --matching {like,binary}
                        the 'like' means 7 classes classification (wallking, running, lying, standing, sitting, 												kneeling, and falling), and
                        'binary' means binary classification (falling and non-falling) (default: like)
                        
  -o, --outlier         apply outlier detection on dataset
  
  -l, --lowpass         apply low-pass filter on dataset
  
  -n {rf,svm,convlstm,gru}, --model-name {rf,svm,convlstm,gru}
                        the name of machine learning model
```

### Example

1. Generate **random forest model** for 7 classes classification (wallking, running, lying, standing, sitting, kneeling, and falling).

   ```bash
   python3 main.py -n rf
   ```

2. Generate **ConvLSTM model** with **outlier detection** and **low-pass filter**  for 7 classes classification.

   ```bash
   python3 main.py -o -l -n rf
   ```

3. Aggregate raw data with **granularity 20 ms** and generate **GRU model**  for 7 classes classification.

   ```bash
   python3 main.py -g 20 -n gru
   ```

4. Generate **SVM model** for **binary classification** (falling and non-falling).

   ```bash
   python3 main.py -m binary -n svm
   ```

   
