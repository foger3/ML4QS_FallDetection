# labeling
label_map = dict()
label_categories = [
    "Label Walking",
    "Label Running",
    "Label Kneeling",
    "Label Lying",
    "Label Sitting",
    "Label Standing",
    "Label Falling",
    "Error",
]
label_columns = [
    "Label Walking",
    "Label Running",
    "Label Kneeling",
    "Label Lying",
    "Label Sitting",
    "Label Standing",
    "Label Falling"
]

# round1-luca
labels = [
    label_categories[0],
    label_categories[1],
    label_categories[4],
    label_categories[5],
    label_categories[6],
    label_categories[3],
    label_categories[2],
    label_categories[2],
    label_categories[4],
    label_categories[1],
    label_categories[0],
    label_categories[6],
    label_categories[3],
    label_categories[6],
    label_categories[4],
    label_categories[6],
    label_categories[7],
    label_categories[2],
    label_categories[6],
    label_categories[7],
]
label_map["round1-luca"] = labels
# round1-nicole
labels = [
    label_categories[0],
    label_categories[1],
    label_categories[4],
    label_categories[6],
    label_categories[2],
    label_categories[4],
    label_categories[6],
    label_categories[7],
    label_categories[7],
    label_categories[5],
    label_categories[4],
    label_categories[6]
]
label_map["round1-nicole"] = labels
# round1-sam
labels = [
    label_categories[0],
    label_categories[6],
    label_categories[4],
    label_categories[6],
    label_categories[1],
    label_categories[2],
    label_categories[3],
    label_categories[6],
    label_categories[5],
    label_categories[6],
    label_categories[2],
    label_categories[0],
    label_categories[6],
    label_categories[4]
]
label_map["round1-sam"] = labels
# round2-luca
labels = [
    label_categories[5],
    label_categories[0],
    label_categories[1],
    label_categories[6],
    label_categories[4],
    label_categories[3],
    label_categories[6],
    label_categories[2],
    label_categories[6],
    label_categories[5],
    label_categories[6],
    label_categories[4],
    label_categories[6],
    label_categories[2]
]
label_map["round2-luca"] = labels
# round2-nicole
labels = [
    label_categories[4],
    label_categories[2],
    label_categories[3],
    label_categories[0],
    label_categories[1],
    label_categories[6],
    label_categories[4],
    label_categories[4],
    label_categories[6],
    label_categories[2],
    label_categories[0],
    label_categories[7],
    label_categories[3],
]
label_map["round2-nicole"] = labels
# round3-nicole
labels = [
    label_categories[0],
    label_categories[4],
    label_categories[6],
    label_categories[6],
    label_categories[3],
    label_categories[6],
    label_categories[6],
    label_categories[2],
    label_categories[6],
    label_categories[1],
    label_categories[4],
    label_categories[7],
    label_categories[6],
    label_categories[0],
    label_categories[6],
    label_categories[6],
    label_categories[5],
    label_categories[3],
    label_categories[6],
    label_categories[2],
    label_categories[6],
]
label_map["round3-nicole"] = labels
# round3-sam
labels = [
    label_categories[6],
    label_categories[2],
    label_categories[6],
    label_categories[0],
    label_categories[6],
    label_categories[7],
    label_categories[3],
    label_categories[1],
    label_categories[6],
    label_categories[4],
    label_categories[6],
    label_categories[2],
    label_categories[5],
    label_categories[5],
    label_categories[3],
    label_categories[6],
    label_categories[0],
]
label_map["round3-sam"] = labels


selected_features_outlier = ['Accelerometer Y (m/s^2)_temp_min_ws_500', 
                            'Accelerometer Z (m/s^2)_temp_mean_ws_500', 
                            'Accelerometer Y (m/s^2)_temp_mean_ws_500', 
                            'Accelerometer Y (m/s^2)_temp_max_ws_500', 
                            'Magnetometer Z (µT)_freq_0.0_Hz_ws_500', 
                            'Magnetometer X (µT)_freq_45.0_Hz_ws_500', 
                            'Barometer X (hPa)_temp_sem_ws_500', 
                            'Gyroscope Z (rad/s)_freq_20.4_Hz_ws_500', 
                            'PCA_Component_48', 
                            'PCA_Component_94', 
                            'Barometer X (hPa)_freq_18.4_Hz_ws_500', 
                            'Magnetometer Z (µT)_temp_max_ws_500', 
                            'Accelerometer Y (m/s^2)_freq_8.2_Hz_ws_500', 
                            'Accelerometer Y (m/s^2)_freq_0.0_Hz_ws_500', 
                            'Magnetometer Z (µT)_freq_11.6_Hz_ws_500', 
                            'Accelerometer X (m/s^2)_freq_42.8_Hz_ws_500', 
                            'Magnetometer X (µT)_freq_35.8_Hz_ws_500', 
                            'Accelerometer Z (m/s^2)_freq_40.0_Hz_ws_500', 
                            'Accelerometer Z (m/s^2)_freq_1.4_Hz_ws_500', 
                            'Accelerometer Y (m/s^2)_freq_5.4_Hz_ws_500', 
                            'Magnetometer Y (µT)_freq_30.0_Hz_ws_500', 
                            'Accelerometer X (m/s^2)_freq_18.4_Hz_ws_500', 
                            'Accelerometer Y (m/s^2)_freq_35.2_Hz_ws_500',
                            'Barometer X (hPa)_freq_11.2_Hz_ws_500', 
                            'Barometer X (hPa)_freq_38.8_Hz_ws_500', 
                            'Gyroscope Y (rad/s)_freq_36.8_Hz_ws_500', 
                            'Accelerometer Z (m/s^2)_freq_20.8_Hz_ws_500',
                            'Accelerometer Z (m/s^2)_freq_9.2_Hz_ws_500', 
                            'Accelerometer X (m/s^2)_freq_47.8_Hz_ws_500', 
                            'Accelerometer Y (m/s^2)_freq_44.8_Hz_ws_500']

selected_features_no_outlier = ['Accelerometer Y (m/s^2)_temp_min_ws_500', 
                                'Accelerometer Z (m/s^2)_temp_mean_ws_500', 
                                'Accelerometer Y (m/s^2)_freq_0.0_Hz_ws_500', 
                                'Accelerometer Y (m/s^2)_temp_max_ws_500', 
                                'Magnetometer Z (µT)_temp_mean_ws_500', 
                                'Accelerometer Z (m/s^2)_freq_26.6_Hz_ws_500', 
                                'Accelerometer Y (m/s^2)_freq_47.0_Hz_ws_500',
                                'Magnetometer X (µT)_freq_32.8_Hz_ws_500',
                                'Magnetometer Z (µT)_temp_max_ws_500',
                                'Barometer X (hPa)_freq_6.0_Hz_ws_500',
                                'Gyroscope Z (rad/s)_freq_2.0_Hz_ws_500', 
                                'Accelerometer Z (m/s^2)_freq_43.0_Hz_ws_500', 
                                'Magnetometer Y (µT)_freq_20.4_Hz_ws_500', 
                                'Magnetometer X (µT)_freq_45.0_Hz_ws_500', 
                                'Magnetometer Y (µT)_freq_49.0_Hz_ws_500', 
                                'Accelerometer Z (m/s^2)_freq_39.0_Hz_ws_500', 
                                'Magnetometer X (µT)_freq_2.6_Hz_ws_500', 
                                'PCA_Component_138', 
                                'Magnetometer Z (µT)_freq_50.0_Hz_ws_500', 
                                'Barometer X (hPa)_freq_34.0_Hz_ws_500', 
                                'Magnetometer X (µT)_freq_24.4_Hz_ws_500', 
                                'Magnetometer Z (µT)_freq_46.0_Hz_ws_500', 
                                'Accelerometer X (m/s^2)_freq_7.2_Hz_ws_500', 
                                'Accelerometer Z (m/s^2)_freq_33.0_Hz_ws_500',
                                'Gyroscope Z (rad/s)_freq_48.4_Hz_ws_500', 
                                'Barometer X (hPa)_freq_20.8_Hz_ws_500', 
                                'Barometer X (hPa)_freq_39.4_Hz_ws_500', 
                                'Magnetometer Z (µT)_freq_7.4_Hz_ws_500', 
                                'Magnetometer X (µT)_freq_41.6_Hz_ws_500', 
                                'Accelerometer Z (m/s^2)_freq_36.0_Hz_ws_500']

import logging
logging.basicConfig(
    format="%(levelname)s: %(message)s", encoding="utf-8", level=logging.INFO
)
logger = logging.getLogger()