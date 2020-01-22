import pandas as pd
import numpy as np
from scipy.io.wavfile import write

import mat2py
import porcc
import sound_click


# # Sound Files
# soundtrap_path = "C:/Users/cleap/Documents/Data/Sound Data/SoundTrap/BelwindTest"
# soundtrap_model = "ST300HF"
# serial_number = 67416073

# # Read the calibration file and the sound file with the recorded tags
# calibration = sound_click.read_calibration(serial_number)


# Configuration
models_path = 'models/models.ini'
hq_model_path = 'data/cosentino/trainHQ_data.mat'
lq_model_path = 'data/cosentino/trainLQ_data.mat'
test_model_path = 'data/cosentino/classified_data.mat'
fs_data = 500000
nfft = 512


if __name__ == "__main__":
    """
    Start a PorCC study and apply the classifier to all the sound files 
    """
    write('data/cosentino/standard_click.wav', fs_data, np.array(click_model))
    # porcc_al = porcc.PorCC(load_type='custom', hq_mod=hq_mod, lq_mod=lq_mod, hq_params=hq_params, lq_params=lq_params, fs=fs_data, click_model_path=click_model_path)
    # porcc_al.test_classification_vs_matlab(models.Test)

    # porcc = porcc.PorCC(porcc_config)
    # sound_file = sound_click.SoundTrapFile(sound_folder)

    # # Run the script in every file of the folder
    # for day_folder_name in os.listdir(soundtrap_path):
    #     day_folder_path = os.path.join(soundtrap_path, day_folder_name)
    #     zip_day_folder = zipfile.ZipFile(day_folder_path, 'r', allowZip64=True)

    #     for file_name in zip_day_folder.namelist():
    #         extension = file_name.split(".")[-1]
    #         if extension == 'wav':
    #             print(file_name)
    #             # Read both the wav and the xml file with the same name
    #             wav_file = zip_day_folder.open(file_name)
    #             xml_file = zip_day_folder.open(file_name.replace('wav', 'log') + '.xml')
    #             sound_file = porpoises_soundtrap.SoundTrapFile(wav_file, xml_file, calibration, last_gain='High')
    #             possible_clicks = sound_file.detect_possible_click_zones(lowcut, highcut, min_separation_time)

    #             for zone in possible_clicks: 
    #                 click = sound_file.zone2click(zone)
    #                 class_type = porcc.classify(click)
    #                 clicks_df.loc[click.timestamp] = [click.cf, click.pf, class_type]

    # clicks_df.to_csv('detected_clicks.csv')
    print('hola')

    # models.test_click_calculation(save=True)



def create_and_save_models():
    # Load the pickle data and calculate the models
    df_hq, df_lq, df_test = mat2py.load_pickle_data([hq_model_path, lq_model_path, test_model_path])
    models = porcc.ClickModel(click_model, fs=fs_data, HQ=df_hq, LQ=df_lq, Test=df_test)

    hq_params, hq_mod = models.find_best_model('HQ')
    lq_params, lq_mod = models.find_best_model('LQ')