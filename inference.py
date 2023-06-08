# coding: utf-8
__author__ = 'https://github.com/ZFTurbo/'

if __name__ == '__main__':
    import os

    gpu_use = "0"
    print('GPU use: {}'.format(gpu_use))
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_use)


import numpy as np
import torch
import os
import argparse
import soundfile as sf

from demucs.states import load_model
from demucs.apply import apply_model
from time import time
import librosa


class Demucs4_SeparationModel:

    def __init__(self, options):
        if torch.cuda.is_available():
            device = 'cuda:0'
        else:
            device = 'cpu'
        if 'cpu' in options:
            if options['cpu']:
                device = 'cpu'
        self.device = device

        self.model_list = [
            '97d170e1-dbb4db15.th',
        ]
        if 'high_quality' in options:
            if options['high_quality']:
                print('Use 3 checkpoints!')
                self.model_list = [
                    '97d170e1-a778de4a.th',
                    '97d170e1-dbb4db15.th',
                    '97d170e1-e41a5468.th'
                ]

        self.models = []
        models_folder = os.path.dirname(os.path.abspath(__file__)) + '/models/'
        if not os.path.isdir(models_folder):
            os.mkdir(models_folder)
        for model_name in self.model_list:
            model_path = models_folder + model_name
            if not os.path.isfile(model_path):
                remote_url = 'https://github.com/ZFTurbo/MVSEP-CDX23-Cinematic-Sound-Demixing/releases/download/v.1.0.0/' + model_name
                torch.hub.download_url_to_file(remote_url, model_path)
            model = load_model(model_path)
            model.to(device)
            self.models.append(model)

        self.device = device
        pass

    @property
    def instruments(self):
        return ['dialog', 'effect', 'music']

    def raise_aicrowd_error(self, msg):
        raise NameError(msg)

    def separate_music_file(
            self,
            mixed_sound_array,
            sample_rate,
            update_percent_func=None,
            current_file_number=0,
            total_files=0,
    ):
        """
        Implements the sound separation for a single sound file
        Inputs: Outputs from soundfile.read('mixture.wav')
            mixed_sound_array
            sample_rate

        Outputs:
            separated_music_arrays: Dictionary numpy array of each separated instrument
            output_sample_rates: Dictionary of sample rates separated sequence
        """

        separated_music_arrays = {}
        output_sample_rates = {}

        audio = np.expand_dims(mixed_sound_array.T, axis=0)
        audio = torch.from_numpy(audio).type('torch.FloatTensor').to(self.device)

        all_out = []
        for model in self.models:
            out = apply_model(model, audio, shifts=1, overlap=0.8)[0].cpu().numpy()
            all_out.append(out)
        dnr_demucs = np.array(all_out).mean(axis=0)

        # dialog
        separated_music_arrays['dialog'] = dnr_demucs[2].T
        output_sample_rates['dialog'] = sample_rate

        # music
        separated_music_arrays['music'] = dnr_demucs[0].T
        output_sample_rates['music'] = sample_rate

        # effect
        separated_music_arrays['effect'] = dnr_demucs[1].T
        output_sample_rates['effect'] = sample_rate

        return separated_music_arrays, output_sample_rates


def predict_with_model(options):
    for input_audio in options['input_audio']:
        if not os.path.isfile(input_audio):
            print('Error. No such file: {}. Please check path!'.format(input_audio))
            return
    output_folder = options['output_folder']
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)

    model = Demucs4_SeparationModel(options)

    update_percent_func = None
    if 'update_percent_func' in options:
        update_percent_func = options['update_percent_func']

    for i, input_audio in enumerate(options['input_audio']):
        print('Go for: {}'.format(input_audio))
        audio, sr = librosa.load(input_audio, mono=False, sr=44100)
        if len(audio.shape) == 1:
            audio = np.stack([audio, audio], axis=0)
        print("Input audio: {} Sample rate: {}".format(audio.shape, sr))
        result, sample_rates = model.separate_music_file(audio.T, sr, update_percent_func, i, len(options['input_audio']))
        for instrum in model.instruments:
            output_name = os.path.splitext(os.path.basename(input_audio))[0] + '_{}.wav'.format(instrum)
            sf.write(output_folder + '/' + output_name, result[instrum], sample_rates[instrum], subtype='FLOAT')
            print('File created: {}'.format(output_folder + '/' + output_name))

        # instrumental part 1
        inst = audio.T - result['dialog']
        output_name = os.path.splitext(os.path.basename(input_audio))[0] + '_{}.wav'.format('instrum')
        sf.write(output_folder + '/' + output_name, inst, sr, subtype='FLOAT')
        print('File created: {}'.format(output_folder + '/' + output_name))

        # instrumental part 2
        inst2 = result['music'] + result['effect']
        output_name = os.path.splitext(os.path.basename(input_audio))[0] + '_{}.wav'.format('instrum2')
        sf.write(output_folder + '/' + output_name, inst2, sr, subtype='FLOAT')
        print('File created: {}'.format(output_folder + '/' + output_name))

    if update_percent_func is not None:
        val = 100
        update_percent_func(int(val))


if __name__ == '__main__':
    start_time = time()

    m = argparse.ArgumentParser()
    m.add_argument("--input_audio", "-i", nargs='+', type=str, help="Input audio location. You can provide multiple files at once", required=True)
    m.add_argument("--output_folder", "-r", type=str, help="Output audio folder", required=True)
    m.add_argument("--cpu", action='store_true', help="Choose CPU instead of GPU for processing. Can be very slow.")
    m.add_argument("--high_quality", action='store_true', help="Use 3 checkpoints. Will be 3 times slower.")

    options = m.parse_args().__dict__
    print("Options: ".format(options))
    for el in options:
        print('{}: {}'.format(el, options[el]))
    predict_with_model(options)
    print('Time: {:.0f} sec'.format(time() - start_time))
    print('Presented by https://mvsep.com')


"""
Example:
    python inference.py
    --input_audio 159_mix.wav 312_mix.wav
    --output_folder ./results/
    --cpu
    --high_quality
"""