import itertools
from pathlib import Path
from typing import Dict, Tuple, List, Union
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
from transforms import BVHtoMocapData, MocapDataToExpMap, Pipeline, AudioToLogMelSpec
import librosa
from tqdm import tqdm
import pickle

Number = Union[float, int]


class WavBVHDataset(Dataset):

    def __getitem__(self, index) -> T_co:
        item = torch.load(self.dataset[index])
        return index, item['audio'].float(), item['gesture'].permute(1, 0).float()

    def __len__(self):
        return min(len(self.dataset), self.size)

    def __init__(self, dataset: Path, gesture_fps=60, clip_duration=4, group='train', size='all', repeat=1, transcripts=False):
        self.audio_dir: Path = dataset / 'raw_data' / group / 'Audio'
        self.motion_dir: Path = dataset / 'raw_data' / group / 'Motion'
        self.save_dir: Path = dataset / 'processed_data' / group
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_path: Path = dataset / f'{group}_info.pickle'
        self.gesture_fps: Number = gesture_fps
        self.clip_duration: int = clip_duration
        self.mocap_pipeline = Pipeline([BVHtoMocapData, MocapDataToExpMap])
        self.mel_spec = AudioToLogMelSpec()
        if self.dataset_path.is_file():
            with open(self.dataset_path, 'rb') as f:
                self.dataset = pickle.load(f)
        else:
            self.dataset: List[Path] = self.build_dataset()
            with open(self.dataset_path, 'wb') as f:
                pickle.dump(self.dataset, f)
        self.size = (len(self.dataset) if size == 'all' else size) * repeat
        if repeat > 1:
            self.dataset = list(itertools.chain.from_iterable(itertools.repeat(x, repeat) for x in
                                                              self.dataset))

    def build_dataset(self):
        dataset = []
        for audio_file in tqdm(list(self.audio_dir.iterdir())):
            splits = audio_file.name.split('.')
            name, ext = '.'.join(splits[:-1]), splits[-1]
            if ext != 'wav':
                continue
            bvh_file = self.motion_dir / (name + '.bvh')
            if not audio_file.is_file() or not bvh_file.is_file():
                print(f'not found: {audio_file}')
                continue
            audio_end = int(self.clip_duration * ((librosa.get_duration(filename=str(audio_file)) // self.clip_duration)
                                                  - 1))
            windows = range(0, audio_end, self.clip_duration // 2)
            if (self.save_dir / f'{name}_{len(windows) - 1}.pt').is_file():
                dataset.append(dest)
                continue
            exp_map = self.mocap_pipeline.apply(bvh_file)
            frame = 0
            frame_window, frame_step = self.gesture_fps * self.clip_duration, self.gesture_fps * self.clip_duration // 2
            for i, t in enumerate(windows):
                dest = self.save_dir / f'{name}_{i}.pt'
                a = librosa.load(audio_file, offset=t, duration=self.clip_duration, mono=True)
                audio = self.mel_spec.apply(a)
                gesture = exp_map[frame:frame + frame_window, :]
                if gesture.shape[0] != frame_window:
                    print('skipping', dest)
                    continue
                torch.save({'audio': torch.from_numpy(audio), 'gesture': torch.from_numpy(gesture)}, dest)
                dataset.append(dest)
                frame += frame_step
        return dataset



import pickle as pkl
import numpy as np
import os
from .dataset import Dataset
import pickle
from transforms import BVHtoMocapData, MocapDataToExpMap, Pipeline, AudioToLogMelSpec
import librosa
import os
from tqdm import tqdm

# currently Audio, Motion, Facial Animation.
class MikeDataset(Dataset):
    dataname = "mike"

    def __init__(self, datapath="data/Mike", **kargs):
        self.datapath = datapath
        # hardcoded now for easy use
        self.train_name_list = [
            "Mike_Lecture_Part01",
            "Mike_Lecture_Part02",
            "Mike_Lecture_Part03",
            "Mike_Lecture_Part04",
            "Mike_Lecture_Part05",
            "Mike_Lecture_Part06",
            "Mike_Lecture_Part07",
            "Mike_Lecture_Part08",
            "Mike_Lecture_Part09",
            "Mike_Lecture_Part10",
            "Mike_Lecture_Part11",
            "Mike_Lecture_Part12",
            "Mike_Lecture_Part13",
            "Mike_Lecture_Part14",
            "Mike_Lecture_Part15",
            "Mike_Lecture_Part16",
            "Mike_Lecture_Part17",
            "Mike_Lecture_Part18",
            "Take2_Finance_1202-22802",
            "Take3_Cinematography_728-24128",
            "Take4_RelaxedConversation_1781-25182",
            "Take5_Idle_1185-23025_GoPro",
            "Take6-Presentation_529-24409",
            "Take7-RelaxedConversation_1027-25388",
        ]

        self.test_name_list = [
            "Mike_Fullbody_Mocap_Take1",
            "Mike_Fullbody_Mocap_Take2",
            "Mike_Fullbody_Mocap_Take3",
        ]
        self.all_name_list = self.train_name_list + self.test_name_list

        super().__init__(**kargs)

        # TODO: self._train, self._test

        # filename list
        # train_list = []
        # test_list = []

        # read motion data
        self.motion_data_cache_path = os.path.join(self.datapath, "motion_data_cache.bin")
        self.audio_data_cache_path = os.path.join(self.datapath, "audio_data_cache.bin")

        if os.path.isfile(self.motion_data_cache_path):
            with open(self.motion_data_cache_path, 'rb') as f:
                self.motion_data_cache = pickle.load(f)
        else:
            self.motion_data_cache = self.build_motion_cache()
            with open(self.motion_data_cache_path, 'wb') as f:
                pickle.dump(self.motion_data_cache, f)

        # TODO: if no preprocessed audio, preprocess audio
        # to mel spectrum
        # preprocessing audio

        # TODO: read the audio data



        # TODO: if no preprocessed facial, preprocess facial data
        # TODO: read the facial data

    def build_motion_cache(self):
        # TODO: preprocess the bvh data
        bvh_file_dir = os.path.join(self.datapath, "BVH")
        motion_data = []
        for name in tqdm(self.all_name_list):
            bvh_file_path = os.path.join(bvh_file_dir, name + '.bvh')
            assert os.path.isfile(bvh_file_path)
            


        # preprocessing bvh
        # copy 2nd frame to 1st frame
        # normalize values based on Min Max
        pass



    def build_dataset(self):
        dataset = []
        for audio_file in tqdm(list(self.audio_dir.iterdir())):
            splits = audio_file.name.split('.')
            name, ext = '.'.join(splits[:-1]), splits[-1]
            if ext != 'wav':
                continue
            bvh_file = self.motion_dir / (name + '.bvh')
            if not audio_file.is_file() or not bvh_file.is_file():
                print(f'not found: {audio_file}')
                continue
            audio_end = int(self.clip_duration * ((librosa.get_duration(filename=str(audio_file)) // self.clip_duration)
                                                  - 1))
            windows = range(0, audio_end, self.clip_duration // 2)
            if (self.save_dir / f'{name}_{len(windows) - 1}.pt').is_file():
                dataset.append(dest)
                continue
            exp_map = self.mocap_pipeline.apply(bvh_file)
            frame = 0
            frame_window, frame_step = self.gesture_fps * self.clip_duration, self.gesture_fps * self.clip_duration // 2
            for i, t in enumerate(windows):
                dest = self.save_dir / f'{name}_{i}.pt'
                a = librosa.load(audio_file, offset=t, duration=self.clip_duration, mono=True)
                audio = self.mel_spec.apply(a)
                gesture = exp_map[frame:frame + frame_window, :]
                if gesture.shape[0] != frame_window:
                    print('skipping', dest)
                    continue
                torch.save({'audio': torch.from_numpy(audio), 'gesture': torch.from_numpy(gesture)}, dest)
                dataset.append(dest)
                frame += frame_step
        return dataset