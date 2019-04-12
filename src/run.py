import numpy as np
import traceback

from os import walk
from os.path import join
from pypianoroll import Multitrack
from strategies import map_to_closest
from musegan.utils import load_yaml
from musegan.dirohe import Dirohe

def npz_to_midi(npz_path, midi_path):
    Multitrack(npz_path).write(midi_path)
    print('Successfully created midi file.')

def midi_to_npz(midi_path, npz_path):
    Multitrack(midi_path).save(npz_path)


class MIDI(object):
    """
    Load, transform, and export midi files
    """

    def __init__(self, midi_path, beat_res=12):
        self.multitrack = Multitrack(midi_path, beat_resolution=beat_res)
        self.multitrack.pad_to_multiple(4 * beat_res)
        self.multitrack.binarize()
        self.beat_res = beat_res

    def normalize(self, target_programs, strategies_=['closest']):
        for strategy in strategies_:
            if strategy == 'closest':
                self.multitrack = map_to_closest(self.multitrack, target_programs)

    def compute_pianoroll(self):
        self.num_tracks = len(self.multitrack.tracks)
        self.pianoroll = self.multitrack.get_stacked_pianorolls()

        self.pianoroll = self.pianoroll[:, 24:108]
        self.pianoroll = self.pianoroll.reshape(-1, 4 * self.beat_res, \
            84, self.num_tracks)

    def export(self, exp_path):
        np.savez_compressed(
            exp_path, nonzero=np.array(self.pianoroll.nonzero()),
            shape=self.pianoroll.shape)
        print('Successfully compressed and saved the pianoroll.')

class MIDIGroup(object):
    def __init__(self, midi_list=None, midi_dir=None, \
        config_path='./src/musegan/default_config.yaml', has_labels=True):
        self.list_midis = []
        self.config = load_yaml(config_path)
        self.has_labels = has_labels
        
        if self.has_labels:
            if not midi_dir:
                raise Exception('Must provide midi_dir when expecting labels.')
            self.dirohe = Dirohe(midi_dir, filter_='dirs')
            self.labels = []

        if midi_list is not None:
            self.list_midis = midi_list
            self.list_pianorolls = [x.pianoroll for x in self.list_midis]
        elif midi_dir is not None:
            for dir, _, midi_list in walk(midi_dir):
                for x in midi_list:
                    if not x.endswith('mid'):
                        continue
                    try:
                        midi_path = join(dir, x)
                        midi = self._get_normalized_MIDI(midi_path)
                        self.list_midis.append(midi)

                        if self.has_labels:
                            label = self.dirohe.encode(midi_path)
                            self.labels.append(label)

                    except:
                        print('Crap')
                        traceback.print_exc()
                        continue

            self.list_pianorolls = [x.pianoroll for x in self.list_midis]
            self.labels = np.array(self.labels)

    def _get_normalized_MIDI(self, midi_path):
        midi = MIDI(midi_path)
        midi.normalize(self.config['midi']['programs'])
        midi.compute_pianoroll()
        return midi
        
    def _generate_pianoroll(self):
        batch_size = len(self.list_midis)
        n_dims = self.list_pianorolls[0].ndim

        shapes = np.array([x.shape for x in self.list_pianorolls])
        max_of_each_axis = np.max(shapes, axis=0)
        pad_requirements = max_of_each_axis - shapes
        pad_requirements = pad_requirements.reshape(-1,1)
        pad_requirements = np.hstack([np.zeros((pad_requirements.size,1)), \
            pad_requirements]).reshape(-1, n_dims, 2)

        #(batch_size, n_bars, n_timesteps, n_pitches, n_tracks)
        self.pianorolls = np.zeros((batch_size, *max_of_each_axis))

        print('Shape of pianorolls:', self.pianorolls.shape)

        for i in range(batch_size):
            self.pianorolls[i] = np.pad(self.list_pianorolls[i], \
                pad_requirements[i].astype(int), mode='constant') 

    def export(self, exp_path):
        try:
            self._generate_pianoroll()
            np.savez_compressed(
                exp_path, nonzero=np.array(self.pianorolls.nonzero()),
                shape=self.pianorolls.shape)

            if self.has_labels:
                np.save(exp_path + '_labels', self.labels)
            
            print('Successfully created a compressed dataset of Midi files.')
        except:
            print('Export failed.')