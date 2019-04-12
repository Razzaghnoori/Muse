import run

midi_group = run.MIDIGroup(midi_dir='../midi_dataset/', config_path='../exp/conditional/config.yaml')
midi_group.export('../data/train')
del midi_group
print('Network data generated successfully.')