from pypianoroll import Multitrack
from math import inf
from copy import deepcopy


def map_to_closest(multitrack, target_programs, match_len=True, drums_first=True):
    """
    Keep closest tracks to the target_programs and map them to corresponding
    programs in available in target_programs.

    multitrack (pypianoroll.Multitrack): Track to normalize.
    target_programs (list): List of available programs.
    match_len (bool): If True set multitrack track length to length of target_programs.
        (return only the len(target_programs) closest tracks in multitrack).
    """

    new_multitrack = deepcopy(multitrack)

    for track in new_multitrack.tracks:
        min_dist = inf
        for target in target_programs:
            dist = abs(track.program - target)
            if dist < min_dist:
                min_dist = dist
                track.program = target
        track.min_dist = min_dist

    if match_len:
        length = len(target_programs)
        new_multitrack.tracks.sort(key=lambda x: x.min_dist)
        new_multitrack.tracks = new_multitrack.tracks[:length]
        
    if drums_first:
        new_multitrack.tracks.sort(key=lambda x: not x.is_drum)

    return new_multitrack

