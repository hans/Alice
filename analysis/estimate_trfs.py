"""This script estimates TRFs for several models and saves them"""
from argparse import ArgumentParser
from pathlib import Path
import re
import sys

import eelbrain
import mne
import trftools
from tqdm import tqdm


STIMULI = [str(i) for i in range(1, 13)]

def build_models(stimuli, data_dir):
    predictor_dir = data_dir / "predictors"

    # Load stimuli
    # ------------
    # Make sure to name the stimuli so that the TRFs can later be distinguished
    # Load the gammatone-spectrograms; use the time axis of these as reference
    gammatone = [eelbrain.load.unpickle(predictor_dir / f'{stimulus}~gammatone-8.pickle') for stimulus in stimuli]
    # Resample the spectrograms to 100 Hz (time-step = 0.01 s), which we will use for TRFs
    gammatone = [x.bin(0.01, dim='time', label='start') for x in gammatone]
    # Pad onset with 100 ms and offset with 1 second; make sure to give the predictor a unique name as that will make it easier to identify the TRF later
    gammatone = [trftools.pad(x, tstart=-0.100, tstop=x.time.tstop + 1, name='gammatone') for x in gammatone]
    # Load the broad-band envelope and process it in the same way
    envelope = [eelbrain.load.unpickle(predictor_dir / f'{stimulus}~gammatone-1.pickle') for stimulus in stimuli]
    envelope = [x.bin(0.01, dim='time', label='start') for x in envelope]
    envelope = [trftools.pad(x, tstart=-0.100, tstop=x.time.tstop + 1, name='envelope') for x in envelope]
    onset_envelope = [eelbrain.load.unpickle(predictor_dir / f'{stimulus}~gammatone-on-1.pickle') for stimulus in stimuli]
    onset_envelope = [x.bin(0.01, dim='time', label='start') for x in onset_envelope]
    onset_envelope = [trftools.pad(x, tstart=-0.100, tstop=x.time.tstop + 1, name='onset') for x in onset_envelope]
    # Load onset spectrograms and make sure the time dimension is equal to the gammatone spectrograms
    gammatone_onsets = [eelbrain.load.unpickle(predictor_dir / f'{stimulus}~gammatone-on-8.pickle') for stimulus in stimuli]
    gammatone_onsets = [x.bin(0.01, dim='time', label='start') for x in gammatone_onsets]
    gammatone_onsets = [eelbrain.set_time(x, gt.time, name='gammatone_on') for x, gt in zip(gammatone_onsets, gammatone)]
    # Load word tables and convert tables into continuous time-series with matching time dimension
    word_tables = [eelbrain.load.unpickle(predictor_dir / f'{stimulus}~word.pickle') for stimulus in stimuli]
    word_onsets = [eelbrain.event_impulse_predictor(gt.time, ds=ds, name='word') for gt, ds in zip(gammatone, word_tables)]
    # Function and content word impulses based on the boolean variables in the word-tables
    word_lexical = [eelbrain.event_impulse_predictor(gt.time, value='lexical', ds=ds, name='lexical') for gt, ds in zip(gammatone, word_tables)]
    word_nlexical = [eelbrain.event_impulse_predictor(gt.time, value='nlexical', ds=ds, name='non_lexical') for gt, ds in zip(gammatone, word_tables)]

    # Extract the duration of the stimuli, so we can later match the EEG to the stimuli
    durations = [gt.time.tmax for stimulus, gt in zip(stimuli, gammatone)]

    # Models
    # ------
    # Pre-define models here to have easier access during estimation. In the future, additional models could be added here and the script re-run to generate additional TRFs.
    models = {
        # Acoustic models
        'envelope': [envelope],
        'envelope+onset': [envelope, onset_envelope],
        'acoustic': [gammatone, gammatone_onsets],
        # Models with word-onsets and word-class
        'words': [word_onsets],
        'words+lexical': [word_onsets, word_lexical, word_nlexical],
        'acoustic+words': [gammatone, gammatone_onsets, word_onsets],
        'acoustic+words+lexical': [gammatone, gammatone_onsets, word_onsets, word_lexical, word_nlexical],
    }

    return models, durations


def estimate_trf(subject, eeg_dir, stimuli, durations, models, out_dir):
    """
    Estimate TRFs for a single subject.

    Args:
        subject: subject ID string
        eeg_dir: path to Brennan EEG dir
        models: dict of mapping string names to lists of predictors
        out_dir: path where unique subject--model results should be saved
    """
    subject_path  = eeg_dir / subject / f"{subject}_alice-raw.fif"

    # Generate all TRF paths so we can check whether any new TRFs need to be estimated
    trf_paths = {model: out_dir / f'{subject} {model}.pickle' for model in models}
    # Skip this subject if all files already exist
    if all(path.exists() for path in trf_paths.values()):
        return

    # Load the EEG data
    raw = mne.io.read_raw(subject_path, preload=True)

    # Band-pass filter the raw data between 0.2 and 20 Hz
    raw.filter(0.5, 20)
    # Interpolate bad channels
    raw.interpolate_bads()

    # Extract the events marking the stimulus presentation from the EEG file
    events = eelbrain.load.fiff.events(raw)
    # Not all subjects have all trials; determine which stimuli are present
    trial_indexes = [stimuli.index(stimulus) for stimulus in events['event']]
    # Extract the EEG data segments corresponding to the stimuli
    trial_durations = [durations[i] for i in trial_indexes]

    eeg = eelbrain.load.fiff.variable_length_epochs(events, -0.100, trial_durations, decim=5, connectivity='auto')
    # Since trials are of unequal length, we will concatenate them for the TRF estimation.
    eeg_concatenated = eelbrain.concatenate(eeg)
    for model, predictors in tqdm(models.items(), total=len(models), desc="models"):
        path = trf_paths[model]
        # Skip if this file already exists
        if path.exists():
            continue
        print(f"Estimating: {subject} ~ {model}")
        # Select and concetenate the predictors corresponding to the EEG trials
        predictors_concatenated = []
        for predictor in predictors:
            predictors_concatenated.append(eelbrain.concatenate([predictor[i] for i in trial_indexes]))
        # Fit the mTRF
        trf = eelbrain.boosting(eeg_concatenated, predictors_concatenated, -0.100, 1.000, error='l1', basis=0.050, partitions=5, test=1, selective_stopping=True)
        # Save the TRF for later analysis
        eelbrain.save.pickle(trf, path)


def main(args):
    args.out_dir.mkdir(exist_ok=True)

    models, durations = build_models(STIMULI, args.data_dir)

    eeg_dir = args.data_dir / "eeg"
    subjects = [path.name for path in eeg_dir.iterdir()
                if re.match(r'S\d*', path.name)]

    for subject in tqdm(subjects, desc="subjects"):
        estimate_trf(subject, eeg_dir, stimuli=STIMULI, durations=durations,
                     models=models, out_dir=args.out_dir)


if __name__ == "__main__":
    p = ArgumentParser()

    p.add_argument("data_dir", type=Path)
    p.add_argument("-o", "--out_dir", type=Path, required=True)

    main(p.parse_args())
