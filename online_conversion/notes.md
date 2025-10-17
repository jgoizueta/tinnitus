## MatLab files

### Generate_Full_Experiment_Stimuli.m

generate stimuli mp4 files; 
n=1 files per all types (noise/amplitude-modulation/phase-modulation),
all frequency bands,
all correction values (0, 15, 30, 45 max. dB difference between 8 kHz and 1 Hz)
names can be randomized if randnames=1 (for the study purposes, we don't need that)

it uses the mod_ripple to generate the 4s audio segments and concatenates 900 of those in each file for 1h of sound

### Hearing_Tin_Stim_Generation.m

generates mp3 files to estimate the tinnitus frequency and hearing loss slope

generates files per correction (0-NH, 30-MildHL, 50-ModHL, 70-SevHL)
  [Normal Hearing / Mild Hearing Loss / Moderate Hearing Loss / Severe Hearing Loss]
for PT for tonal tinnitus using sin waves and NBN (narrowband noise) using the farpn function
for each of 17 frequencies in ascending order 1-16 kHz in 1/4 octave steps
(1, 1.2, 1.4, 1.7, 2, 2.4, 2.8, 3.4, 4.0, 4.8, 5.7, 6.7, 8.0, 9.5, 11.0, 13.0, 16.0 kHz)
