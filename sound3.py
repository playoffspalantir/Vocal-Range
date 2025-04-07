import streamlit as st
import numpy as np
import pyaudio
import struct
from scipy.signal import butter, lfilter, find_peaks
import matplotlib.pyplot as plt
from io import BytesIO
import sounddevice as sd
import soundfile as sf

# Configuration
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
LOW_FREQ = 60  # Lower limit for bass voice (Hz)
HIGH_FREQ = 2000  # Upper limit for soprano voice (Hz)
CHUNK = 4000
GRAPH_LOW_FREQ = 50 #New lower limit for x-axis
GRAPH_HIGH_FREQ = 1200 # Limit the graph to 1200 Hz
AMPLITUDE_THRESHOLD = 15 #Lower amplitude threshold. Adjust this.
HARMONIC_WINDOW = 15 #Window for searching for harmonics
HARMONIC_AMPLITUDE_THRESHOLD = 2  # Lower amplitude threshold *specifically for harmonics*. Adjust this.

# Voice type ranges (in Hz)
VOICE_RANGES = {
    'Bass': (60, 250),
    'Baritone': (100, 350),
    'Tenor': (130, 500),
    'Contralto': (180, 700),
    'Mezzo-Soprano': (200, 900),
    'Soprano': (250, 1100)
}

# Colors for each voice type
VOICE_COLORS = {
    'Bass': 'blue',
    'Baritone': 'cyan',
    'Tenor': 'green',
    'Contralto': 'yellow',
    'Mezzo-Soprano': 'orange',
    'Soprano': 'red'
}


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a


def bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return lfilter(b, a, data)


def estimate_fundamental_frequency(magnitude, frequencies):
    # Use argmax to find the frequency with the highest magnitude
    if len(magnitude) > 0:
        fundamental_index = np.argmax(magnitude)
        return frequencies[fundamental_index]
    return None


def classify_voice_type(fundamental_freq):
    if fundamental_freq is None:
        return "Unknown"
    for voice_type, (low, high) in VOICE_RANGES.items():
        if low <= fundamental_freq <= high:
            return voice_type
    return "Other"


def freq_to_note(freq):
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    a4_freq = 440
    a4_index = notes.index('A')
    half_steps = int(round(12 * np.log2(freq / a4_freq)))
    octave = 4 + ((half_steps + a4_index) // 12)
    note = notes[(half_steps + a4_index) % 12]
    return f"{note}{octave}"


def note_to_freq(note):
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave = int(note[-1])
    note_name = note[:-1]
    semitones_from_a4 = (octave - 4) * 12 + notes.index(note_name) - notes.index('A')
    return 440 * (2 ** (semitones_from_a4 / 12))


def record_audio(duration):
    recording = sd.rec(int(duration * RATE), samplerate=RATE, channels=CHANNELS)
    sd.wait()
    return recording.flatten()

def find_peak_near(frequencies, magnitude, target_freq, window=HARMONIC_WINDOW):
    """Finds the frequency with the highest magnitude within a window around target_freq."""
    low_freq = target_freq - window
    high_freq = target_freq + window
    indices = np.where((frequencies >= low_freq) & (frequencies <= high_freq))[0] #Find indices within the window

    if len(indices) > 0:
        peak_index = indices[np.argmax(magnitude[indices])]
        return frequencies[peak_index], magnitude[peak_index]
    return None, None

def analyze_audio(audio_data, duration, participant_notes, fundamental_freq, voice_type):
    filtered_data = bandpass_filter(audio_data, LOW_FREQ, HIGH_FREQ, RATE)

    # Averaging spectrum for better fundamental frequency estimation
    num_segments = 5  # Divide the recording into 5 segments
    segment_length = len(filtered_data) // num_segments
    frequencies = np.fft.rfftfreq(segment_length, d=1.0 / RATE)

    # Limit frequency range to GRAPH_HIGH_FREQ
    mask = (frequencies >= GRAPH_LOW_FREQ) & (frequencies <= GRAPH_HIGH_FREQ) # Apply *both* low and high limits!
    frequencies = frequencies[mask]

    averaged_magnitude = np.zeros_like(frequencies, dtype=float) # Initialize with correct length *AFTER* limiting frequencies

    for i in range(num_segments):
        start = i * segment_length
        end = (i + 1) * segment_length
        segment = filtered_data[start:end]
        fft_data = np.fft.rfft(segment)
        magnitude = np.abs(fft_data)

        # Truncate magnitude to match the length of frequencies
        magnitude = magnitude[:len(frequencies)]
        # Apply amplitude threshold here, *before* averaging.

        magnitude[magnitude < AMPLITUDE_THRESHOLD] = 0

        averaged_magnitude += magnitude


    averaged_magnitude /= num_segments


    fig, ax = plt.subplots(figsize=(14, 6))

    #Fill the area under the curve
    ax.fill_between(frequencies, averaged_magnitude, color='skyblue', alpha=0.7) #Filled Area
    ax.plot(frequencies, averaged_magnitude, label="Amplitude", color='blue') #Line on top of the filled area.
    ax.set_title(f"Voice Frequency Spectrum (Duration: {duration} sec)")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude")
    ax.set_xlim(GRAPH_LOW_FREQ, GRAPH_HIGH_FREQ) # Set the x-axis limit - both low and high!

    # Voice range overlay - only show ranges within GRAPH_HIGH_FREQ
    for voice, (low, high) in VOICE_RANGES.items():
        low = max(low, GRAPH_LOW_FREQ)  # Ensure low is not below GRAPH_LOW_FREQ
        high = min(high, GRAPH_HIGH_FREQ)  # Limit high to GRAPH_HIGH_FREQ
        if low < high: # only draw if there's a valid range
            ax.axvspan(low, high, alpha=0.2, color=VOICE_COLORS[voice], label=f'{voice} Range')


    if fundamental_freq and GRAPH_LOW_FREQ <= fundamental_freq <= GRAPH_HIGH_FREQ:  #Check if frequency within range
        ax.axvline(x=fundamental_freq, color='black', linestyle='--',
                   label=f'Fundamental Frequency: {fundamental_freq:.2f} Hz')


        # Harmonic Analysis (Peak Searching)
        for i in range(2, 6):  # Check first 4 harmonics
            harmonic_freq_ideal = fundamental_freq * i
            harmonic_freq, harmonic_magnitude = find_peak_near(frequencies, averaged_magnitude, harmonic_freq_ideal)

    if harmonic_freq and harmonic_magnitude > HARMONIC_AMPLITUDE_THRESHOLD:
        ax.axvline(x=harmonic_freq, color='purple', linestyle=':', alpha=0.7,
                   label=f'Harmonic {i}: {harmonic_freq:.2f} Hz')


    # Filter note frequencies for display
    note_freqs = [note_to_freq(note) for note in ['C2', 'C3', 'C4', 'C5', 'C6', 'C7'] if GRAPH_LOW_FREQ <= note_to_freq(note) <= GRAPH_HIGH_FREQ]
    note_labels = [f"{freq_to_note(freq)}\n{freq:.0f} Hz" for freq in note_freqs]
    ax.set_xticks(note_freqs)
    ax.set_xticklabels(note_labels, rotation=45, ha='right')

    #all_note_freqs = [note_to_freq(f"{note}{octave}") for octave in range(2, 8) for note in
    #                  ['C', 'C#', 'D', 'E', 'F', 'G', 'A', 'B']]

    all_note_freqs = [note_to_freq(f"{note}{octave}") for octave in range(2, 8) for note in ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'] if GRAPH_LOW_FREQ <= note_to_freq(f"{note}{octave}") <= GRAPH_HIGH_FREQ]

    ax.set_xticks(all_note_freqs, minor=True)



    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, which='both', linestyle=':', linewidth=0.5)
    plt.tight_layout()

    # Add participant notes
    fig.text(0.98, 0.02, participant_notes, ha='right', va='bottom', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

    # Add fundamental frequency and voice type
    if fundamental_freq:
        fig.text(0.98, 0.06, f"Fundamental Frequency: {fundamental_freq:.2f} Hz", ha='right', va='bottom', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
        fig.text(0.98, 0.10, f"Nearest Musical Note: {freq_to_note(fundamental_freq)}", ha='right', va='bottom', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
        fig.text(0.98, 0.14, f"Estimated Voice Type: {voice_type}", ha='right', va='bottom', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

    return fig


def main():
    st.title("Zoey's Patented Voice Type Analyzer")

    duration = st.slider("Recording Duration (seconds)", 1, 30, 5)
    participant_notes = st.text_area("Participant Notes:", height=200)

    if st.button("Record and Analyze"):
        st.write("Recording... Please sing or speak into your microphone.")
        audio_data = record_audio(duration)

        #Bandpass Filter first, calculate frequencies, then estimate F0
        filtered_data = bandpass_filter(audio_data, LOW_FREQ, HIGH_FREQ, RATE)
        frequencies = np.fft.rfftfreq(len(filtered_data), d=1.0 / RATE)

        # Limit frequency range to GRAPH_HIGH_FREQ BEFORE estimating F0
        mask = (frequencies >= GRAPH_LOW_FREQ) & (frequencies <= GRAPH_HIGH_FREQ)  # Apply *both* low and high limits!
        frequencies = frequencies[mask]

        fft_data = np.fft.rfft(filtered_data)
        magnitude = np.abs(fft_data)
        # Truncate magnitude to match the length of frequencies
        magnitude = magnitude[:len(frequencies)] #critical

        magnitude_threshold = 60.0
        magnitude[magnitude < magnitude_threshold] = 0
        fundamental_freq = estimate_fundamental_frequency(magnitude, frequencies)


        st.write("Recording complete. Analyzing...")

        voice_type = classify_voice_type(fundamental_freq)

        fig = analyze_audio(filtered_data, duration, participant_notes, fundamental_freq, voice_type)

        st.pyplot(fig)

        # Save audio option
        wav_file = BytesIO()
        sf.write(wav_file, audio_data, RATE, format='wav')
        st.download_button(
            label="Download Audio",
            data=wav_file.getvalue(),
            file_name="recorded_audio.wav",
            mime="audio/wav"
        )


if __name__ == "__main__":
    main()