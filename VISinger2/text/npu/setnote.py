import librosa

def generate_all_notes():
    # Define the range of MIDI note numbers
    min_midi = 0  # C-1
    max_midi = 127  # G9
    
    all_notes = set()
    
    for midi_note in range(min_midi, max_midi + 1):
        # Convert MIDI note to frequency
        frequency = librosa.midi_to_hz(midi_note)
        # Convert frequency to note name
        note_name = librosa.hz_to_note(frequency)
        all_notes.add(note_name)
    
    return sorted(all_notes)

# Get all possible notes
all_possible_notes = generate_all_notes()
print(all_possible_notes)
