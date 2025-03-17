from pydub import AudioSegment
import whisper

# Load Whisper model
model = whisper.load_model("large")

# Function to split and transcribe audio in chunks
def split_and_transcribe(audio_file):
    # Load the audio file
    audio = AudioSegment.from_wav(audio_file)

    # Split the audio into chunks (e.g., 30 seconds each)
    chunk_length_ms = 30000  # 30 seconds
    chunks = []
    for start_ms in range(0, len(audio), chunk_length_ms):
        end_ms = min(start_ms + chunk_length_ms, len(audio))
        chunk = audio[start_ms:end_ms]
        chunks.append(chunk)

    # Transcribe each chunk
    full_transcription = ""
    for i, chunk in enumerate(chunks):
        # Save chunk to a temporary file
        temp_chunk_path = f"chunk_{i}.wav"
        chunk.export(temp_chunk_path, format="wav")

        # Transcribe the chunk
        result = model.transcribe(temp_chunk_path, language="ar", task="translate", word_timestamps=True, fp16=False)
        
        # Combine results
        for segment in result['segments']:
            full_transcription += segment['text'] + " "
            # Optionally print word timestamps here
            for word in segment['words']:
                print(f"Word: {word['word']} | Start: {word['start']} - End: {word['end']}")

    return full_transcription

# Path to your Arabic WAV file
audio_file = "path_to_your_arabic_audio.wav"
transcription = split_and_transcribe(audio_file)
print("Full Transcription:\n", transcription)
