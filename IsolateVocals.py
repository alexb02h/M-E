import torch, torchaudio, librosa
from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB_PLUS
from pyannote.audio import Pipeline

def separate_sources(model, waveform, device, sample_rate, segment=10, overlap=1) : 
    segment_samples = int(segment * sample_rate)
    overlap_samples = int(overlap * sample_rate)
    total_samples = waveform.shape[-1]
    step = segment_samples - overlap_samples

    outputs, positions = [], []
    start = 0
    
    while start < total_samples : 
        end = min(start + segment_samples, total_samples)
        chunk = waveform[:, start:end]
        
        original_length = chunk.shape[-1]
        if original_length < segment_samples : 
            chunk = torch.nn.functional.pad(chunk, (0, segment_samples - original_length))
        
        chunk = chunk.unsqueeze(0).to(device)
        
        with torch.no_grad() : separated = model(chunk)
        
        separated = separated[0]
        separated = separated[..., :original_length]
        
        outputs.append(separated.cpu())
        positions.append(start)
        
        del chunk, separated
        
        if device.type == "mps" : torch.mps.empty_cache()
        
        start += step
    
    num_sources = outputs[0].shape[0]
    num_channels = outputs[0].shape[1]
    
    separated_audio = torch.zeros(num_sources, num_channels, total_samples)
    weights = torch.zeros(total_samples)
    
    for output, start in zip(outputs, positions) : 
        length = output.shape[-1]
        end = min(start + length, total_samples)
        
        actual_length = end - start
        separated_audio[..., start:end] += output[..., :actual_length]
        
        weights[start:end] += 1
        
    separated_audio /= weights.clamp_min(1).view(1, 1, -1)
    return separated_audio

def separate_speakers(vocal_waveform, sample_rate, output_pre, hf_token) :
    temp_vocal = "temp_vocals.wav"
    torchaudio.save(temp_vocal, vocal_waveform, sample_rate)
    
    pipeline = Pipeline.from_pretrained("pyannote/speech-separation-ami-1.0", use_auth_token=hf_token)
    output = pipeline(temp_vocal)
    
    for speaker_label, speaker_audio in output.iter_tracks() : 
        out_file = f"{output_pre}_{speaker_label}.wav"
        torchaudio.save(out_file, speaker_audio, sample_rate)

def create_me_stem(input_path, output_me, device, hf_token) :
    if device == "mps" and torch.backends.mps.is_available() : device = torch.device("mps")
    elif device == "cuda" and torch.cuda.is_available() : device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else : device = torch.device("cpu")
    
    bundle = HDEMUCS_HIGH_MUSDB_PLUS
    model = bundle.get_model().to(device)
    model.eval()
    
    waveform, sample_rate = librosa.load(input_path, sr=None, mono=False)
    waveform = torch.from_numpy(waveform)
    if waveform.ndim == 1 : waveform = waveform.unsqueeze(0)
    
    if sample_rate != bundle.sample_rate :
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=bundle.sample_rate)
        waveform = resampler(waveform)
        sample_rate = bundle.sample_rate
        
    separated_stems = separate_sources(model=model, waveform=waveform, device=device, sample_rate=sample_rate, segment=5, overlap=1)
    
    drums = separated_stems[0]
    bass = separated_stems[1]
    other = separated_stems[2]
    vocals = separated_stems[3]
        
    me_waveform = drums + bass + other
    torchaudio.save(output_me, me_waveform, sample_rate)
    
    separate_speakers(vocals, sample_rate, "speaker_vocal", hf_token)
    
if __name__ == "__main__" : create_me_stem(input="", output_me="M_and_E.wav", device="mps", hf_token="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")