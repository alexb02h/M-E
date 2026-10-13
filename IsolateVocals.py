import torch, torchaudio, librosa, os
import numpy as np
from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB_PLUS
from pyannote.audio import Pipeline, Model, Inference
from pyannote.audio.pipelines.utils.hook import ProgressHook
from scipy.spatial.distance import cosine

def is_silent(waveform, threshold=-30.0) : 
    rms = torch.sqrt(torch.mean(waveform.float() ** 2)).item()
    db = 20 * torch.log10(torch.tensor(max(rms, 1e-10))).item()
    
    print(f"Segment RMS: {db:.1f} dBFS", flush=True)
    return db < threshold

def load_embedding_model(hf_token) : 
    model = Model.from_pretrained("pyannote/embedding", use_auth_token=hf_token)
    inference = Inference(model, window="whole")
    inference.to(torch.device("cpu"))
    return inference

def save_embedding(waveform, sample_rate, speaker_id, output_dir, embedding_inference) :
    os.makedirs(output_dir, exist_ok=True)
    audio = {"waveform" : waveform.cpu(), "sample_rate" : sample_rate}
    
    embedding = embedding_inference(audio)
    embedding = np.asarray(embedding).squeeze()
    embedding_path = os.path.join(output_dir, f"{speaker_id}_embedding.npy")
    np.save(embedding_path, embedding)
    
    audio_path = os.path.join(output_dir, f"{speaker_id}_reference.wav")
    torchaudio.save(audio_path, waveform.cpu(), sample_rate)
    
    print(f"Saved embedding: {embedding_path}", flush=True)
    print(f"Saved reference audio: {audio_path}", flush=True)
    return embedding

def match_speaker(new_embedding, speaker_embeddings) : 
    if not speaker_embeddings : return None, None
    
    scores = {speaker_id : 1.0 - cosine(new_embedding, reference) for speaker_id, reference in speaker_embeddings.items()}
    most_likely = max(scores, key=scores.get)
    return most_likely, scores[most_likely]

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

def separate_speakers(vocal_waveform, output_dir, hf_token) :
    os.makedirs(output_dir, exist_ok=True)
    waveform, sample_rate = torchaudio.load(vocal_waveform)
    
    print(f"Sample rate: {sample_rate}", flush=True)
    print(f"Audio shape: {waveform.shape}", flush=True)
    print(f"Duration: {waveform.shape[-1] / sample_rate:.2f} seconds", flush=True)

    start = int(0 * sample_rate)
    end = int((0 + 60) * sample_rate)
    waveform = waveform[:, start:end]

    if waveform.shape[-1] == 0 : raise ValueError("The selected test segment is empty.")
    waveform = waveform.mean(dim=0, keepdim=True)
    
    if sample_rate != 8000 :
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=8000)
        waveform = resampler(waveform)
        sample_rate = 8000
        
    print(f"Test clip: {waveform.shape[-1]} samples at {sample_rate} Hz ({waveform.shape[-1] / sample_rate:.2f} seconds)", flush=True,)
    
    temp_vocal = os.path.join(output_dir, "test_mixture.wav")
    torchaudio.save(temp_vocal, waveform, 8000)
    
    if is_silent(waveform) : 
        print("Segment is silent; skipping speaker separation.", flush=True)
        return
    
    pipeline = Pipeline.from_pretrained("pyannote/speech-separation-ami-1.0", use_auth_token=hf_token)
    pipeline.to(torch.device("cpu"))
    print("Separating overlapping speech...", flush=True)
    
    with ProgressHook() as hook : diarization, sources = pipeline({"waveform" : waveform, "sample_rate" : sample_rate}, hook=hook)
    overlap_timeline = diarization.get_overlap()
    overlap_regions = list(overlap_timeline)
    print(f"Detected speakers: {diarization.labels()}", flush=True)
    print(f"Overlapping regions: {len(overlap_regions)}", flush=True,)
    
    speaker_embeds = load_embedding_model(hf_token)
    embedding_dir = os.path.join(output_dir, "speaker_embeddings")
    os.makedirs(embedding_dir, exist_ok=True)
    
    for speaker in diarization.labels() :
        turns = [(segment.start, segment.end) for segment, _, label in diarization.itertracks(yield_label=True) if label == speaker]
        
        for turn_index, (turn_start, turn_end) in enumerate(turns) :
            if turn_end - turn_start < 2.0 : print(f"Skipping short turn for {speaker}: {turn_start:.2f}-{turn_end:.2f}s", flush=True)
            continue
        
        a = max(0, int(turn_start * sample_rate))
        b = min(waveform.shape[-1], int(turn_end * sample_rate))
        turn_audio = waveform[:, a:b]
        
        if turn_audio.shape[-1] == 0 or is_silent(turn_audio) : continue
        
        has_overlap = any(segment.start < turn_end and segment.end > turn_start for segment in overlap_regions)
        if has_overlap :
            print(f"Skipping overlapping turn: {speaker} {turn_start:.2f}-{turn_end:.2f}s", flush=True)
            continue
        
        save_embedding(turn_audio, sample_rate, f"{speaker}_turn{turn_index:03d}", embedding_dir, speaker_embeds)
    
    for index, speaker in enumerate(diarization.labels()) : 
        audio = torch.as_tensor(sources.data[:, index], dtype=torch.float32).unsqueeze(0)
        output_path = os.path.join(output_dir, f"{speaker}.wav")
        torchaudio.save(output_path, audio, 16000)
        print(f"Saved : {output_path}", flush=True) 
        
    rttm = os.path.join(output_dir, "diarization.rttm")
    with open(rttm, "w") as file: diarization.write_rttm(file)
    
    print(f"Saved speaker timings: {rttm}", flush=True)
    print("PixIT test complete.", flush=True)

def create_me_stem(input_path, output_me, device, hf_token) :
    if device == "mps" and torch.backends.mps.is_available() : device = torch.device("mps")
    elif device == "cuda" and torch.cuda.is_available() : device = torch.device("cuda")
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
    
    sources = {name : separated_stems[i] for i, name in enumerate(model.sources)}
    
    drums = sources["drums"]
    bass = sources["bass"]
    other = sources["other"]
    vocals = sources["vocals"]
        
    torchaudio.save(output_me, (drums + bass + other), sample_rate)
    separate_speakers(vocals, sample_rate, "speaker_vocal", hf_token)
    
if __name__ == "__main__" : 
    separate_speakers(vocal_waveform="vocals.wav", output_dir="speaker_chunks", hf_token="")
    #create_me_stem(input_path="KUTT10.wav", output_me="M_and_E.wav", device="cpu", hf_token="")