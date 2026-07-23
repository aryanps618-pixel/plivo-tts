"""Row-aware, split-dimension perturbation search for Kokoro voice cloning.

Verified against kokoro==0.9.4 source (model.py / pipeline.py):
  * style vector splits at dim 128:
      ref_s[:, :128]  -> decoder (timbre / acoustic identity)
      ref_s[:, 128:]  -> prosody predictor (duration, F0, energy)
  * a synth call indexes exactly ONE row: pack[len(phonemes_string) - 1]
    out of the (510, 1, 256) tensor. Perturbing all 510 rows uniformly
    (the skeleton's randn_like) wastes budget on rows your fitness
    sentences never touch, AND held-out grading sentences of a different
    phoneme length will land on rows you never optimized at all. This is
    the "evaporates on unseen sentences" trap the handout warns about.

Strategy:
  1. Only perturb rows actually exercised by SENTENCES (found for free from
     r.phonemes during synthesis), smoothed into a small neighborhood so
     nearby-length unseen sentences still benefit.
  2. Separate step sizes for the acoustic half (:128, drives similarity)
     and prosodic half (128:, drives intelligibility risk) — prosodic
     moves much more conservatively.
  3. Fitness = mean speaker-similarity across several sentences of
     different lengths, minus cheap DSP naturalness penalties (no ASR,
     no extra pretrained model -> stays inside the Resemblyzer-only rule):
       - spectral flatness (raspy/noise artifacts the embedding loves)
       - silence ratio (degenerate near-silent audio)
       - duration sanity (collapsed or runaway output)
       - cross-sentence self-consistency (same "identity" on both
         sentences, not a fluke on one)
  4. Adaptive step size (1/5-success rule) + fitness cache.

    python search.py --reference_dir ../reference --start blend_baseline.pt \
        --iters 300 --out voice.pt
"""
import argparse
import hashlib

import numpy as np
import torch
import soundfile as sf
import librosa

import synth
import similarity as sim

# Sentences spanning short/medium/long phoneme counts, pulled from the
# reference transcripts + the brief's own example, so optimized rows cover
# a spread rather than a single point. Held-out sentences are unseen text
# but presumably similar length/register to these.
SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "I will call you back tomorrow at three thirty.",
    "Please read out the last four digits of the reference number on your screen.",
    "The delivery should arrive between nine thirty and eleven on Saturday morning.",
    "Are you sure the cable is plugged in properly? Sometimes the connector comes loose.",
]

ACOUSTIC = slice(0, 128)
PROSODIC = slice(128, 256)


def synth_with_rows(text, voice):
    """Like synth.synthesize but also returns which tensor rows were used."""
    pipe = synth.get_pipeline()
    v = synth.load_voice(voice)
    rows, chunks = [], []
    for r in pipe(text, voice=v, speed=1.0):
        rows.append(len(r.phonemes) - 1)
        chunks.append(r.audio)
    audio = torch.cat([c if isinstance(c, torch.Tensor) else torch.tensor(c)
                        for c in chunks])
    return audio.detach().cpu().numpy().astype(np.float32), rows


def used_rows(sentences, voice, neighborhood=2):
    """All rows exercised by `sentences`, expanded with a small neighborhood
    so nearby unseen sentence lengths still land on optimized rows."""
    rows = set()
    for t in sentences:
        _, rs = synth_with_rows(t, voice)
        for r in rs:
            for d in range(-neighborhood, neighborhood + 1):
                rr = r + d
                if 0 <= rr < voice.shape[0]:
                    rows.add(rr)
    return sorted(rows)


def naturalness_penalty(wav, sr=synth.SR):
    """Cheap DSP sanity checks — no pretrained model, no ASR."""
    if wav.size == 0 or not np.isfinite(wav).all():
        return 10.0  # hard reject
    rms = np.sqrt(np.mean(wav ** 2))
    if rms < 1e-4:
        return 10.0  # near-silent / degenerate

    penalty = 0.0

    # 1) spectral flatness: high flatness ~ noisy/raspy artifacts that
    #    Resemblyzer can be fooled by but ears hate.
    flat = librosa.feature.spectral_flatness(y=wav).mean()
    if flat > 0.35:
        penalty += (flat - 0.35) * 4.0

    # 2) silence ratio within the utterance (dropped audio, glitches)
    frame_rms = librosa.feature.rms(y=wav, frame_length=1024, hop_length=256)[0]
    silent_ratio = float(np.mean(frame_rms < rms * 0.05))
    if silent_ratio > 0.35:
        penalty += (silent_ratio - 0.35) * 2.0

    # 3) duration sanity: kokoro at speed=1.0 is roughly 12-20 phones/sec
    #    of audio for natural speech; wildly short/long output = collapse.
    dur = len(wav) / sr
    if dur < 0.3:
        penalty += 5.0

    return penalty


def fitness(voice, target_emb, texts, cache):
    key = hashlib.sha1(voice.numpy().tobytes()).hexdigest()
    if key in cache:
        return cache[key]

    sims, embs, pens = [], [], []
    for t in texts:
        wav = synth.synthesize(t, voice)
        pens.append(naturalness_penalty(wav))
        emb = sim.embed_wav_array(wav)
        embs.append(emb)
        sims.append(sim.cosine(emb, target_emb))

    mean_sim = float(np.mean(sims))

    # cross-sentence self-consistency: candidate should sound like the SAME
    # person across different sentences, not just get lucky on one.
    consistency = np.mean([
        sim.cosine(embs[i], embs[j])
        for i in range(len(embs)) for j in range(i + 1, len(embs))
    ]) if len(embs) > 1 else 1.0

    score = mean_sim + 0.15 * consistency - 0.1 * float(np.mean(pens))
    cache[key] = score
    return score


def perturb(base, rows, step_acoustic, step_prosodic):
    cand = base.clone()
    for r in rows:
        cand[r, :, ACOUSTIC] += step_acoustic * torch.randn(128)
        cand[r, :, PROSODIC] += step_prosodic * torch.randn(128)
    return cand


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference_dir", required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--step", type=float, default=0.04,
                     help="initial acoustic-half step; prosodic is 0.3x this")
    ap.add_argument("--out", default="voice.pt")
    ap.add_argument("--listen_every", type=int, default=10)
    args = ap.parse_args()

    target = sim.target_embedding(args.reference_dir)
    best = synth.load_voice(args.start).clone()
    cache = {}

    rows = used_rows(SENTENCES, best)
    print(f"optimizing {len(rows)} / {best.shape[0]} rows: {rows}")

    best_f = fitness(best, target, SENTENCES, cache)
    print(f"start fitness: {best_f:.4f}")

    step_a = args.step
    step_p = args.step * 0.3
    window, successes = [], 0
    accepted = 0

    for i in range(1, args.iters + 1):
        cand = perturb(best, rows, step_a, step_p)
        f = fitness(cand, target, SENTENCES, cache)

        window.append(f > best_f)
        if len(window) > 20:
            window.pop(0)

        if f > best_f:
            best, best_f, accepted = cand, f, accepted + 1
            torch.save(best, args.out)
            print(f"iter {i:4d}  accepted #{accepted}  fitness {best_f:.4f}  "
                  f"step(a={step_a:.4f} p={step_p:.4f})")
            if accepted % args.listen_every == 0:
                for t in SENTENCES[:1]:
                    wav = synth.synthesize(t, best)
                    sf.write(f"listen_{accepted}.wav", wav, synth.SR)
                print(f"  -> wrote listen_{accepted}.wav — GO LISTEN")

        # 1/5-success-rule step adaptation, checked every 20 iters
        if len(window) == 20 and i % 20 == 0:
            rate = sum(window) / 20
            if rate > 0.2:
                step_a, step_p = step_a * 1.2, step_p * 1.2
            elif rate < 0.2:
                step_a, step_p = step_a * 0.85, step_p * 0.85

    torch.save(best, args.out)
    wav = synth.synthesize(SENTENCES[0], best)
    sf.write("listen_final.wav", wav, synth.SR)
    print(f"final fitness {best_f:.4f} -> saved {args.out}")
    print("wrote listen_final.wav — LISTEN BEFORE YOU SUBMIT")


if __name__ == "__main__":
    main()
