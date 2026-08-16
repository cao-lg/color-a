#!/usr/bin/env python3
"""Generate high-impact game SFX as mastered .wav files (zero external assets).
Run: python3 gen_sfx.py  ->  writes sfx/*.wav
Impact recipe per sound: harmonic pluck + sub transient + sparkle, then
soft-clip (grit) + convolution reverb (space) + normalize to -2.5 dBFS.
"""
import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve

SR = 44100

def softclip(x, g=1.15):
    return np.tanh(x * g)

def make_ir(dur=0.28, decay=3.2):
    n = int(dur * SR)
    ir = np.random.randn(n) * np.power(np.linspace(1, 1e-4, n), decay)
    return ir / np.max(np.abs(ir))

def reverb(x, ir, wet=0.28):
    wet_sig = fftconvolve(x, ir, mode='full')[:len(x)]
    return x * (1 - wet) + wet_sig * wet

def normalize(x, peak_db=-2.5):
    peak = 10 ** (peak_db / 20.0)
    m = np.max(np.abs(x))
    if m < 1e-6:
        return x.astype(np.float32)
    return (x / m * peak).astype(np.float32)

def pluck(freq, dur, decay=0.15, harm_ratios=(1, 2, 3), harm_amps=(1, 0.5, 0.25)):
    n = int(dur * SR)
    t = np.arange(n) / SR
    sig = np.zeros(n)
    for r, a in zip(harm_ratios, harm_amps):
        sig += a * np.sin(2 * np.pi * freq * r * t * (1 - 0.02 * (t / dur)))  # slight downward glide
    a = int(0.0015 * SR)
    env = np.exp(-t / decay)
    if a > 0:
        env[:a] = np.linspace(0, 1, a)
    return sig * env

def noise_burst(dur, hp=5000, amp=1.0, decay=None):
    n = int(dur * SR)
    sig = (np.random.randn(n) * 2 - 1)
    if decay:
        t = np.arange(n) / SR
        sig = sig * np.exp(-t / decay)
    return sig * amp

def save(name, sig):
    sig = normalize(sig - np.mean(sig), -2.5)
    sf.write('sfx/%s.wav' % name, sig, SR)
    print('%-8s len=%.2fs  peak=%.3f  rms=%.4f' % (name, len(sig) / SR, np.max(np.abs(sig)), np.sqrt(np.mean(sig ** 2))))

# ---------- correct: pleasant major-chord "ding" (marimba/bell), soft attack, warm, musical ----------
root = 523.25  # C5
chord = [(root, 1.0), (root * 1.25, 0.70), (root * 1.5, 0.55), (root * 2.0, 0.22)]  # C5 E5 G5 C6
n = int(0.42 * SR)
t = np.arange(n) / SR
body = np.zeros(n)
for f, a in chord:
    sig = np.sin(2 * np.pi * f * t)
    sig += 0.12 * np.sin(2 * np.pi * f * 2.76 * t)  # bell inharmonic shimmer (glockenspiel-like)
    body += a * sig * np.exp(-t / 0.30)
atk = int(0.006 * SR)
body[:atk] = body[:atk] * np.linspace(0, 1, atk)
warm = pluck(130, 0.12, decay=0.08, harm_ratios=(1,), harm_amps=(1,)) * 0.16  # soft low warmth, no boom
warm = np.pad(warm, (0, n - len(warm)))
shim = np.sin(2 * np.pi * 2600 * t + np.pi / 3) * np.exp(-t / 0.05) * 0.05  # gentle high shimmer
correct = softclip((body + warm + shim) * 1.12)
correct = reverb(correct, make_ir(0.35, 3.2), 0.30)

# ---------- wrong: detuned descending tones + low noise thud ----------
n = int(0.36 * SR)
t = np.arange(n) / SR
env = np.exp(-t / 0.18)
env[:int(0.002 * SR)] = np.linspace(0, 1, int(0.002 * SR))
o1 = np.sin(2 * np.pi * (140 * np.exp(-t / 0.12)) * t) * 0.6
o2 = np.sin(2 * np.pi * (93 * np.exp(-t / 0.12)) * t) * 0.5
sq = np.zeros(n)
for k in (1, 3, 5):
    sq += np.sin(2 * np.pi * (110 * np.exp(-t / 0.12)) * t * k) / k
sq *= 0.15
lownoise = noise_burst(0.10, hp=200, amp=0.5, decay=0.04)
lownoise = np.pad(lownoise, (0, n - len(lownoise)))
wrong = softclip(((o1 + o2 + sq) * env + lownoise) * 1.4)
wrong = reverb(wrong, make_ir(0.30, 2.6), 0.18)

# ---------- level: ascending arpeggio C-E-G-C ----------
parts = []
for i, f in enumerate([523.25, 659.25, 783.99, 1046.5]):
    p = pluck(f, 0.14, decay=0.12, harm_ratios=(1, 2), harm_amps=(1, 0.4))
    parts.append(np.concatenate([np.zeros(int(i * 0.085 * SR)), p]))
total = max(len(p) for p in parts)
parts = [np.pad(p, (0, total - len(p))) for p in parts]
level = reverb(softclip(sum(parts) * 1.1), make_ir(0.30, 3.0), 0.30)

# ---------- pb: victory fanfare + sparkle tail ----------
parts = []
for i, f in enumerate([523.25, 659.25, 783.99, 1046.5, 1318.5]):
    p = pluck(f, 0.30, decay=0.22, harm_ratios=(1, 2, 3), harm_amps=(1, 0.4, 0.2))
    parts.append(np.concatenate([np.zeros(int(i * 0.10 * SR)), p]))
total = max(len(p) for p in parts)
parts = [np.pad(p, (0, total - len(p))) for p in parts]
spark = noise_burst(0.15, hp=6000, amp=0.25, decay=0.06)
spark = np.pad(spark, (int(0.5 * SR), 0))
spark = np.pad(spark, (0, total - len(spark)))
pb = reverb(softclip(sum(parts) * 1.1 + spark), make_ir(0.40, 3.0), 0.32)

# ---------- tick: short bright click (pitch shifted in-game for tension) ----------
n = int(0.05 * SR)
t = np.arange(n) / SR
tick = np.sin(2 * np.pi * 1500 * t) * np.exp(-t / 0.012) * 0.8
nb = noise_burst(0.01, hp=4000, amp=0.3, decay=0.003) * 0.5
tick = tick + np.pad(nb, (0, n - len(nb)))
tick = softclip(tick * 1.2)

if __name__ == '__main__':
    save('correct', correct)
    save('wrong', wrong)
    save('level', level)
    save('pb', pb)
    save('tick', tick)
    print('done -> sfx/')
