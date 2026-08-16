#!/usr/bin/env python3
# Generates 3 seamless looping background-music beds for 墨色挑战 (Stroop H5).
# Every note envelope returns to zero at bar boundaries, so loop = buffer end -> start is silent->silent (seamless).
import numpy as np, soundfile as sf

SR = 44100

def hpf(x, fc=20.0):
    a = 1.0 / (1.0 + (2*np.pi*fc/SR))
    y = np.zeros_like(x); y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = a*(y[i-1] + x[i] - x[i-1])
    return y

def note(freq, dur, wtype='sine', amp=0.2, a=0.01, d=0.05, s=0.0, r=0.08, sr=SR):
    n = max(1, int(dur*sr))
    t = np.arange(n)/sr
    env = np.ones(n)
    na, nd, nr = int(a*sr), int(d*sr), int(r*sr)
    ns = max(0, n - na - nd - nr)
    env[:na] = np.linspace(0, 1, na) if na else 0
    if nd:
        env[na:na+nd] = np.linspace(1, s, nd)
    if ns:
        env[na+nd:na+nd+ns] = s
    if nr:
        env[-nr:] = np.linspace(s, 0, nr)
    if wtype == 'sine':
        w = np.sin(2*np.pi*freq*t)
    elif wtype == 'triangle':
        w = 2*np.abs(2*(t*freq - np.floor(t*freq + 0.5))) - 1
    elif wtype == 'square':
        w = np.sign(np.sin(2*np.pi*freq*t)) * 0.6 + 0.4*np.sin(2*np.pi*freq*t)
    else:  # saw
        w = 2*(t*freq - np.floor(t*freq + 0.5))
    return w * env * amp

def mix(events, dur):
    buf = np.zeros(int(dur*SR))
    for (t0, n) in events:
        k = int(t0*SR)
        m = min(len(n), len(buf)-k)
        if m > 0:
            buf[k:k+m] += n[:m]
    return buf

def norm(x, peak=0.6):
    x = hpf(x)
    m = np.max(np.abs(x))
    if m > 0:
        x = x/m*peak
    return np.float32(np.clip(x, -1, 1))

def write(path, x):
    x = norm(x)
    # duplicate to stereo with tiny L/R detune for width
    l = x
    r = np.interp(np.arange(len(x)) + 1.5, np.arange(len(x)), x)  # 1.5-sample delay on right
    sf.write(path, np.stack([l, r], axis=1), SR)
    print(f"  wrote {path}  ({len(x)/SR:.2f}s, peak {np.max(np.abs(x)):.3f})")

# ---------------- chill: slow Cmaj7 - Gmaj7 - Amin7 - Fmaj7 pad, 12s ----------------
def gen_chill():
    chords = [
        [261.63, 329.63, 392.00, 493.88],   # Cmaj7
        [196.00, 246.94, 293.66, 369.99],   # Gmaj7
        [220.00, 261.63, 329.63, 392.00],   # Amin7
        [174.61, 220.00, 261.63, 329.63],   # Fmaj7
    ]
    bar = 3.0
    ev = []
    for i, ch in enumerate(chords):
        t0 = i*bar
        for f in ch:
            ev.append((t0, note(f, bar, 'sine', amp=0.16, a=0.5, d=0.3, s=0.75, r=0.6)))
        # soft high bell at bar start and midpoint
        ev.append((t0, note(ch[0]*2, 0.5, 'sine', amp=0.06, a=0.02, d=0.4, r=0.1)))
        ev.append((t0+1.5, note(ch[2]*2, 0.5, 'sine', amp=0.05, a=0.02, d=0.4, r=0.1)))
    return mix(ev, bar*4)

# ---------------- arcade: bouncy chiptune, 132 BPM, 8 beats loop ----------------
def gen_arcade():
    beat = 60.0/132.0
    roots = [261.63, 392.00, 220.00, 349.23]  # C G A F (one per 2 beats)
    ev = []
    for b in range(8):
        t0 = b*beat
        ch = roots[b//2]
        # bass square on the beat
        ev.append((t0, note(ch/2, beat*0.9, 'square', amp=0.16, a=0.005, d=0.18, r=0.05)))
        # 8th-note arp (triangle) cycling root/third/fifth/octave
        arp = [ch, ch*1.26, ch*1.5, ch*2.0]
        for k in range(2):
            tt = t0 + k*beat/2
            f = arp[(b*2+k) % 4]
            ev.append((tt, note(f, beat*0.42, 'triangle', amp=0.11, a=0.004, d=0.12, r=0.05)))
    return mix(ev, beat*8)

# ---------------- pulse: focus-friendly 4-on-floor + ostinato, 100 BPM, 8 beats ----------------
def gen_pulse():
    beat = 60.0/100.0
    ev = []
    # soft kick on every beat (4-on-the-floor)
    for b in range(8):
        ev.append((b*beat, note(60, 0.26, 'sine', amp=0.26, a=0.004, d=0.22, r=0.02)))
    # two-note ostinato on offbeats
    ost = [392.00, 523.25]
    for b in range(8):
        tt = b*beat + beat/2
        ev.append((tt, note(ost[b % 2], beat*0.42, 'triangle', amp=0.10, a=0.01, d=0.2, r=0.06)))
    return mix(ev, beat*8)

if __name__ == '__main__':
    print("Generating BGM beds...")
    write('bgm/chill.wav', gen_chill())
    write('bgm/arcade.wav', gen_arcade())
    write('bgm/pulse.wav', gen_pulse())
    print("Done.")
