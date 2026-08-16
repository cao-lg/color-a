#!/usr/bin/env python3
# Generates looping background-music beds for 墨色挑战 (Stroop H5).
# Focus bed (gen_focus) follows science-backed concentration principles:
#   instrumental only, ~60-80 BPM low arousal, predictable/repeating, low dynamic
#   range, ambient pad + sparse motif + brown-noise masker + optional binaural alpha
#   (8-12 Hz; needs headphones; evidence is mixed -- see design notes).
# Every note envelope returns to zero at bar boundaries, so loop = end -> start is
# silent -> silent (seamless). The focus bed also gets an explicit seam crossfade.
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

def norm_mono(x, peak=0.6):
    x = hpf(x)
    m = np.max(np.abs(x))
    if m > 0:
        x = x/m*peak
    return np.float32(np.clip(x, -1, 1))

def write(path, x):
    xn = norm_mono(x)
    sf.write(path, np.stack([xn, xn], axis=1), SR)   # duplicate to stereo with slight width
    print(f"  wrote {path}  ({len(xn)/SR:.2f}s, peak {np.max(np.abs(xn)):.3f})")

def write_stereo(path, L, R):
    Ln = norm_mono(L); Rn = norm_mono(R)
    sf.write(path, np.stack([Ln, Rn], axis=1), SR)
    print(f"  wrote {path}  ({len(Ln)/SR:.2f}s, peak {np.max(np.abs(Ln)):.3f})")

def pinkish_noise(N):
    # brown-noise (low-rumble) masker: integrated white noise, -6dB/octave tilt
    out = np.zeros(N); last = 0.0
    for i in range(N):
        white = np.random.randn()
        last = (last + 0.02*white) / 1.02
        out[i] = last
    return out

# ---------------- chill: slow Cmaj7 - Gmaj7 - Amin7 - Fmaj7 pad, 12s ----------------
def gen_chill():
    chords = [
        [261.63, 329.63, 392.00, 493.88],
        [196.00, 246.94, 293.66, 369.99],
        [220.00, 261.63, 329.63, 392.00],
        [174.61, 220.00, 261.63, 329.63],
    ]
    bar = 3.0
    ev = []
    for i, ch in enumerate(chords):
        t0 = i*bar
        for f in ch:
            ev.append((t0, note(f, bar, 'sine', amp=0.16, a=0.5, d=0.3, s=0.75, r=0.6)))
        ev.append((t0, note(ch[0]*2, 0.5, 'sine', amp=0.06, a=0.02, d=0.4, r=0.1)))
        ev.append((t0+1.5, note(ch[2]*2, 0.5, 'sine', amp=0.05, a=0.02, d=0.4, r=0.1)))
    return mix(ev, bar*4)

# ---------------- arcade: bouncy chiptune, 132 BPM, 8 beats loop ----------------
def gen_arcade():
    beat = 60.0/132.0
    roots = [261.63, 392.00, 220.00, 349.23]
    ev = []
    for b in range(8):
        t0 = b*beat
        ch = roots[b//2]
        ev.append((t0, note(ch/2, beat*0.9, 'square', amp=0.16, a=0.005, d=0.18, r=0.05)))
        arp = [ch, ch*1.26, ch*1.5, ch*2.0]
        for k in range(2):
            tt = t0 + k*beat/2
            f = arp[(b*2+k) % 4]
            ev.append((tt, note(f, beat*0.42, 'triangle', amp=0.11, a=0.004, d=0.12, r=0.05)))
    return mix(ev, beat*8)

# ---------------- pulse: light groove, 100 BPM, 8 beats (keep as 'groove' option) ----------------
def gen_pulse():
    beat = 60.0/100.0
    ev = []
    for b in range(8):
        ev.append((b*beat, note(60, 0.26, 'sine', amp=0.26, a=0.004, d=0.22, r=0.02)))
    ost = [392.00, 523.25]
    for b in range(8):
        tt = b*beat + beat/2
        ev.append((tt, note(ost[b % 2], beat*0.42, 'triangle', amp=0.10, a=0.01, d=0.2, r=0.06)))
    return mix(ev, beat*8)

# ---------------- focus: science-backed concentration bed ----------------
# ambient pad (low info density) + sparse motif (long silences) + brown-noise
# masker + binaural alpha beat (carrier 180Hz, 10Hz difference => relaxed focus;
# requires headphones; efficacy is scientifically mixed -- not a magic bullet).
def gen_focus():
    chords = [
        [130.81, 196.00, 261.63, 329.63],  # Cadd9
        [98.00, 146.83, 196.00, 246.94],   # Gsus2
        [110.00, 164.81, 220.00, 261.63],  # Am7
        [87.31, 130.81, 174.61, 261.63],   # Fmaj7
    ]
    bar = 4.0
    ev = []
    for i, ch in enumerate(chords):
        t0 = i*bar
        for f in ch:
            ev.append((t0, note(f, bar, 'sine', amp=0.12, a=1.4, d=0.4, s=0.7, r=1.4)))
        if i in (0, 2):  # sparse bell, long silence (C418 / Minecraft style)
            ev.append((t0+1.2, note(ch[2]*2, 1.2, 'sine', amp=0.05, a=0.05, d=0.8, r=0.3)))
    dur = bar*4
    x = mix(ev, dur)
    N = len(x)
    bn = pinkish_noise(N); bn = bn/np.max(np.abs(bn))*0.04   # very low brown-noise masker
    x = x + bn[:N]
    t = np.arange(N)/SR
    bL = 0.025*np.sin(2*np.pi*180*t)   # binaural alpha: L 180Hz
    bR = 0.025*np.sin(2*np.pi*190*t)   #                 R 190Hz -> 10Hz beat
    L = x + bL; R = x + bR
    k = int(0.03*SR)                    # equal-power seam crossfade => click-free loop
    a = np.sqrt(np.linspace(1, 0, k)); b = np.sqrt(np.linspace(0, 1, k))
    L[:k] = L[:k]*a + L[-k:]*b
    R[:k] = R[:k]*a + R[-k:]*b
    return L, R

if __name__ == '__main__':
    print("Generating BGM beds...")
    write('bgm/chill.wav', gen_chill())
    write('bgm/arcade.wav', gen_arcade())
    write('bgm/pulse.wav', gen_pulse())
    write_stereo('bgm/focus.wav', *gen_focus())
    print("Done.")
