"""render_audio.py — selected generations -> level-matched MP3s + a blinded manifest.

Needs, in WSL:
    sudo apt install fluidsynth fluid-soundfont-gm ffmpeg
"""
import json, hashlib, subprocess, shutil
from pathlib import Path

import pretty_midi
from features import load_notes, window, piece_name
from seeds import load_seed
from presentation import assemble, write, instrument_for

SF2      = "/usr/share/sounds/sf2/FluidR3_GM.sf2"
OUT      = Path("site")                 # becomes the repo's audio/ folder
SALT     = "change-me-before-rendering" # keeps opaque names unguessable
SYSTEMS = ("A", "Aprime", "B")          
LUFS     = -20                          # every clip normalised to this loudness


def sh(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(" ".join(cmd[:3]) + "\n" + r.stderr[-600:])


def duration(path):
    r = subprocess.run(["ffprobe","-v","error","-show_entries","format=duration",
                        "-of","csv=p=0",str(path)], capture_output=True, text=True)
    return round(float(r.stdout.strip()), 3)


def render(midi_path, mp3_path):
    wav = midi_path.with_suffix(".wav")
    sh(["fluidsynth","-ni","-F",str(wav),"-r","44100","-g","0.7",
        "-R","0","-C","0",SF2,str(midi_path)])
    # loudnorm is the important part: unmatched levels between the two clips in a
    # pair would be judged instead of the music. Fades only guard encoder pops.
    sh(["ffmpeg","-y","-loglevel","error","-i",str(wav),
        "-af","dynaudnorm=f=250:g=15,afade=t=in:st=0:d=0.03,areverse,"
              "afade=t=in:st=0:d=0.12,areverse",
        "-codec:a","libmp3lame","-b:a","128k",str(mp3_path)])
    wav.unlink()


def opaque(*parts):
    return hashlib.sha1("|".join(map(str, parts)).encode() + SALT.encode()).hexdigest()[:10]

VAL = {"a_letter","chorale_and_procession","concerto","el_mar",
       "elegy","forest_overture","spanish_sunrise"}


def excerpt(path, seconds=10.0):
    """First `seconds` of actual playing, not of the timeline. These are extracted
    orchestral parts -- a flute may rest for 30s before entering, and windowing from
    zero then yields an empty clip."""
    notes = load_notes(path)
    if not notes:
        return []
    t0 = notes[0].start
    return window([pretty_midi.Note(n.velocity, n.pitch, n.start - t0, n.end - t0)
                   for n in notes], seconds)

def main():
    if not Path(SF2).exists():
        raise SystemExit(f"soundfont missing: {SF2}")
    for tool in ("fluidsynth","ffmpeg","ffprobe"):
        if not shutil.which(tool):
            raise SystemExit(f"{tool} not installed")

    selected = json.loads(Path("selected.json").read_text())
    seed_list = [s["seed"] for s in selected]
    tmp = Path("_render"); tmp.mkdir(exist_ok=True)
    (OUT/"c").mkdir(parents=True, exist_ok=True)
    (OUT/"calib").mkdir(parents=True, exist_ok=True)

    trials, key = [], []
    for rec in selected:
        seed = rec["seed"]
        seed_notes = load_seed(seed)

        # Assemble BOTH systems before choosing a timbre: the instrument must be a
        # property of the seed, not of one system's output, or the pair stops being
        # comparable and people judge the timbre instead of the melody.
        built = {}
        for system in SYSTEMS:
            from features import window
            cont = window(load_notes(Path(rec["paths"][system])), 10.5)
            built[system] = assemble(seed_notes, cont)      # (notes, boundary)

        combined = [n for notes, _ in built.values() for n in notes]
        program, iname = instrument_for(seed, seed_list, combined)

        clips = {}
        for system in SYSTEMS:
            notes, boundary = built[system]
            stem = opaque(seed, system)
            mid = tmp/f"{stem}.mid"
            write(notes, mid, program=program)
            name = stem + ".mp3"
            render(mid, OUT/"c"/name)
            dur = duration(OUT/"c"/name)
            clips[system] = {"url": f"audio/c/{name}", "boundary": round(boundary,3),
                             "duration": dur}
            key.append({"file": name, "seed": seed, "system": system,
                        "sample": rec["sample"], "instrument": iname})
            print(f"  {seed[:38]:38s} {system:7s} {iname:9s} {dur:5.1f}s -> {name}")

        # No pairing here: rendering produces clips, the manifest decides comparisons.
        trials.append({"seed": seed, "instrument": iname, "clips": clips})

        # Choose by content, not by position in a sorted list.
    cands = []
    for p in sorted(Path("corpus").glob("*.mid")):
        if piece_name(p) in VAL:
            continue
        e = excerpt(p, 10.0)
        if len(e) >= 12 and max(n.end for n in e) >= 7.0:
            cands.append((p, e))

    picks = cands[:: max(1, len(cands)//3)][:3]
    assert len(picks) == 3, f"only {len(picks)} viable calibration candidates"

    calib = []
    for i, (p, notes) in enumerate(picks):
        mid = tmp/f"calib{i}.mid"
        write(notes, mid, program=[73, 68, 71][i])
        name = f"{i+1:02d}.mp3"
        render(mid, OUT/"calib"/name)
        d = duration(OUT/"calib"/name)
        assert d >= 5.0, f"calibration {name} is only {d:.1f}s -- source was near-empty"
        calib.append({"url": f"audio/calib/{name}", "boundary": 0, "duration": d})
        print(f"  calibration {i+1}: {p.name[:44]:44s} {len(notes):3d} notes  {d:.1f}s")

        Path("trials_rendered.json").write_text(json.dumps(
            {"calibration": calib, "trials": trials}, indent=2))
        Path("ANSWER_KEY.json").write_text(json.dumps(key, indent=2))
        shutil.rmtree(tmp)
        print(f"\n{len(trials)} trials, {len(key)} clips -> {OUT}/")
        print("ANSWER_KEY.json is PRIVATE — never commit it to the public repo.")


if __name__ == "__main__":
    main()