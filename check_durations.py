"""check_durations.py — do the manifest's durations match the rendered files?

The web app positions its playhead as t / duration, so a stale duration makes the
bar race ahead of the audio. Run from the music-rag directory.
"""
import json, subprocess, shutil
from pathlib import Path

SITE = Path("site")          # where render_audio.py wrote its output


def probe(path):
    """Actual duration in seconds, or None if ffprobe can't read the file."""
    if not path.exists():
        return None
    r = subprocess.run(["ffprobe", "-v", "error",
                        "-show_entries", "format=duration",
                        "-of", "csv=p=0", str(path)],
                       capture_output=True, text=True)
    try:
        return float(r.stdout.strip())
    except ValueError:
        print(f"    ffprobe said: {r.stderr.strip()[:120]}")
        return None


def local(url):
    """'audio/calib/01.mp3' -> site/calib/01.mp3"""
    return SITE / url.split("audio/", 1)[-1]


def main():
    if not shutil.which("ffprobe"):
        raise SystemExit("ffprobe not installed")
    src = Path("trials_rendered.json")
    if not src.exists():
        raise SystemExit(f"{src} not found — are you in the music-rag directory?")

    d = json.loads(src.read_text())
    print(f"{src}: {len(d['trials'])} trials, {len(d['calibration'])} calibration\n")

    rows, bad = [], 0
    for c in d["calibration"]:
        rows.append(("calib", c["url"], c["duration"], c.get("boundary")))
    for t in d["trials"]:
        clips = t.get("clips") or {k: t[k] for k in ("clip_a", "clip_b") if k in t}
        for system, c in clips.items():
            rows.append((system, c["url"], c["duration"], c.get("boundary")))

    for system, url, claimed, boundary in rows:
        p = local(url)
        actual = probe(p)
        if actual is None:
            print(f"  MISSING  {system:8s} {url}  (looked in {p})")
            bad += 1
            continue
        delta = abs(actual - claimed)
        flag = "  <-- MISMATCH" if delta > 0.25 else ""
        if flag:
            bad += 1
        print(f"  {system:8s} {url:26s} manifest {claimed:6.2f}s  "
              f"file {actual:6.2f}s  boundary {boundary}{flag}")

    print(f"\n{len(rows)} clips checked, {bad} problem(s)")
    if bad:
        print("Stale manifest: re-run render_audio.py, then manifest.py, "
              "then re-paste sessions.json into index.html.")


if __name__ == "__main__":
    main()