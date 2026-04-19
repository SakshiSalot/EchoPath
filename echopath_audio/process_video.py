import cv2
import easyocr
import os
import subprocess
import tempfile
import shutil
from collections import defaultdict

from label_parser import parse_label, build_alert_phrase, PROXIMITY_PRIORITY

# ─────────────────────────────────────────────
# CONFIG — only edit these lines
# ─────────────────────────────────────────────
INPUT_VIDEO     = "echopath_Persons.mp4"        # ← your input video filename
OUTPUT_VIDEO    = "echopath_with_audio_pt1.mp4" # ← output video with audio
COOLDOWN_SEC    = 2.5   # min seconds between same alert repeating
PROCESS_EVERY_N = 4     # OCR every Nth frame (lower = more accurate, slower)
MIN_OCR_CONF    = 0.35  # minimum OCR confidence (lower if labels not detected)
# ─────────────────────────────────────────────

def extract_alerts(reader, cap, fps, total_frames):
    """
    Pass 1: Scan every Nth frame, OCR bounding box labels,
    parse and return a timestamped alert log.
    """
    alert_log   = []
    last_spoken = defaultdict(float)
    frame_idx   = 0

    print(f"[Pass 1] Scanning {total_frames} frames for MiDaS labels...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        timestamp  = frame_idx / fps

        if frame_idx % PROCESS_EVERY_N != 0:
            continue

        if frame_idx % 100 == 0:
            pct = 100 * frame_idx / total_frames
            print(f"  [{pct:.0f}%] Frame {frame_idx}/{total_frames}  ({timestamp:.1f}s)")

        ocr_results = reader.readtext(frame)

        frame_alerts = []
        for (_, text, conf) in ocr_results:
            if conf < MIN_OCR_CONF:
                continue
            if "|" not in text:
                continue
            if "d" not in text.lower() or "=" not in text:
                continue

            parsed = parse_label(text)
            if parsed is None:
                continue

            phrase   = build_alert_phrase(parsed)
            priority = PROXIMITY_PRIORITY.get(parsed["proximity"], 0)
            d_val    = parsed["d_value"] or 0
            frame_alerts.append((priority, d_val, phrase))

        if not frame_alerts:
            continue

        # Highest priority first, then highest d_value as tiebreaker
        frame_alerts.sort(key=lambda x: (x[0], x[1]), reverse=True)

        # Apply cooldown — log only top alert per frame
        for priority, d_val, phrase in frame_alerts:
            last_t = last_spoken[phrase]
            if (timestamp - last_t) >= COOLDOWN_SEC:
                alert_log.append((timestamp, phrase, priority))
                last_spoken[phrase] = timestamp
                print(f"  ✔ [{timestamp:.1f}s] d={d_val:.2f}  '{phrase}'")
                break

    print(f"[Pass 1] Done. {len(alert_log)} alerts found.")
    return alert_log


def generate_audio_clips(alert_log, tmp_dir):
    """
    Pass 2: Generate a WAV file for each unique alert phrase using pyttsx3.
    Returns list of (timestamp_sec, wav_path).
    """
    import pyttsx3

    print("[Pass 2] Generating TTS audio clips...")

    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.setProperty('volume', 1.0)

    for voice in engine.getProperty('voices'):
        if 'english' in voice.name.lower() or 'en' in voice.id.lower():
            engine.setProperty('voice', voice.id)
            break

    clip_list    = []
    seen_phrases = {}

    for i, (ts, phrase, priority) in enumerate(alert_log):
        if phrase in seen_phrases:
            clip_list.append((ts, seen_phrases[phrase]))
            continue

        wav_path = os.path.join(tmp_dir, f"clip_{i:04d}.wav")
        engine.save_to_file(phrase, wav_path)
        engine.runAndWait()

        seen_phrases[phrase] = wav_path
        clip_list.append((ts, wav_path))
        print(f"  Saved: [{ts:.1f}s] '{phrase}'")

    print(f"[Pass 2] Done. {len(seen_phrases)} unique clips generated.")
    return clip_list


def merge_audio_video(clip_list, duration_sec, tmp_dir):
    """
    Pass 3: Use ffmpeg to mix all TTS clips at correct timestamps
    and merge with the original video.
    """
    print("[Pass 3] Mixing audio and merging with video...")

    # Silent base audio track matching video duration
    silent_wav = os.path.join(tmp_dir, "silent.wav")
    subprocess.run([
        "ffmpeg", "-y", "-f", "lavfi",
        "-i", "anullsrc=r=22050:cl=mono",
        "-t", str(duration_sec),
        silent_wav
    ], check=True, capture_output=True)

    # Build ffmpeg filter_complex
    inputs  = ["-i", INPUT_VIDEO, "-i", silent_wav]
    filters = []
    labels  = ["[1:a]"]

    for i, (ts, wav_path) in enumerate(clip_list):
        inputs   += ["-i", wav_path]
        input_idx = i + 2
        delay_ms  = int(ts * 1000)
        filters.append(
            f"[{input_idx}:a]adelay={delay_ms}|{delay_ms}[a{i}]"
        )
        labels.append(f"[a{i}]")

    n = len(labels)
    filters.append(
        "".join(labels) +
        f"amix=inputs={n}:duration=first:dropout_transition=0[aout]"
    )

    filter_complex = ";".join(filters)
    out_tmp        = os.path.join(tmp_dir, "merged_tmp.mp4")

    cmd = [
        "ffmpeg", "-y",
        *inputs,
        "-filter_complex", filter_complex,
        "-map", "0:v",
        "-map", "[aout]",
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        out_tmp
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("[ffmpeg ERROR]\n", result.stderr[-2000:])
        return False

    shutil.move(out_tmp, OUTPUT_VIDEO)
    return True


def debug_ocr(frame_number=60):
    """
    Run this FIRST to verify EasyOCR is reading your bounding box labels.
    Try different frame numbers until you see your labels with ✔
    """
    reader = easyocr.Reader(['en'], gpu=False)
    cap    = cv2.VideoCapture(INPUT_VIDEO)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"[DEBUG] Could not read frame {frame_number}.")
        return

    print(f"[DEBUG] OCR results for frame {frame_number}:")
    for (_, text, conf) in reader.readtext(frame):
        is_label = "|" in text and "d" in text.lower() and "=" in text
        marker   = "✔" if is_label else " "
        print(f"  {marker} conf={conf:.2f}  '{text}'")


def main():
    # ── STEP 1: Run debug_ocr() first to confirm OCR works ──
    # Uncomment the two lines below, run once, then comment them back out
    # debug_ocr(frame_number=60)
    # return

    # ── STEP 2: Full processing ──
    print("[EchoPath] Loading EasyOCR model...")
    reader = easyocr.Reader(['en'], gpu=False)  # change to gpu=True if you have CUDA

    cap          = cv2.VideoCapture(INPUT_VIDEO)
    fps          = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps

    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {INPUT_VIDEO}")
        return

    print(f"[EchoPath] Video: {total_frames} frames @ {fps:.1f} FPS ({duration_sec:.1f}s)")

    tmp_dir = tempfile.mkdtemp()

    try:
        # Pass 1 — scan and extract alerts
        alert_log = extract_alerts(reader, cap, fps, total_frames)
        cap.release()

        if not alert_log:
            print("\n[!] No alerts found.")
            print("    → Run debug_ocr() on different frame numbers")
            print("    → Try lowering MIN_OCR_CONF to 0.25")
            print("    → Try lowering PROCESS_EVERY_N to 2")
            return

        # Pass 2 — generate TTS WAV clips
        clip_list = generate_audio_clips(alert_log, tmp_dir)

        # Pass 3 — merge with video
        success = merge_audio_video(clip_list, duration_sec, tmp_dir)

        if success:
            print(f"\n Done! Output saved: {OUTPUT_VIDEO}")
            print(f"   Total alerts:    {len(alert_log)}")
            print(f"   Unique phrases:  {len(set(p for _, p, _ in alert_log))}")
        else:
            print("\n[ERROR] ffmpeg merge failed. Check error above.")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()