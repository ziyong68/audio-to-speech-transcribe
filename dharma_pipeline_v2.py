"""
Dharma Archive Pipeline — Multi-Track + Manual Metadata Edition
===============================================================
- Manual entry: teacher, teaching name, date, location
- CD folder input: merges all tracks → single transcript per CD
- Functional style: TypedDict, pure functions, pipe composition
- Skip logic: already-processed discs are skipped automatically

Requirements:
    pip install whisperx torch anthropic ffmpeg-python tqdm python-dotenv

System:
    - CUDA-capable GPU (4GB+ VRAM for large-v2)
    - ffmpeg installed
    - ANTHROPIC_API_KEY in .env file
"""

import os
import re
import json
import sqlite3
import subprocess
import logging
from pathlib import Path
from datetime import datetime
from typing import TypedDict, Optional
from functools import reduce

import torch
import whisperx
import anthropic
from dotenv import load_dotenv
from tqdm import tqdm

# ── Load .env before anything else ────────────────────────────
load_dotenv()

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE  = "float16" if DEVICE == "cuda" else "int8"
WHISPER_MODEL = "large-v2"
BATCH_SIZE    = 16
HF_TOKEN      = os.getenv("HF_TOKEN", "")
DB_PATH       = "db/dharma_archive.db"
AUDIO_OUT_DIR = Path("data/processed")
LOG_PATH      = "logs/pipeline.log"

AUDIO_EXTENSIONS = {".mp3", ".flac", ".wav", ".m4a", ".aac", ".ogg", ".wma", ".mp4"}


# ─────────────────────────────────────────────
# ENSURE REQUIRED DIRS EXIST
# ─────────────────────────────────────────────

def ensure_dirs() -> None:
    """Create required project folders if they don't exist."""
    for folder in ["db", "data/processed", "data/exports", "data/raw", "logs"]:
        Path(folder).mkdir(parents=True, exist_ok=True)

ensure_dirs()


# ─────────────────────────────────────────────
# LOGGING — after ensure_dirs so logs/ exists
# ─────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler()]
)
log = logging.getLogger(__name__)

log.info(f"Device: {DEVICE}  |  Torch: {torch.__version__}")
if DEVICE == "cuda":
    log.info(f"GPU: {torch.cuda.get_device_name(0)}")
    log.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# ─────────────────────────────────────────────
# DATA SHAPES
# ─────────────────────────────────────────────

class DiscMeta(TypedDict):
    disc_id:       str
    teacher:       str
    teaching_name: str
    teaching_date: str    # YYYY-MM-DD or free text e.g. "Summer 2019"
    location:      str
    tradition:     str
    series:        str
    disc_number:   int
    source_format: str
    notes:         str

class TrackInfo(TypedDict):
    track_number: int
    filename:     str
    path:         str
    duration_sec: float

class TranscriptResult(TypedDict):
    text:           str
    language:       str
    segments:       list
    speakers:       list
    track_segments: list

class LLMResult(TypedDict):
    cleaned_transcript:    str
    summary:               str
    key_teachings:         list
    tibetan_terms:         list
    tags:                  list
    language:              str
    estimated_duration_min: int

class TeachingRecord(TypedDict):
    disc_id:             str
    teacher:             str
    teaching_name:       str
    teaching_date:       str
    location:            str
    tradition:           str
    series:              str
    disc_number:         int
    source_format:       str
    notes:               str
    audio_path:          str
    track_count:         int
    track_listing:       str   # JSON array
    language:            str
    duration_min:        int
    transcript_raw:      str
    transcript_clean:    str
    transcript_segments: str   # JSON — word-level
    track_segments:      str   # JSON — per-track slices
    summary:             str
    key_teachings:       str   # JSON array
    tibetan_terms:       str   # JSON array
    tags:                str   # JSON array
    speakers:            str   # JSON array
    created_at:          str
    error:               str


# ─────────────────────────────────────────────
# PURE RECORD HELPERS
# ─────────────────────────────────────────────

def make_record(meta: DiscMeta, tracks: list = None) -> TeachingRecord:
    """Construct a blank TeachingRecord from disc metadata."""
    tracks = tracks or []
    return {
        **meta,
        "audio_path":          "",
        "track_count":         len(tracks),
        "track_listing":       json.dumps(tracks),
        "language":            "",
        "duration_min":        0,
        "transcript_raw":      "",
        "transcript_clean":    "",
        "transcript_segments": "[]",
        "track_segments":      "[]",
        "summary":             "",
        "key_teachings":       "[]",
        "tibetan_terms":       "[]",
        "tags":                "[]",
        "speakers":            "[]",
        "created_at":          datetime.now().isoformat(),
        "error":               "",
    }

def merge_transcript(record: TeachingRecord, result: TranscriptResult) -> TeachingRecord:
    return {
        **record,
        "transcript_raw":      result["text"],
        "language":            result["language"],
        "transcript_segments": json.dumps(result["segments"]),
        "track_segments":      json.dumps(result["track_segments"]),
        "speakers":            json.dumps(result["speakers"]),
    }

def merge_llm(record: TeachingRecord, result: LLMResult) -> TeachingRecord:
    return {
        **record,
        "transcript_clean": result.get("cleaned_transcript", ""),
        "summary":          result.get("summary", ""),
        "key_teachings":    json.dumps(result.get("key_teachings", [])),
        "tibetan_terms":    json.dumps(result.get("tibetan_terms", [])),
        "tags":             json.dumps(result.get("tags", [])),
        "language":         result.get("language", record["language"]),
        "duration_min":     result.get("estimated_duration_min", 0),
    }

def set_error(record: TeachingRecord, error: str) -> TeachingRecord:
    return {**record, "error": str(error)}


# ─────────────────────────────────────────────
# MANUAL METADATA ENTRY (terminal prompts)
# ─────────────────────────────────────────────

def prompt_field(label: str, default: str = "", required: bool = False) -> str:
    """Single field prompt with optional default."""
    hint = f" [{default}]" if default else ""
    req  = " *" if required else ""
    while True:
        val = input(f"  {label}{req}{hint}: ").strip()
        if val:
            return val
        if default:
            return default
        if not required:
            return ""
        print(f"  ✗ '{label}' is required.")


def validate_date(s: str) -> str:
    """Accept YYYY-MM-DD, YYYY-MM, YYYY, or free text like 'Summer 2019'."""
    if re.match(r"^\d{4}(-\d{2}(-\d{2})?)?$", s):
        return s
    return s    # accept free text as-is


def enter_metadata(cd_folder: str, existing: Optional[DiscMeta] = None) -> DiscMeta:
    """
    Interactive manual metadata entry for one CD.
    Pre-fills from existing if provided (useful for series of discs).
    """
    folder_name = Path(cd_folder).name
    prev        = existing or {}

    print(f"\n{'═'*55}")
    print(f"  METADATA ENTRY — {folder_name}")
    print(f"{'═'*55}")
    print("  (* required    [default] = press Enter to keep)\n")

    teacher       = prompt_field("Teacher name",     prev.get("teacher", ""),       required=True)
    teaching_name = prompt_field("Teaching name",    prev.get("teaching_name", ""), required=True)
    raw_date      = prompt_field("Date of teaching", prev.get("teaching_date", ""))
    teaching_date = validate_date(raw_date) if raw_date else ""
    location      = prompt_field("Location",         prev.get("location", ""))
    tradition     = prompt_field("Tradition",        prev.get("tradition", ""))
    series        = prompt_field("Series / retreat", prev.get("series", ""))
    disc_number   = prompt_field("Disc number",      str(prev.get("disc_number", 1)))
    notes         = prompt_field("Notes",            prev.get("notes", ""))

    safe_teacher  = re.sub(r"\W+", "-", teacher.upper())[:20]
    safe_name     = re.sub(r"\W+", "-", teaching_name.upper())[:20]
    disc_id       = prev.get("disc_id") or f"{safe_teacher}__{safe_name}__D{disc_number.zfill(3)}"

    meta: DiscMeta = {
        "disc_id":       disc_id,
        "teacher":       teacher,
        "teaching_name": teaching_name,
        "teaching_date": teaching_date,
        "location":      location,
        "tradition":     tradition,
        "series":        series,
        "disc_number":   int(disc_number) if disc_number.isdigit() else 1,
        "source_format": "CD",
        "notes":         notes,
    }
    print(f"\n  ✓ disc_id: {disc_id}\n")
    return meta


def enter_metadata_batch(cd_folders: list[str]) -> list[tuple[str, DiscMeta]]:
    """
    Enter metadata for multiple CD folders sequentially.
    Carries teacher/tradition/series/location forward as defaults.
    """
    jobs     = []
    defaults = {}

    for i, folder in enumerate(cd_folders, 1):
        print(f"\n  CD {i} of {len(cd_folders)}")
        meta = enter_metadata(folder, defaults)
        jobs.append((folder, meta))
        defaults = {
            "teacher":   meta["teacher"],
            "tradition": meta["tradition"],
            "series":    meta["series"],
            "location":  meta["location"],
        }
    return jobs


# ─────────────────────────────────────────────
# TRACK DISCOVERY & ORDERING
# ─────────────────────────────────────────────

def natural_sort_key(s: str) -> list:
    """Sort 'Track2' before 'Track10'."""
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r"(\d+)", s)]


def get_audio_duration(path: str) -> float:
    """Return duration in seconds via ffprobe."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        capture_output=True, text=True
    )
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0


def discover_tracks(cd_folder: str) -> list[TrackInfo]:
    """
    Scan a CD folder and return tracks in natural sort order.
    Handles: Track01.flac, 01-teaching.mp3, track_1.wav, etc.
    """
    folder = Path(cd_folder)
    files  = sorted(
        [f for f in folder.iterdir() if f.suffix.lower() in AUDIO_EXTENSIONS],
        key=lambda f: natural_sort_key(f.name)
    )
    if not files:
        raise FileNotFoundError(f"No audio files found in {cd_folder}")

    tracks = []
    for i, f in enumerate(files, 1):
        duration = get_audio_duration(str(f))
        tracks.append({
            "track_number": i,
            "filename":     f.name,
            "path":         str(f),
            "duration_sec": duration,
        })
        log.info(f"  Track {i:02d}: {f.name} ({duration/60:.1f} min)")

    log.info(f"Found {len(tracks)} tracks in {cd_folder}")
    return tracks


# ─────────────────────────────────────────────
# TRACK MERGING
# ─────────────────────────────────────────────

def build_ffmpeg_concat_file(tracks: list[TrackInfo], tmp_path: Path) -> str:
    concat_file = tmp_path / "concat_list.txt"
    with open(concat_file, "w") as f:
        for track in tracks:
            escaped = str(track["path"]).replace("'", r"\'")
            f.write(f"file '{escaped}'\n")
    return str(concat_file)


def merge_tracks(tracks: list[TrackInfo], disc_id: str,
                 out_dir: Path = AUDIO_OUT_DIR) -> str:
    """
    Concatenate all tracks → single mono 16kHz MP3.
    Returns path to merged file. Skips if already exists.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{disc_id}_merged.mp3"

    if out_path.exists():
        log.info(f"Merged audio exists, reusing: {out_path}")
        return str(out_path)

    concat_file = build_ffmpeg_concat_file(tracks, out_dir)
    cmd = [
        "ffmpeg", "-f", "concat", "-safe", "0", "-i", concat_file,
        "-ac", "1",       # mono
        "-ar", "16000",   # 16kHz — Whisper native rate
        "-b:a", "64k",
        str(out_path), "-y", "-loglevel", "error"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg merge failed: {result.stderr}")

    total_min = sum(t["duration_sec"] for t in tracks) / 60
    log.info(f"Merged {len(tracks)} tracks → {out_path} ({total_min:.1f} min)")
    return str(out_path)


def compute_track_offsets(tracks: list[TrackInfo]) -> list[dict]:
    """Cumulative time offsets — used to map timestamps back to tracks."""
    offsets, cursor = [], 0.0
    for t in tracks:
        offsets.append({
            "track_number": t["track_number"],
            "filename":     t["filename"],
            "start_sec":    cursor,
            "end_sec":      cursor + t["duration_sec"],
            "duration_sec": t["duration_sec"],
        })
        cursor += t["duration_sec"]
    return offsets


def assign_segments_to_tracks(segments: list[dict],
                               track_offsets: list[dict]) -> list[dict]:
    """Annotate each Whisper segment with its source track."""
    annotated = []
    for seg in segments:
        mid   = (seg.get("start", 0) + seg.get("end", 0)) / 2
        track = next(
            (t for t in track_offsets if t["start_sec"] <= mid < t["end_sec"]),
            track_offsets[-1]
        )
        annotated.append({
            **seg,
            "track_number":   track["track_number"],
            "track_filename": track["filename"],
        })
    return annotated


# ─────────────────────────────────────────────
# WHISPER (model passed explicitly — no hidden state)
# ─────────────────────────────────────────────

def load_whisper_model() -> dict:
    log.info(f"Loading WhisperX {WHISPER_MODEL} on {DEVICE} ({COMPUTE_TYPE})")
    model = whisperx.load_model(
        WHISPER_MODEL, device=DEVICE,
        compute_type=COMPUTE_TYPE, language=None
    )
    log.info("WhisperX model loaded ✓")
    return {"model": model, "align_cache": {}}


def align_segments(segments: list, language: str, audio,
                   model_bundle: dict) -> list:
    alignable = ("en", "zh", "ja", "ko", "fr", "de", "es")
    if language not in alignable:
        return segments   # Tibetan: skip alignment
    cache = model_bundle["align_cache"]
    if language not in cache:
        am, md = whisperx.load_align_model(language_code=language, device=DEVICE)
        cache[language] = (am, md)
    am, md = cache[language]
    return whisperx.align(segments, am, md, audio, DEVICE,
                          return_char_alignments=False)["segments"]


def transcribe(audio_path: str, model_bundle: dict,
               track_offsets: list[dict],
               diarize: bool = False) -> TranscriptResult:
    log.info(f"Transcribing: {audio_path}")
    audio    = whisperx.load_audio(audio_path)
    raw      = model_bundle["model"].transcribe(audio, batch_size=BATCH_SIZE,
                                                print_progress=True)
    lang     = raw.get("language", "unknown")
    log.info(f"Detected language: {lang}")

    segments = align_segments(raw["segments"], lang, audio, model_bundle)

    speakers = []
    if diarize and HF_TOKEN:
        dm       = whisperx.DiarizationPipeline(use_auth_token=HF_TOKEN, device=DEVICE)
        result   = whisperx.assign_word_speakers(dm(audio), {"segments": segments})
        segments = result["segments"]
        speakers = list({s.get("speaker", "") for s in segments if s.get("speaker")})

    track_segs = assign_segments_to_tracks(segments, track_offsets) if track_offsets else segments
    full_text  = " ".join(seg["text"].strip() for seg in segments)

    return {
        "text":           full_text,
        "language":       lang,
        "segments":       segments,
        "speakers":       speakers,
        "track_segments": track_segs,
    }


def free_whisper(model_bundle: dict) -> None:
    """Release GPU VRAM. Call once after all discs are processed."""
    del model_bundle["model"]
    model_bundle["align_cache"].clear()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    log.info("GPU memory freed ✓")


# ─────────────────────────────────────────────
# LLM CLEANING (Claude Haiku)
# ─────────────────────────────────────────────

CLEAN_SYSTEM = """You process dharma teaching transcripts from CDs.
Speakers may mix Tibetan, Sanskrit, and English.

Return ONLY valid JSON — no markdown, no preamble:
{
  "cleaned_transcript": "full corrected transcript with paragraph breaks",
  "summary": "3-5 sentence summary",
  "key_teachings": ["main point 1", "main point 2"],
  "tibetan_terms": ["rigpa", "dzogchen"],
  "language": "english | tibetan | mixed",
  "tags": ["meditation", "mahamudra"],
  "estimated_duration_min": 45
}

Rules:
- Preserve all Tibetan/Sanskrit terms exactly
- Fix speech artifacts (um, uh, false starts)
- Add paragraph breaks at topic shifts
- Keep the teacher's authentic voice
- If transcript spans multiple tracks, maintain continuity"""


def build_llm_prompt(raw: str, meta: DiscMeta, track_count: int) -> str:
    return (
        f"Teacher: {meta['teacher']}\n"
        f"Teaching: {meta['teaching_name']}\n"
        f"Date: {meta['teaching_date']}\n"
        f"Location: {meta['location']}\n"
        f"Tradition: {meta['tradition']}\n"
        f"Series: {meta['series']}\n"
        f"Tracks: {track_count}\n\n"
        f"RAW TRANSCRIPT:\n{raw[:12000]}"
    )


def clean_transcript(raw: str, meta: DiscMeta,
                     track_count: int = 1) -> LLMResult:
    client = anthropic.Anthropic()
    try:
        response = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=4096,
            system=CLEAN_SYSTEM,
            messages=[{"role": "user", "content": build_llm_prompt(raw, meta, track_count)}]
        )
        text = response.content[0].text.replace("```json", "").replace("```", "").strip()
        return json.loads(text)
    except json.JSONDecodeError as e:
        log.warning(f"LLM JSON parse failed: {e} — using raw transcript")
        return {
            "cleaned_transcript": raw, "summary": "",
            "key_teachings": [], "tibetan_terms": [],
            "language": "unknown", "tags": [], "estimated_duration_min": 0
        }
    except Exception as e:
        log.error(f"LLM call failed: {e}")
        raise


# ─────────────────────────────────────────────
# DATABASE
# ─────────────────────────────────────────────

COLUMNS = (
    "disc_id", "teacher", "teaching_name", "teaching_date", "location",
    "tradition", "series", "disc_number", "source_format", "notes",
    "audio_path", "track_count", "track_listing",
    "language", "duration_min",
    "transcript_raw", "transcript_clean", "transcript_segments", "track_segments",
    "summary", "key_teachings", "tibetan_terms", "tags", "speakers",
    "created_at", "error"
)


def init_db(db_path: str = DB_PATH) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS teachings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            {', '.join(f'{c} TEXT' for c in COLUMNS)},
            UNIQUE(disc_id)
        )
    """)
    conn.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS teachings_fts USING fts5(
            disc_id, teacher, teaching_name, location, tradition,
            transcript_clean, summary, key_teachings, tibetan_terms,
            content='teachings', content_rowid='id'
        )
    """)
    conn.commit()
    return conn


def save_record(conn: sqlite3.Connection,
                record: TeachingRecord) -> TeachingRecord:
    """Persist record to DB. Returns same record (pass-through)."""
    placeholders = ", ".join(f":{c}" for c in COLUMNS)
    conn.execute(
        f"INSERT OR REPLACE INTO teachings ({', '.join(COLUMNS)}) VALUES ({placeholders})",
        {c: record.get(c, "") for c in COLUMNS}
    )
    conn.execute("""
        INSERT OR REPLACE INTO teachings_fts
        (disc_id, teacher, teaching_name, location, tradition,
         transcript_clean, summary, key_teachings, tibetan_terms)
        VALUES (?,?,?,?,?,?,?,?,?)
    """, (
        record["disc_id"], record["teacher"], record["teaching_name"],
        record["location"], record["tradition"], record["transcript_clean"],
        record["summary"], record["key_teachings"], record["tibetan_terms"]
    ))
    conn.commit()
    log.info(f"Saved: {record['disc_id']} ({record['track_count']} tracks)")
    return record


def already_processed(disc_id: str, db_path: str = DB_PATH) -> bool:
    """Return True if disc_id exists in DB with no error."""
    try:
        conn = sqlite3.connect(db_path)
        row  = conn.execute(
            "SELECT id FROM teachings WHERE disc_id = ? AND error = ''",
            (disc_id,)
        ).fetchone()
        conn.close()
        return row is not None
    except sqlite3.OperationalError:
        return False   # table doesn't exist yet


# ─────────────────────────────────────────────
# PIPELINE STEPS
# Each step: TeachingRecord → TeachingRecord
# ─────────────────────────────────────────────

def step_discover_and_merge(cd_folder: str):
    def _step(record: TeachingRecord) -> TeachingRecord:
        tracks        = discover_tracks(cd_folder)
        track_offsets = compute_track_offsets(tracks)
        merged_path   = merge_tracks(tracks, record["disc_id"])
        return {
            **record,
            "audio_path":     merged_path,
            "track_count":    len(tracks),
            "track_listing":  json.dumps(tracks),
            "_track_offsets": track_offsets,   # ephemeral key
        }
    return _step


def step_transcribe(model_bundle: dict, diarize: bool = False):
    def _step(record: TeachingRecord) -> TeachingRecord:
        offsets = record.get("_track_offsets", [])
        result  = transcribe(record["audio_path"], model_bundle, offsets, diarize)
        updated = merge_transcript(record, result)
        updated.pop("_track_offsets", None)   # remove ephemeral key
        return updated
    return _step


def step_clean_llm(meta: DiscMeta):
    def _step(record: TeachingRecord) -> TeachingRecord:
        if not record["transcript_raw"].strip():
            log.warning(f"Empty transcript for {record['disc_id']} — skipping LLM")
            return record
        result = clean_transcript(record["transcript_raw"], meta, record["track_count"])
        return merge_llm(record, result)
    return _step


def step_save(conn: sqlite3.Connection):
    def _step(record: TeachingRecord) -> TeachingRecord:
        return save_record(conn, record)
    return _step


def safe_pipe(*fns):
    """
    Left-to-right function composition with error capture.
    On exception: stores error in record and short-circuits.
    """
    def _run(record: TeachingRecord) -> TeachingRecord:
        current = record
        for fn in fns:
            try:
                current = fn(current)
            except Exception as e:
                fn_name = getattr(fn, "__name__", str(fn))
                log.error(f"Step '{fn_name}' failed for {record['disc_id']}: {e}")
                current = set_error(current, e)
                break
        return current
    return _run


def process_cd_folder(
    cd_folder:    str,
    meta:         DiscMeta,
    model_bundle: dict,
    conn:         sqlite3.Connection,
    diarize:      bool = False,
    skip_llm:     bool = False,
) -> TeachingRecord:
    """One CD folder → one TeachingRecord saved to DB."""
    steps = [
        step_discover_and_merge(cd_folder),
        step_transcribe(model_bundle, diarize),
    ]
    if not skip_llm:
        steps.append(step_clean_llm(meta))
    steps.append(step_save(conn))

    return safe_pipe(*steps)(make_record(meta))


# ─────────────────────────────────────────────
# BATCH RUNNER
# ─────────────────────────────────────────────

def batch_process(
    jobs:     list[tuple[str, DiscMeta]],
    db_path:  str  = DB_PATH,
    diarize:  bool = False,
    skip_llm: bool = False,
) -> list[TeachingRecord]:
    """
    Process a list of (cd_folder, meta) pairs.
    - Whisper loaded once, reused for all CDs
    - Already-processed discs are skipped automatically
    """
    conn         = init_db(db_path)
    model_bundle = load_whisper_model()
    results      = []

    for folder, meta in tqdm(jobs, desc="Processing CDs"):
        log.info(f"\n{'─'*50}")
        log.info(f"  {meta['teacher']} — {meta['teaching_name']}")
        log.info(f"  {meta['teaching_date']}  |  {meta['location']}")

        if already_processed(meta["disc_id"], db_path):
            log.info(f"  ↷ Skipping {meta['disc_id']} — already in DB")
            continue

        record = process_cd_folder(folder, meta, model_bundle, conn, diarize, skip_llm)
        results.append(record)

        if record["error"]:
            log.warning(f"  ✗ Error: {record['error']}")
        else:
            log.info(
                f"  ✓ Saved  |  Tracks: {record['track_count']}"
                f"  |  Lang: {record['language']}"
                f"  |  {record['duration_min']} min"
            )

    free_whisper(model_bundle)
    conn.close()

    ok     = [r for r in results if not r["error"]]
    failed = [r for r in results if r["error"]]
    log.info(f"\n✓ Complete — {len(ok)} ok, {len(failed)} failed → {db_path}")
    if failed:
        for r in failed:
            log.warning(f"  ✗ {r['disc_id']}: {r['error']}")
    return results


# ─────────────────────────────────────────────
# INTERACTIVE ENTRY POINT
# ─────────────────────────────────────────────

def run_interactive(cd_folders: list[str], **kwargs):
    """
    Prompt for metadata on each CD folder, then run the pipeline.

    Usage:
        run_interactive(["data/raw/disc_01", "data/raw/disc_02"])
    """
    print(f"\n{'═'*55}")
    print(f"  DHARMA ARCHIVE — {len(cd_folders)} CD(s) to process")
    print(f"{'═'*55}")

    jobs = enter_metadata_batch(cd_folders)

    print(f"\n{'─'*55}")
    print("  Starting pipeline...")
    print(f"{'─'*55}\n")

    return batch_process(jobs, **kwargs)


# ─────────────────────────────────────────────
# QUERY HELPERS
# ─────────────────────────────────────────────

def search_teachings(query: str, db_path: str = DB_PATH) -> list[dict]:
    """Full-text search across transcripts, summaries, teachers."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT t.disc_id, t.teacher, t.teaching_name, t.teaching_date,
               t.location, t.tradition, t.summary, t.language,
               t.duration_min, t.track_count,
               snippet(teachings_fts, 5, '[', ']', '...', 20) AS excerpt
        FROM teachings_fts
        JOIN teachings t ON teachings_fts.rowid = t.id
        WHERE teachings_fts MATCH ?
        ORDER BY rank LIMIT 20
    """, (query,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_by_teacher(teacher: str, db_path: str = DB_PATH) -> list[dict]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT disc_id, teaching_name, teaching_date, series, summary "
        "FROM teachings WHERE teacher = ? AND error = '' ORDER BY teaching_date",
        (teacher,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_track_transcript(disc_id: str, track_number: int,
                          db_path: str = DB_PATH) -> list[dict]:
    """Retrieve transcript segments for a specific track within a CD."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    row  = conn.execute(
        "SELECT track_segments FROM teachings WHERE disc_id = ?", (disc_id,)
    ).fetchone()
    conn.close()
    if not row:
        return []
    return [s for s in json.loads(row["track_segments"])
            if s.get("track_number") == track_number]


def export_jsonl(
    db_path:  str = DB_PATH,
    out_path: str = "data/exports/dharma_export.jsonl"
) -> int:
    """Export all clean records as JSONL for RAG / fine-tuning."""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM teachings WHERE error = ''").fetchall()
    conn.close()
    with open(out_path, "w") as f:
        for row in rows:
            f.write(json.dumps(dict(row)) + "\n")
    log.info(f"Exported {len(rows)} records → {out_path}")
    return len(rows)


# ─────────────────────────────────────────────
# EXAMPLE USAGE
# ─────────────────────────────────────────────

if __name__ == "__main__":

    # ── Option A: Interactive (prompts for metadata) ──────────────
    run_interactive([
        "data/raw/disc_01",
        "data/raw/disc_02",
    ], diarize=False, skip_llm=False)


    # ── Option B: Scripted (metadata hardcoded, no prompts) ───────
    # jobs = [
    #     (
    #         "data/raw/disc_01",
    #         {
    #             "disc_id":       "TSOKNYI__NATURE-OF-MIND__D001",
    #             "teacher":       "Tsoknyi Rinpoche",
    #             "teaching_name": "Nature of Mind",
    #             "teaching_date": "2019-08",
    #             "location":      "Crestone, Colorado",
    #             "tradition":     "Nyingma",
    #             "series":        "Summer Retreat 2019",
    #             "disc_number":   1,
    #             "source_format": "CD",
    #             "notes":         "Morning session",
    #         }
    #     ),
    # ]
    # batch_process(jobs, diarize=False, skip_llm=False)


    # ── Query examples ────────────────────────────────────────────
    # for hit in search_teachings("nature of mind rigpa"):
    #     print(f"{hit['teacher']} | {hit['teaching_name']} | {hit['teaching_date']}")
    #     print(f"  Location : {hit['location']}")
    #     print(f"  Tracks   : {hit['track_count']}  Duration: {hit['duration_min']} min")
    #     print(f"  {hit['excerpt']}\n")

    # segs = get_track_transcript("TSOKNYI__NATURE-OF-MIND__D001", track_number=2)
    # for s in segs:
    #     print(f"[{s['start']:.1f}s] {s['text']}")

    # export_jsonl()
