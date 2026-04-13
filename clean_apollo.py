#!/usr/bin/env python3
"""
Apollo.io Contact Export Cleaner
Cleans and transforms Apollo CSV exports for cold outreach.

Usage:
    python clean_apollo.py

Required files:
    apollo-contacts-export.csv   - raw Apollo export
    industries_mapping.csv       - columns: Original_Industry, Target_Industry
    title_mapping.csv            - columns: Title, Right Title
    company_name_training.csv    - columns: Company Name for Emails, Right Company Name

Environment:
    ANTHROPIC_API_KEY            - set in .env or shell
"""

import asyncio
import csv
import hashlib
import json
import logging
import os
import re
import time
import unicodedata
from pathlib import Path

import anthropic
import pandas as pd
from dotenv import load_dotenv

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
load_dotenv()

INPUT_FILE = "apollo-contacts-export.csv"
OUTPUT_FILE = "cleaned_apollo_contacts.csv"
INDUSTRIES_FILE = "industries_mapping.csv"
TITLE_FILE = "title_mapping.csv"
COMPANY_FILE = "company_name_training.csv"

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
CLAUDE_MODEL = "claude-sonnet-4-5-20250929"
BATCH_SIZE = 75          # rows per Claude request
MAX_CONCURRENT = 5       # simultaneous in-flight API calls
MAX_RETRIES = 5
FEW_SHOT_EXAMPLES = 40   # examples pulled from mapping CSVs for the system prompt
LLM_CACHE_DIR = Path("llm_cache")  # folder for checkpoint files

APOLLO_COLUMNS = [
    "First Name", "Last Name", "Title", "Company Name", "Email",
    "# Employees", "Industry", "Person Linkedin Url", "Website",
    "Company Linkedin Url", "City", "State", "Country",
    "Company City", "Company State", "Company Country",
    "Apollo Contact Id", "Apollo Account Id",
]

FINAL_COLUMNS = [
    "Company", "Website", "Industry", "Country", "State", "City",
    "First name", "Last name", "Title", "Email",
    "Linkedin Person", "Linkedin Company", "Number of employees",
    "Person Country", "Person State", "Person City",
    "Empty_1", "Empty_2", "Empty_3", "Empty_4", "Empty_5",
    "Email_Domain_Match", "Apollo Contact Id", "Apollo Account Id",
    "Company_Original", "Title_Original",
]

# Umlaut / diacritic → ASCII map (German convention as requested)
_UMLAUT_MAP = str.maketrans({
    "ä": "ae", "Ä": "Ae",
    "ö": "oe", "Ö": "Oe",
    "ü": "ue", "Ü": "Ue",
    "ß": "ss",
    "ø": "oe", "Ø": "Oe",   # Danish / Norwegian
    "å": "a",  "Å": "A",
    "æ": "ae", "Æ": "Ae",
    "é": "e",  "è": "e",  "ê": "e",  "ë": "e",
    "à": "a",  "â": "a",  "ã": "a",
    "î": "i",  "ï": "i",
    "ô": "o",  "õ": "o",
    "ù": "u",  "û": "u",
    "ç": "c",  "Ç": "C",
    "ñ": "n",  "Ñ": "N",
})


# ── Post-LLM text fixups ───────────────────────────────────────────────────────

def _transliterate(text: str) -> str:
    """Replace umlauts/diacritics with ASCII equivalents."""
    if not text:
        return text
    return text.translate(_UMLAUT_MAP)


def _fix_caps(name: str) -> str:
    """Title-case a company name that is mostly ALL CAPS.
    Short tokens (≤4 alpha chars) that are fully uppercase are kept as-is
    because they are likely acronyms (e.g. NTIC, BRP, ZDS).
    """
    if not name:
        return name
    letters = [c for c in name if c.isalpha()]
    if not letters:
        return name
    upper_ratio = sum(c.isupper() for c in letters) / len(letters)
    if upper_ratio < 0.7:
        return name  # already mixed-case, leave alone
    result = []
    for word in name.split():
        alpha_only = re.sub(r"[^A-Za-zÀ-ÿ]", "", word)
        if word.isupper() and len(alpha_only) <= 4:
            result.append(word)          # keep acronym: NTIC, BRP, ZDS
        else:
            result.append(word.capitalize())   # Schütz, Variovac, Walterwerk
    return " ".join(result)


def clean_company_text(name: str) -> str:
    """Apply caps-fix then umlaut transliteration to a company name."""
    return _transliterate(_fix_caps(str(name) if name else ""))


def clean_title_text(title: str) -> str:
    """Transliterate diacritics in titles (no caps-fix needed)."""
    return _transliterate(str(title) if title else "")


# ══════════════════════════════════════════════════════════════════════════════
# A. Name Cleaning
# ══════════════════════════════════════════════════════════════════════════════

def _clean_name(val) -> str:
    if pd.isna(val) or not str(val).strip():
        return ""
    name = str(val).strip()
    # ALL CAPS → Title Case (e.g. "JOHN" → "John", "VAN DEN BERG" → "Van Den Berg")
    if name.isupper():
        name = name.title()
    # Strip diacritics for names: é→e, ä→a, ö→o, ü→u (not German ae/oe/ue convention)
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    # Collapse multiple spaces
    name = re.sub(r" {2,}", " ", name)
    return name.strip()


def clean_names(df: pd.DataFrame) -> pd.DataFrame:
    before_first = df["First Name"].copy()
    before_last  = df["Last Name"].copy()
    df["First Name"] = df["First Name"].apply(_clean_name)
    df["Last Name"]  = df["Last Name"].apply(_clean_name)
    changed = ((df["First Name"] != before_first.fillna("")) |
               (df["Last Name"]  != before_last.fillna(""))).sum()
    log.info("Names cleaned (%d rows changed)", changed)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# B. Deduplication
# ══════════════════════════════════════════════════════════════════════════════

def deduplicate_contacts(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    # Rows with a non-empty email: keep first occurrence, drop the rest
    has_email = df["Email"].notna() & (df["Email"].str.strip() != "")
    email_norm = df["Email"].str.strip().str.lower()
    is_dup = has_email & email_norm.duplicated(keep="first")
    removed = is_dup.sum()
    df = df[~is_dup].copy().reset_index(drop=True)
    if removed:
        log.info("Deduplication: removed %d duplicate email(s) (%d → %d rows)",
                 removed, before, len(df))
    else:
        log.info("Deduplication: no duplicates found")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 1. Data Loading and Filtering
# ══════════════════════════════════════════════════════════════════════════════

def _normalize(s: str) -> str:
    """Lowercase, remove spaces/special chars for fuzzy column matching."""
    return re.sub(r"[^a-z0-9]", "", s.lower())


def load_and_filter(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, low_memory=False)
    log.info("Loaded %d rows × %d cols from '%s'", len(df), len(df.columns), path)

    # Build a fuzzy name → canonical name map
    col_rename: dict[str, str] = {}
    target_norm = {_normalize(t): t for t in APOLLO_COLUMNS}
    for col in df.columns:
        canon = target_norm.get(_normalize(col))
        if canon:
            col_rename[col] = canon

    missing = [c for c in APOLLO_COLUMNS if c not in col_rename.values()]
    if missing:
        log.warning("Columns not found in source (will be empty): %s", missing)

    df = df.rename(columns=col_rename)
    for col in APOLLO_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA

    df = df[APOLLO_COLUMNS].copy().reset_index(drop=True)
    log.info("Kept %d target columns, %d rows", len(df.columns), len(df))
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 2. URL Cleaning
# ══════════════════════════════════════════════════════════════════════════════

_URL_PREFIX = re.compile(r"^https?://(www\.)?", re.IGNORECASE)

def _clean_url(val) -> str:
    if pd.isna(val) or not str(val).strip():
        return ""
    return _URL_PREFIX.sub("", str(val).strip()).rstrip("/")


def clean_urls(df: pd.DataFrame) -> pd.DataFrame:
    for col in ("Website", "Person Linkedin Url", "Company Linkedin Url"):
        df[col] = df[col].apply(_clean_url)
    log.info("URL columns cleaned")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 3. Email / Domain Match
# ══════════════════════════════════════════════════════════════════════════════

def _bare_domain(url: str) -> str:
    """Return only the hostname part of a (already cleaned) URL."""
    return url.split("/")[0].lower() if url else ""


def email_domain_match(df: pd.DataFrame) -> pd.DataFrame:
    def _match(row) -> bool:
        email = str(row["Email"]) if pd.notna(row["Email"]) else ""
        website = str(row["Website"]) if pd.notna(row["Website"]) else ""
        if "@" not in email or not website:
            return False
        email_domain = email.split("@")[-1].lower()
        site_domain = _bare_domain(website)
        return site_domain in email_domain or email_domain in site_domain

    df["Email_Domain_Match"] = df.apply(_match, axis=1)
    log.info("Email_Domain_Match column added (True: %d)", df["Email_Domain_Match"].sum())
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 4. Employee Bucketing
# ══════════════════════════════════════════════════════════════════════════════

_BUCKETS = [
    (1,    10,   "1-10"),
    (11,   50,   "11-50"),
    (51,   100,  "51-100"),
    (101,  500,  "101-500"),
    (501,  1000, "501-1000"),
    (1001, 5000, "1001-5000"),
]


def _bucket(val) -> str:
    if pd.isna(val) or not str(val).strip():
        return ""
    nums = re.findall(r"[\d,]+", str(val))
    if not nums:
        return ""
    try:
        n = int(nums[0].replace(",", ""))
    except ValueError:
        return ""
    for lo, hi, label in _BUCKETS:
        if lo <= n <= hi:
            return label
    return "5000+"


def apply_employee_buckets(df: pd.DataFrame) -> pd.DataFrame:
    df["# Employees"] = df["# Employees"].apply(_bucket)
    log.info("Employee column bucketed")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 5. Industry Mapping
# ══════════════════════════════════════════════════════════════════════════════

def load_industry_mapping(path: str) -> dict[str, str]:
    if not Path(path).exists():
        log.warning("'%s' not found – industry values left as-is", path)
        return {}
    mapping: dict[str, str] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            orig = row.get("Original_Industry", "").strip().lower()
            target = row.get("Target_Industry", "").strip()
            if orig:
                mapping[orig] = target
    log.info("Loaded %d industry mappings from '%s'", len(mapping), path)
    return mapping


def apply_industry_mapping(df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    def _map(val):
        if pd.isna(val) or not str(val).strip():
            return val
        return mapping.get(str(val).strip().lower(), val)

    df["Industry"] = df["Industry"].apply(_map)
    log.info("Industry mapping applied")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 6. LLM Processing (Claude 3.5 Sonnet, async batched)
# ══════════════════════════════════════════════════════════════════════════════

def _load_examples(path: str, orig_col: str, clean_col: str, n: int) -> list[dict]:
    if not Path(path).exists():
        log.warning("Example file '%s' not found", path)
        return []
    examples = []
    with open(path, newline="", encoding="utf-8") as f:
        for i, row in enumerate(csv.DictReader(f)):
            if i >= n:
                break
            orig = row.get(orig_col, "").strip()
            clean = row.get(clean_col, "").strip()
            if orig and clean and orig != clean:
                examples.append({"original": orig, "cleaned": clean})
    return examples


def _build_company_prompt(examples: list[dict]) -> str:
    return f"""You clean company names for B2B cold email personalisation.
Return ONLY the conversational brand name a person would say out loud.

RULES (apply every rule to every input):
1. Strip ALL legal suffixes regardless of language:
   Inc., LLC, Ltd., GmbH, AG, KG, KGaA, ULC, AB, AS, A/S, Oy, OY, B.V., BV,
   N.V., NV, S.A., SA, SAS, SARL, S.r.l., Srl, ApS, Pte., Pvt., Corp., Co.,
   SpA, OÜ, UAB, and any similar suffix. Also "& Co.", "& Co. KG", "& Co. KGaA".
2. Remove geographic qualifiers appended to a brand name (they are NOT part of the brand):
   country names (Germany, France, UK, Ireland, Switzerland, Sweden, Austria,
   Denmark, Finland, Netherlands, Belgium, USA, America…),
   city names (Hamburg, Berlin, Kiel, München, Vienna, London, Chicago…),
   region words (Europe, Northern Europe, EMEA, Nordic, International, Global,
   North America, Western, Eastern, Southern).
   Exception: keep geographic words that ARE the brand (e.g. "American Pan" stays
   "American Pan" because "American" is the brand word, not a qualifier).
3. Remove personal first names or full names that appear before/after the real brand
   (e.g. "Georg Hagelschuer GmbH" → "Hagelschuer",
        "Willi Mäder AG" → "Pamasol" if "Pamasol" is the product name,
        "Wilfried Heinzel AG" → "Heinzel").
4. If a word is ALL CAPS and has more than 4 letters, convert to Title Case
   (e.g. WALTERWERK → Walterwerk, PRODITEC → Proditec, VARIOVAC → Variovac).
   Short ALL-CAPS acronyms (≤4 letters) stay uppercase (NTIC, BRP, ZDS, NNZ, PWR).
5. Split fused CamelCase or concatenated words into separate words when the parts
   are common business/technical words
   (e.g. "Schobertechnologies" → "Schober Technologies",
        "FormerFab" → "Former Fab", "PartsPak" → "Parts Pak",
        "ControlTech" → "Control Tech").
   Do NOT split intentional brand stylisations like "AstroNova", "PinMeTo", "knoell".
6. Convert umlauts/diacritics to plain ASCII:
   ö→oe, ü→ue, ä→ae, ß→ss, ø→oe, å→a, æ→ae, é/è/ê→e, ç→c.
7. Preserve stylised lower-case brands (knoell, iPhone).
8. Max 3 words; strip "Group", "Holdings", "Solutions", "Services", "Technologies"
   when they are generic filler and not the brand identity.

Examples (original → cleaned):
{json.dumps(examples, ensure_ascii=False, indent=2)}

INPUT : {{"rows": [{{"key": "<str>", "company": "<str>"}}, ...]}}
OUTPUT: {{"results": [{{"key": "<str>", "company": "<cleaned str>"}}, ...]}}
Return ONLY the raw JSON. No markdown, no explanation."""


def _build_title_prompt(examples: list[dict]) -> str:
    return f"""You standardise job titles for B2B cold email personalisation.

RULES:
1. Shorten to the most concise, well-known form (target ≤ 4 words).
2. Use standard English abbreviations:
   CEO, CTO, CFO, COO, VP, SVP, EVP, CMO, CRO, CPO, GM,
   Head of, Director of, Manager of.
3. Drop filler qualifiers — "Global", "Regional", "Senior", "Junior",
   "North America", "EMEA", country/city names — UNLESS they change the meaning.
4. Translate non-English titles to English:
   Geschäftsführer → Managing Director, Produktmanager → Product Manager,
   Algemeen directeur → Managing Director, Teknisk Direktör → Technical Director,
   GF → Managing Director, Inhaber → Owner.
5. Resolve slash/combo titles to the most senior role
   (e.g. "VP / CFO / Treasurer" → "CFO").

Examples (original → cleaned):
{json.dumps(examples, ensure_ascii=False, indent=2)}

INPUT : {{"rows": [{{"key": "<str>", "title": "<str>"}}, ...]}}
OUTPUT: {{"results": [{{"key": "<str>", "title": "<cleaned str>"}}, ...]}}
Return ONLY the raw JSON. No markdown, no explanation."""


# ── Deduplication helpers ──────────────────────────────────────────────────────

def _dedup_companies(df: pd.DataFrame) -> tuple[list[dict], dict[int, str]]:
    """
    Group rows by website domain (primary) or lowercased company name (fallback).
    Returns:
      unique_items  – list of {key, company} for each unique group
      row_to_key    – {row_index → group key}
    """
    row_to_key: dict[int, str] = {}
    key_to_company: dict[str, str] = {}

    for i in df.index:
        domain = _bare_domain(str(df.at[i, "Website"]) if pd.notna(df.at[i, "Website"]) else "")
        company = str(df.at[i, "Company Name"]) if pd.notna(df.at[i, "Company Name"]) else ""
        key = domain if domain else (f"name:{company.lower().strip()}" if company else f"row:{i}")
        row_to_key[i] = key
        if key not in key_to_company and company:
            key_to_company[key] = company

    unique_items = [{"key": k, "company": v} for k, v in key_to_company.items()]
    return unique_items, row_to_key


def _dedup_titles(df: pd.DataFrame) -> tuple[list[dict], dict[int, str]]:
    """
    Deduplicate titles by lowercased/stripped value.
    Returns:
      unique_items  – list of {key, title} for each unique title
      row_to_key    – {row_index → normalised title key}
    """
    row_to_key: dict[int, str] = {}
    key_to_title: dict[str, str] = {}

    for i in df.index:
        title = str(df.at[i, "Title"]) if pd.notna(df.at[i, "Title"]) else ""
        key = title.lower().strip()
        row_to_key[i] = key
        if key not in key_to_title and title:
            key_to_title[key] = title

    unique_items = [{"key": k, "title": v} for k, v in key_to_title.items()]
    return unique_items, row_to_key


# ── Cache helpers ──────────────────────────────────────────────────────────────

def _cache_path(input_file: str) -> Path:
    h = hashlib.md5(Path(input_file).read_bytes()).hexdigest()[:10]
    LLM_CACHE_DIR.mkdir(exist_ok=True)
    return LLM_CACHE_DIR / f"{Path(input_file).stem}_{h}.json"


def _load_cache(path: Path) -> dict:
    """Cache format: {"companies": {key: cleaned}, "titles": {key: cleaned}}"""
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        companies = data.get("companies", {})
        titles = data.get("titles", {})
        log.info("Checkpoint: %d companies, %d titles cached from '%s'",
                 len(companies), len(titles), path)
        return {"companies": companies, "titles": titles}
    return {"companies": {}, "titles": {}}


def _save_cache(path: Path, cache: dict) -> None:
    path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")


# ── API call helpers ───────────────────────────────────────────────────────────

async def _call_with_backoff(client: anthropic.AsyncAnthropic, system: str, user_content: str) -> str:
    delay = 2.0
    for attempt in range(MAX_RETRIES):
        try:
            msg = await client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=8192,
                system=system,
                messages=[{"role": "user", "content": user_content}],
            )
            return msg.content[0].text
        except anthropic.RateLimitError:
            if attempt == MAX_RETRIES - 1:
                raise
            wait = delay * (2 ** attempt)
            log.warning("Rate limit – retry %d/%d in %.1fs", attempt + 1, MAX_RETRIES, wait)
            await asyncio.sleep(wait)
        except anthropic.APIStatusError as exc:
            if exc.status_code >= 500 and attempt < MAX_RETRIES - 1:
                wait = delay * (2 ** attempt)
                log.warning("API %d error – retry %d/%d in %.1fs",
                            exc.status_code, attempt + 1, MAX_RETRIES, wait)
                await asyncio.sleep(wait)
            else:
                raise
    raise RuntimeError("Exhausted retries")


def _parse_llm_response(raw: str, field: str, batch: list[dict]) -> list[dict]:
    """Parse JSON response; fall back to originals on failure."""
    try:
        return json.loads(raw).get("results", [])
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", raw)
        if m:
            try:
                return json.loads(m.group()).get("results", [])
            except Exception:
                pass
    log.error("JSON parse failed; keeping originals for %d items", len(batch))
    return [{**item} for item in batch]  # return unchanged


async def _process_unique_batch(
    client: anthropic.AsyncAnthropic,
    system: str,
    batch: list[dict],
    field: str,            # "company" or "title"
    sem: asyncio.Semaphore,
    batch_idx: int,
    total_batches: int,
    subcache: dict[str, str],   # shared {key → cleaned}
    cache: dict,
    cache_path: Path,
    cache_lock: asyncio.Lock,
    progress: dict,
) -> None:
    async with sem:
        raw = await _call_with_backoff(
            client, system, json.dumps({"rows": batch}, ensure_ascii=False)
        )

    results = _parse_llm_response(raw, field, batch)

    async with cache_lock:
        for item in results:
            subcache[item["key"]] = item.get(field, "")
        _save_cache(cache_path, cache)
        progress["done"] += len(results)

    pct = progress["done"] / progress["total"] * 100
    elapsed = time.perf_counter() - progress["t0"]
    eta = elapsed / progress["done"] * (progress["total"] - progress["done"]) if progress["done"] else 0
    log.info("Batch %d/%d [%s] done – %d/%d (%.0f%%) – %.0fs elapsed – ETA %.0fs",
             batch_idx + 1, total_batches, field,
             progress["done"], progress["total"], pct, elapsed, eta)


async def run_llm_processing(df: pd.DataFrame) -> pd.DataFrame:
    company_ex = _load_examples(COMPANY_FILE, "Company Name for Emails", "Right Company Name", FEW_SHOT_EXAMPLES)
    title_ex   = _load_examples(TITLE_FILE, "Title", "Right Title", FEW_SHOT_EXAMPLES)
    company_prompt = _build_company_prompt(company_ex)
    title_prompt   = _build_title_prompt(title_ex)

    # Build deduplicated unique lists + row→key maps
    unique_companies, row_to_company_key = _dedup_companies(df)
    unique_titles,    row_to_title_key   = _dedup_titles(df)
    log.info(
        "Deduplication: %d rows → %d unique companies, %d unique titles",
        len(df), len(unique_companies), len(unique_titles),
    )

    cp_path = _cache_path(INPUT_FILE)
    cache = _load_cache(cp_path)
    company_cache: dict[str, str] = cache["companies"]
    title_cache:   dict[str, str] = cache["titles"]

    pending_companies = [x for x in unique_companies if x["key"] not in company_cache]
    pending_titles    = [x for x in unique_titles    if x["key"] not in title_cache]
    total_pending = len(pending_companies) + len(pending_titles)

    if total_pending == 0:
        log.info("All values already in checkpoint – skipping API calls")
    else:
        log.info("Pending: %d companies, %d titles to process",
                 len(pending_companies), len(pending_titles))

        client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
        sem = asyncio.Semaphore(MAX_CONCURRENT)
        cache_lock = asyncio.Lock()
        already_done = (len(unique_companies) - len(pending_companies) +
                        len(unique_titles)    - len(pending_titles))
        progress = {"done": already_done, "total": len(unique_companies) + len(unique_titles),
                    "t0": time.perf_counter()}

        company_batches = [pending_companies[i: i + BATCH_SIZE]
                           for i in range(0, len(pending_companies), BATCH_SIZE)]
        title_batches   = [pending_titles[i: i + BATCH_SIZE]
                           for i in range(0, len(pending_titles), BATCH_SIZE)]
        total_batches = len(company_batches) + len(title_batches)

        tasks = [
            _process_unique_batch(client, company_prompt, b, "company", sem,
                                  idx, total_batches, company_cache,
                                  cache, cp_path, cache_lock, progress)
            for idx, b in enumerate(company_batches)
        ] + [
            _process_unique_batch(client, title_prompt, b, "title", sem,
                                  len(company_batches) + idx, total_batches,
                                  title_cache, cache, cp_path, cache_lock, progress)
            for idx, b in enumerate(title_batches)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for idx, r in enumerate(results):
            if isinstance(r, Exception):
                log.error("Batch %d failed: %s", idx, r)

    # Map cleaned values back to each row, then apply deterministic post-processing
    df["Company_Clean"] = [
        clean_company_text(company_cache.get(row_to_company_key[i])
                           or str(df.at[i, "Company Name"] or ""))
        for i in df.index
    ]
    df["Title_Clean"] = [
        clean_title_text(title_cache.get(row_to_title_key[i])
                         or str(df.at[i, "Title"] or ""))
        for i in df.index
    ]
    log.info("LLM processing complete. Checkpoint at '%s'", cp_path)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 7. Rename and Reorder
# ══════════════════════════════════════════════════════════════════════════════

def rename_and_reorder(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["Company"]             = df["Company_Clean"]
    out["Website"]             = df["Website"]
    out["Industry"]            = df["Industry"]
    out["Country"]             = df["Company Country"]
    out["State"]               = df["Company State"]
    out["City"]                = df["Company City"]
    out["First name"]          = df["First Name"]
    out["Last name"]           = df["Last Name"]
    out["Title"]               = df["Title_Clean"]
    out["Email"]               = df["Email"]
    out["Linkedin Person"]     = df["Person Linkedin Url"]
    out["Linkedin Company"]    = df["Company Linkedin Url"]
    out["Number of employees"] = df["# Employees"]
    out["Person Country"]      = df["Country"]
    out["Person State"]        = df["State"]
    out["Person City"]         = df["City"]
    out["Empty_1"]             = ""
    out["Empty_2"]             = ""
    out["Empty_3"]             = ""
    out["Empty_4"]             = ""
    out["Empty_5"]             = ""
    out["Email_Domain_Match"]  = df["Email_Domain_Match"]
    out["Apollo Contact Id"]   = df["Apollo Contact Id"]
    out["Apollo Account Id"]   = df["Apollo Account Id"]
    out["Company_Original"]    = df["Company Name"]
    out["Title_Original"]      = df["Title"]
    return out[FINAL_COLUMNS]


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

async def main() -> None:
    t0 = time.perf_counter()
    log.info("═══ Apollo Contact Cleaner START ═══")

    # 1. Load & filter
    df = load_and_filter(INPUT_FILE)

    # 1a. Clean names (ALL CAPS, diacritics)
    df = clean_names(df)

    # 1b. Remove duplicate emails (keep first occurrence)
    df = deduplicate_contacts(df)

    # 2. Clean URLs
    df = clean_urls(df)

    # 3. Email / domain match
    df = email_domain_match(df)

    # 4. Employee bucketing
    df = apply_employee_buckets(df)

    # 5. Industry mapping
    industry_map = load_industry_mapping(INDUSTRIES_FILE)
    df = apply_industry_mapping(df, industry_map)

    # 6. LLM processing
    if not ANTHROPIC_API_KEY:
        log.warning("ANTHROPIC_API_KEY not set – skipping LLM step, using raw values")
        df["Title_Clean"] = df["Title"].fillna("")
        df["Company_Clean"] = df["Company Name"].fillna("")
    else:
        df = await run_llm_processing(df)

    # 7. Rename & reorder
    final = rename_and_reorder(df)

    # 8. Save
    final.to_csv(OUTPUT_FILE, index=False)
    elapsed = time.perf_counter() - t0
    log.info("═══ Done! %d rows → '%s' in %.1fs ═══", len(final), OUTPUT_FILE, elapsed)


if __name__ == "__main__":
    asyncio.run(main())
