from flask import Flask, render_template, request,current_app
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import event
from pgvector.sqlalchemy import Vector
from sqlalchemy.sql import text as sa_text
from sentence_transformers import SentenceTransformer
from werkzeug.utils import secure_filename
from functools import lru_cache
from pdfminer.high_level import extract_text as extract_text_pdfminer
import argparse
import os
import re
from dotenv import load_dotenv
import unicodedata
import markdown2
from app.config import Config

from app.models import KBChunk,QAHist,db

#============Configuration=======
#
#================================

PDF_FOLDER = Config.PDF_FOLDER
EMBEDDING_MODEL = Config.EMBEDDING_MODEL
MODEL_PATH = Config.MODEL_PATH
PGVECTOR_DIM = Config.PGVECTOR_DIM
MAX_RESULTS = Config.MAX_RESULTS
ALLOWED_EXTENSIONS = Config.ALLOWED_EXTENSIONS




def get_embedder():
    return current_app.embedder

def get_llama():
    return current_app.llama




# ====================
# PDF CLEANING + MERGING
# ====================
def clean_and_merge_lines(lines):
    """
    Keep section headers, keep CLI in fenced blocks, and keep normal text.
    """
    chunks, current = [], []
    section_pattern = re.compile(r"^(Section\s+\d+[:.]|^\d+\.\d+|Step\s+\d+(\.\d+)*|\d+\.)")
    #command_pattern = re.compile(r"^(#|sudo\s+)?(kubectl|kubeadm|helm|cat|curl|docker|systemctl|cp|rm|openssl|base64|grep|awk|sed|tee|chown|chmod|bdconfig|bd_mgmt|erlang|mnesia|/opt/)")
    command_pattern = re.compile(
    r"(?:(?:#|sudo\s+)?(?:kubectl|kubeadm|helm|cat|curl|docker|systemctl|cp|rm|openssl|base64|grep|awk|sed|tee|chown|chmod|bdconfig|bd_mgmt|erlang|mnesia|/opt/).*)",
    re.IGNORECASE,
)
    for line in lines:
        line = line.strip()

        line = unicodedata.normalize("NFKC", line)
        if not line:
            continue


        if section_pattern.match(line) or line.lower().startswith("issue") or line.lower().startswith("cause") or line.lower().startswith("resolution") or line.lower().startswith("environment") or re.match(r'^\s*(##\s*)?(resolution|procedure|steps?|issue|cause|workaround)\b', line.strip(), re.I):
            if current:
                chunks.append("\n".join(current).strip())
                current = []
            # turn any header into markdown header
            if not line.startswith("## "):
                current.append(f"## {line}")
            else:
                current.append(line)
            continue
        #if command_pattern.match(line):
        #    # Split into multiple commands if many appear in one line
        #    commands = re.split(r'\s+(?=(kubectl|kubeadm|helm|cat|curl|docker|systemctl|cp|rm|openssl|base64|grep|awk|sed|tee|chown|chmod|bdconfig|bd_mgmt|erlang|mnesia|/opt/))', line)
        #    for cmd in commands:
        #        cmd = cmd.strip()
        #        if cmd:
        #            current.append(f"```bash\n{cmd}\n```")

        #current.append(line)

        if command_pattern.match(line):
              # Split multiple commands in same line (e.g., 'systemctl start a systemctl start b')
              cmds = re.findall(command_pattern, line)
              for cmd in cmds:
                  cmd_clean = cmd.strip()
                  if cmd_clean:
                        current.append(f"```bash\n{cmd_clean}\n```")
              continue  # prevent adding raw line again


        if line.strip().lower().startswith("note:"):
             current.append(f"> **{line.strip()}**")
             continue


    if current:
        #chunks.append("\n".join(current).strip())
        chunks.append("\n\n".join(current).strip())

    # filter tiny non-code fragments
    final = []

    for c in chunks:
        if "```bash" in c or len(c.strip()) >= 10:
            final.append(c)
    for i, c in enumerate(final):
        c = re.sub(r'(```bash\s*)+\n*', '```bash\n', c)   # collapse duplicate code fence openings
        c = re.sub(r'\n*(```)+', '\n```', c)              # ensure single closing fence
        final[i] = c.strip()

    return final

def normalize_chunk(chunk):
    # Collapse multiple spaces, strip leading/trailing spaces
    chunk = re.sub(r"\s+", " ", chunk.strip())
    return chunk


def clean_pdf_text(full_text):
    """
    Cleans PDF-extracted text:
    1) de-duplicate identical lines (preserve order)
    2) merge wrapped lines (incl. hyphenated breaks)
    3) collapse repeated tokens and repeated n-gram sequences
    4) normalize whitespace/punctuation spacing
    """

    # --- 1) normalize + dedup lines ---
    lines = full_text.splitlines()
    dedup_lines, seen = [], set()
    for line in lines:
        line = unicodedata.normalize("NFKC", line).strip()
        if line and line not in seen:
            dedup_lines.append(line)
            seen.add(line)

    # --- 2) merge wrapped lines (handle hyphen line-breaks) ---
    merged = []
    buf = ""
    for line in dedup_lines:
        if not buf:
            buf = line
            continue

        # If previous buffer looks unfinished, join; fix hyphenation
        if (not re.search(r'[.!?:]$', buf) and len(line) < 120) or buf.endswith('-'):
            if buf.endswith('-'):
                # remove hyphen and don’t insert space (word continuation)
                buf = buf[:-1] + line.lstrip()
            else:
                buf += " " + line
        else:
            merged.append(buf)
            buf = line
    if buf:
        merged.append(buf)

    # --- 3) collapse repeats (tokens & n-grams) ---
    def collapse_repeats(text, max_span=8):
        # split by whitespace; punctuation stays attached (generic)
        tokens = text.split()

        # A) collapse adjacent duplicate tokens quickly (L=1)
        i, out = 0, []
        while i < len(tokens):
            if i+1 < len(tokens) and tokens[i+1] == tokens[i]:
                out.append(tokens[i])
                i += 2
            else:
                out.append(tokens[i])
                i += 1
        tokens = out

        # B) collapse repeated n-grams (handles '-n kubeflow -n kubeflow', etc.)
        changed = True
        while changed:
            changed = False
            # try longer spans first so we prefer collapsing bigger repeats
            for L in range(min(max_span, len(tokens)//2), 1, -1):
                i, out = 0, []
                local_change = False
                while i < len(tokens):
                    if i + 2*L <= len(tokens) and tokens[i:i+L] == tokens[i+L:i+2*L]:
                        # keep one copy of the sequence, skip the duplicate
                        out.extend(tokens[i:i+L])
                        i += 2*L
                        local_change = True
                    else:
                        out.append(tokens[i])
                        i += 1
                if local_change:
                    tokens = out
                    changed = True
                    break  # restart from max span after a change

        return " ".join(tokens)

    cleaned_lines = [collapse_repeats(line) for line in merged]

    # --- 4) normalize spaces & punctuation spacing ---
    text = "\n".join(cleaned_lines)
    text = re.sub(r'\s+([.,:;!?])', r'\1', text)     # no space before punctuation
    text = re.sub(r'[ \t]+', ' ', text)              # collapse runs of spaces
    text = re.sub(r'(?<!:)//+', '/', text)
    return text.strip()


def extract_text_from_pdfs(pdf_folder):
    text_chunks = []
    seen = set()
    for filename in os.listdir(pdf_folder):
        if filename.lower().endswith(".pdf"):
            path = os.path.join(pdf_folder, filename)
            print(f"[INFO] Reading PDF: {filename}")
            try:
                full_text = extract_text_pdfminer(path)
                full_text = clean_pdf_text(full_text)
                lines = full_text.splitlines()
                for chunk in clean_and_merge_lines(lines):
                    chunk_norm = normalize_chunk(chunk)
                    if len(chunk_norm) >= 5 and chunk_norm not in seen:
                        text_chunks.append(chunk_norm)
                        seen.add(chunk_norm)
            except Exception as e:
                print(f"[ERROR] Failed to read {filename}: {e}")
    return [chunk for chunk in text_chunks if len(chunk.strip()) >= 10]

def extract_text_from_pdfs_single(filepath):
    seen = set()
    try:
        full_text = extract_text_pdfminer(filepath)

        full_text = clean_pdf_text(full_text)

        lines = full_text.splitlines()

        deduped_chunks = []
        for chunk in clean_and_merge_lines(lines):

            chunk_norm = normalize_chunk(chunk)
            if len(chunk_norm) >= 5 and chunk_norm not in seen:
                deduped_chunks.append(chunk_norm)
                seen.add(chunk_norm)
        return deduped_chunks
    except Exception as e:
        print(f"[ERROR] Failed to read {filepath}: {e}")

    print(f"Raw chunks from {fname}:", raw_chunks)
    print(f"Overlapped chunks from {fname}:", all_chunks)

    return [chunk for chunk in deduped_chunks if len(chunk.strip()) >= 10]

# ====================
# CHUNK SPLITTING + OVERLAP
# ====================
def split_text(text, chunk_size=900):
    lines = text.strip().split('\n')
    chunks, buf, in_code = [], "", False
    for line in lines:
        if line.strip().startswith("```"):
            in_code = not in_code
        if line.startswith("## ") and not in_code:
            if buf.strip():
                chunks.append(buf.strip())
            buf = line + "\n"
        elif len(buf) + len(line) < chunk_size or in_code:
            buf += line + "\n"
        else:
            if buf.strip():
                chunks.append(buf.strip())
            buf = line + "\n"
    if buf.strip():
        chunks.append(buf.strip())

    return [c for c in chunks if len(c.strip()) >= 10]

def overlapping_chunks(chunks, overlap=2):
    out = []
    for i in range(len(chunks)):
        combo = "\n\n".join(chunks[max(0, i-overlap):i+1])
        out.append(combo)
    return out

# ====================
# EMBEDDINGS
# ====================
def load_embeddings_to_pg(chunks, source_file):
    embedder = current_app.embedder
    print("Embedding and saving chunks to PostgreSQL")
    for i, chunk in enumerate(chunks):
        vec = embedder.encode(chunk).tolist()
        db.session.add(KBChunk(text=chunk, embedding=vec, chunk_id=f"{source_file}_chunk_{i}",source_file=source_file))

    print(f"[DEBUG] Inserted {len(chunks)} chunks into DB")

    db.session.commit()

@lru_cache(maxsize=2000)
def cached_embedding(text):
    embedder = current_app.embedder
    if embedder is None:
        raise RuntimeError("Embedder not loaded. Call load_models() first.")
    return embedder.encode(text).tolist()


# ====================
# RETRIEVAL
# ====================
def detect_doc_style(chunks):
    """
    Heuristically detect doc style:
    - 'numbered' if section headers with ## <number> exist
    - 'kb_style' if headings like 'Issue', 'Cause', 'Resolution', 'Workaround' dominate
    """
    numbered_count = sum(bool(re.match(r"^##\s*\d", c.strip())) for c in chunks)
    kb_headings_count = sum(bool(re.match(r"^(issue|cause|resolution|workaround)", c.strip(), re.I)) for c in chunks)

    if numbered_count >= kb_headings_count and numbered_count > 0:
        return "numbered"
    elif kb_headings_count > 0:
        return "kb_style"
    return "mixed"

def retrieve_relevant_chunks_pg(query, top_k=None):
    top_k = top_k or MAX_RESULTS
    embedder = current_app.embedder
    query_vec = embedder.encode(query).tolist()



    # Step 1: Initial retrieval
    sql = sa_text("""
        SELECT text, source_file, embedding <-> CAST(:query_vec AS vector) AS dist
        FROM kb_chunks
        ORDER BY dist
        LIMIT :top_k
    """)
    results = db.session.execute(sql, {
        "query_vec": query_vec,
        "top_k": top_k
    }).fetchall()

    if not results:
        return []

    chunks = [r[0] for r in results]
    source_file = results[0][1]

    # Step 2: Retrieve more from same source file
    sql2 = sa_text("""
        SELECT text, embedding <-> CAST(:query_vec AS vector) AS dist
        FROM kb_chunks
        WHERE source_file = :src
        ORDER BY dist
        LIMIT :top_k
    """)
    results2 = db.session.execute(sql2, {
        "src": source_file,
        "query_vec": query_vec,
        "top_k": top_k * 2
    }).fetchall()

    chunks = [r[0] for r in results2]


    # Ensure all '## Resolution' chunks from the same source file are included if query asks for resolution

    extra_res_chunks = db.session.execute(sa_text("""
            SELECT text
            FROM kb_chunks
            WHERE source_file = :src
              AND lower(text) ILIKE '%resolution|cause%'
        """), {"src": source_file}).fetchall()
    for (extra_chunk,) in extra_res_chunks:
        if extra_chunk not in chunks:
            chunks.append(extra_chunk)


    # Step 3: Auto-detect doc style
    doc_style = detect_doc_style(chunks)

    if doc_style == "numbered":
        # Merge CLI-only chunks with preceding explanation
        merged_chunks = []
        buffer = ""
        for chunk in chunks:
            if chunk.startswith("```bash") and buffer:
                buffer += "\n" + chunk
            else:
                if buffer:
                    merged_chunks.append(buffer.strip())
                buffer = chunk
        if buffer:
            merged_chunks.append(buffer.strip())

        # Deduplicate by section number
        seen_sections = set()
        deduped_chunks = []
        for c in merged_chunks:
            m = re.match(r"^##\s*([\d\.]+)", c.strip())
            section_id = m.group(1) if m else None
            if section_id and section_id in seen_sections:
                continue
            if section_id:
                seen_sections.add(section_id)
            deduped_chunks.append((section_id, c))

        # Sort by section number & CLI density
        def parse_section_id(sid):
            if not sid: return (9999,)
            return tuple(int(x) if x.isdigit() else 9999 for x in sid.split('.'))

        def cli_score(text):
            cli_keywords = ['kubeadm', 'kubectl', 'openssl', 'docker', 'systemctl', 'bd_mgmt','erlang','rm', 'cp', 'bdconfig', 'base64', 'watch']
            return sum(k in text.lower() for k in cli_keywords) + text.count("```bash")

        deduped_chunks.sort(key=lambda x: (parse_section_id(x[0]), -cli_score(x[1])))
        final_chunks = [c for _, c in deduped_chunks]

    elif doc_style == "kb_style":
        # Keep chunks in original distance order, but group by KB headings
        headings = ["Issue", "Environment", "Cause", "Resolution", "Workaround"]
        final_chunks = []
        for h in headings:
            for c in chunks:
                if c.strip().lower().startswith(h.lower()):
                    final_chunks.append(c)
        # Append any other highly relevant chunks at the end
        remaining = [c for c in chunks if c not in final_chunks]
        final_chunks.extend(remaining)

    else:
        # Fallback: Just return distance-ranked chunks
        final_chunks = chunks

    # Extra boost for queries mentioning resolution/cause/issue
    boost_keywords = ["resolution", "cause", "issue"]

    if any(kw in query.lower() for kw in boost_keywords):
        # Match headings like "## Resolution", "  ## cause", "### issue", "* Cause", etc.
        pattern = re.compile(r"^\W*\s*(?:" + "|".join(boost_keywords) + r")\b", re.IGNORECASE)

        boosted = [c for c in final_chunks if pattern.match(c.strip())]
        others = [c for c in final_chunks if c not in boosted]

        final_chunks = boosted + others

    print(f"[DEBUG] Style detected: {doc_style}, returning {len(final_chunks)} chunks from {source_file}")



    return final_chunks

def clean_context(text):
    # 1. Remove exact duplicate lines
    seen, out = set(), []
    for line in text.splitlines():
        line = line.strip()
        if line and line not in seen:
            out.append(line)
            seen.add(line)
    text = "\n".join(out)

    # 2. Fix broken bash fences (open without closing)
    text = re.sub(r'(```bash\s*)(?![\s\S]*?```)', r'\1\n```', text)
    # 3. Remove partial fenced lines like "```bash systemctl ```"
    text = re.sub(r'```bash\s*systemctl\s*```', '```bash\nsystemctl\n```', text)
    # 4. Collapse consecutive identical command blocks
    text = re.sub(r'(```bash[\s\S]*?```)(\s*\1)+', r'\1', text)
    return text.strip()




# ====================
# LLM FALLBACK (only if extractive fails)
# ====================
def llm_answer(context_chunks, question):

    llama = current_app.llama
    #context = "".join(context_chunks)
    context = clean_context("".join(context_chunks))
    print("Question:", question)
    print("Context retrieved:", context)

    prompt = f"""
You are a Kubernetes platform assistant for HPE Ezmeral.

Use only CLI commands and instructions found **exactly** in the context below.
Ensure numbered steps are consistent and not appended multiple times.
Make sure each bash block starts and ends properly.
Ensure the output do not have duplicated instructions.
Do not invent commands that are not in the context.
Provide a clear, **numbered** list of steps in order.
Each step must be on a new line and contain any commands inside fenced ```bash code blocks.

### Context:
{context}

### Question:
{question}

---
Numbered Steps:
"""


    out = llama(
        prompt=prompt,
        max_tokens=2048,
        stop=["<END>"]
    )


    #pdb.set_trace()

    return out["choices"][0]["text"].strip()
