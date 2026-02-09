import os
from app.utils import (
    extract_text_from_pdfs_single, split_text,
    load_embeddings_to_pg, overlapping_chunks
)
from app.models import KBChunk, db
from app.config import Config
import pdb



def populatedb():
    """Populate the database with embeddings from PDFs in the configured folder."""
    #if KBChunk.query.first() is None:
    print("Populating database from PDFs")
    print("PDF Folder:", Config.PDF_FOLDER)
    print("Files inside:", os.listdir(Config.PDF_FOLDER))


    for filename in os.listdir(Config.PDF_FOLDER):
        if filename.lower().endswith(".pdf"):
            already_in_db = db.session.query(KBChunk).filter_by(source_file=filename).first()

            if not already_in_db:
                path = os.path.join(Config.PDF_FOLDER, filename)
                #"EXtracting Text from File"
                raw_chunks = extract_text_from_pdfs_single(path)
                all_chunks = []
                for doc in raw_chunks:
                    ###split Text and overlapping chunks
                    split_chunks = split_text(doc)
                    overlapped = overlapping_chunks(split_chunks, overlap=1)
                    all_chunks.extend(overlapped)
                if all_chunks:
                    ###"Database populated chunks
                    load_embeddings_to_pg(all_chunks, filename)

    print("[OK] Database populated.")
