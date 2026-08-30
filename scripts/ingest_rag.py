from __future__ import annotations

import hashlib
import io
import json
import re
import sys
from pathlib import Path
import pymupdf
import camelot
import easyocr
import numpy as np
import torch

from PIL import Image

from langchain_core.documents import Document

from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
)


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAG_DIRECTORY = PROJECT_ROOT / "rag_data"

# Manifest stores information about already indexed PDFs.
MANIFEST_FILE = PROJECT_ROOT / "rag_manifest.json"

sys.path.insert(0, str(PROJECT_ROOT))

from backend.rag import (
    add_documents,
    get_vectorstore,
)


# ============================================================
# CONFIGURATION
# ============================================================

SUPPORTED_EXTENSIONS = {
    ".pdf",
}

EMBEDDING_BATCH_SIZE = 64

BLIP_MODEL_NAME = (
    "Salesforce/blip-image-captioning-base"
)

BLIP_MAX_IMAGE_SIZE = 768


# ============================================================
# BLIP IMAGE CAPTION MODEL
# ============================================================

print()
print("=" * 70)
print("LOADING BLIP IMAGE CAPTION MODEL")
print("=" * 70)

try:

    processor = BlipProcessor.from_pretrained(
        BLIP_MODEL_NAME
    )

    caption_model = (
        BlipForConditionalGeneration.from_pretrained(
            BLIP_MODEL_NAME
        )
    )

    caption_model = caption_model.to("cpu")

    caption_model.eval()

    print(
        "BLIP loaded successfully."
    )

except Exception as exc:

    print(
        "WARNING: BLIP model could not be loaded."
    )

    print(
        f"Reason: {exc}"
    )

    print(
        "Image captioning will be disabled."
    )

    processor = None
    caption_model = None


# ============================================================
# EASYOCR
# ============================================================

print()
print("=" * 70)
print("LOADING EASYOCR")
print("=" * 70)

try:

    ocr_reader = easyocr.Reader(
        ["en"],
        gpu=False,
    )

    print(
        "EasyOCR loaded successfully."
    )

except Exception as exc:

    print(
        "WARNING: EasyOCR could not be loaded."
    )

    print(
        f"Reason: {exc}"
    )

    print(
        "OCR will be disabled."
    )

    ocr_reader = None


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(
    text: str,
) -> str:

    if not text:
        return ""

    text = text.replace(
        "\x00",
        " ",
    )

    text = text.replace(
        "\r\n",
        "\n",
    )

    text = text.replace(
        "\r",
        "\n",
    )

    text = re.sub(
        r"[ \t]+",
        " ",
        text,
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )

    return text.strip()


# ============================================================
# FILE HASH
# ============================================================

def calculate_file_hash(
    file_path: Path,
) -> str:
    """
    Calculate SHA-256 hash of the complete PDF file.

    This is used to determine whether a PDF is:
        - NEW
        - UNCHANGED
        - MODIFIED
    """

    sha256 = hashlib.sha256()

    with file_path.open(
        "rb"
    ) as file:

        while True:

            data = file.read(
                1024 * 1024
            )

            if not data:
                break

            sha256.update(
                data
            )

    return sha256.hexdigest()


# ============================================================
# MANIFEST
# ============================================================

def load_manifest() -> dict:
    """
    Load the ingestion manifest.

    If no manifest exists, return an empty dictionary.
    """

    if not MANIFEST_FILE.exists():

        return {}

    try:

        with MANIFEST_FILE.open(
            "r",
            encoding="utf-8",
        ) as file:

            data = json.load(
                file
            )

        if isinstance(
            data,
            dict,
        ):

            return data

    except Exception as exc:

        print()
        print(
            "WARNING: Could not load "
            "rag_manifest.json"
        )

        print(
            f"Reason: {exc}"
        )

    return {}


def save_manifest(
    manifest: dict,
) -> None:
    """
    Save the ingestion manifest.
    """

    try:

        with MANIFEST_FILE.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                manifest,
                file,
                indent=4,
            )

    except Exception as exc:

        print()
        print(
            "WARNING: Could not save "
            "rag_manifest.json"
        )

        print(
            f"Reason: {exc}"
        )


# ============================================================
# DETECT PDF CHANGES
# ============================================================

def get_pdf_changes(
    pdf_files: list[Path],
    manifest: dict,
) -> tuple[
    list[Path],
    list[Path],
    list[Path],
]:
    """
    Compare all current PDFs with the manifest.

    Returns:

        new_files
        changed_files
        unchanged_files
    """

    new_files = []

    changed_files = []

    unchanged_files = []

    print()
    print("=" * 70)
    print("CHECKING DOCUMENT CHANGES")
    print("=" * 70)

    for pdf_path in pdf_files:

        filename = pdf_path.name

        print(
            f"Checking: {filename}"
        )

        try:

            current_hash = calculate_file_hash(
                pdf_path
            )

        except Exception as exc:

            print(
                f"  -> Could not calculate hash: "
                f"{exc}"
            )

            continue

        previous_info = manifest.get(
            filename
        )

        # ----------------------------------------------------
        # NEW PDF
        # ----------------------------------------------------

        if previous_info is None:

            print(
                "  -> NEW"
            )

            new_files.append(
                pdf_path
            )

            continue

        previous_hash = previous_info.get(
            "sha256"
        )

        # ----------------------------------------------------
        # MODIFIED PDF
        # ----------------------------------------------------

        if previous_hash != current_hash:

            print(
                "  -> MODIFIED"
            )

            changed_files.append(
                pdf_path
            )

            continue

        # ----------------------------------------------------
        # UNCHANGED PDF
        # ----------------------------------------------------

        print(
            "  -> UNCHANGED"
        )

        unchanged_files.append(
            pdf_path
        )

    return (
        new_files,
        changed_files,
        unchanged_files,
    )


# ============================================================
# DELETE OLD PDF CHUNKS FROM CHROMADB
# ============================================================

def delete_pdf_from_chromadb(
    pdf_name: str,
) -> None:
    """
    Delete all existing ChromaDB chunks belonging to
    the specified PDF.

    This is used ONLY when a PDF has been modified.
    """

    print()
    print(
        f"Removing old ChromaDB data "
        f"for: {pdf_name}"
    )

    try:

        vectorstore = get_vectorstore()

        collection = vectorstore._collection

        existing = collection.get(
            where={
                "source": pdf_name
            }
        )

        ids = existing.get(
            "ids",
            [],
        )

        if not ids:

            print(
                "No existing chunks found."
            )

            return

        collection.delete(
            ids=ids
        )

        print(
            f"Deleted {len(ids)} old chunks."
        )

    except Exception as exc:

        print()
        print(
            f"ERROR deleting old ChromaDB "
            f"data for {pdf_name}:"
        )

        print(
            str(exc)
        )

        raise


# ============================================================
# BLIP IMAGE CAPTIONING
# ============================================================

def generate_image_caption(
    image: Image.Image,
) -> str:

    if (
        processor is None
        or caption_model is None
    ):
        return ""

    try:

        # Convert to RGB
        if image.mode != "RGB":

            image = image.convert(
                "RGB"
            )

        # Resize large images
        image.thumbnail(
            (
                BLIP_MAX_IMAGE_SIZE,
                BLIP_MAX_IMAGE_SIZE,
            )
        )

        # Prepare BLIP input
        inputs = processor(
            images=image,
            return_tensors="pt",
        )

        # CPU inference
        with torch.no_grad():

            output = caption_model.generate(
                **inputs,
                max_new_tokens=30,
                num_beams=1,
                do_sample=False,
            )

        # Decode result
        caption = processor.decode(
            output[0],
            skip_special_tokens=True,
        )

        return clean_text(
            caption
        )

    except Exception as exc:

        print(
            f"BLIP captioning failed: {exc}"
        )

        return ""


# ============================================================
# EASYOCR
# ============================================================

def extract_ocr_text(
    image: Image.Image,
) -> str:

    if ocr_reader is None:

        return ""

    try:

        # Make sure image is RGB
        if image.mode != "RGB":

            image = image.convert(
                "RGB"
            )

        # ----------------------------------------------------
        # EasyOCR expects a NumPy array
        # ----------------------------------------------------

        image_array = np.array(
            image
        )

        # Run OCR
        results = ocr_reader.readtext(
            image_array,
            detail=0,
            paragraph=True,
        )

        if not results:

            return ""

        text = " ".join(
            str(item)
            for item in results
        )

        return clean_text(
            text
        )

    except Exception as exc:

        print(
            f"EasyOCR failed: {exc}"
        )

        return ""


# ============================================================
# TEXT EXTRACTION
# ============================================================

def extract_text_documents(
    pdf_path: Path,
    pdf: pymupdf.Document,
) -> list[Document]:

    documents: list[Document] = []

    for page_number in range(
        len(pdf)
    ):

        try:

            page = pdf[
                page_number
            ]

            text = clean_text(
                page.get_text()
            )

            if not text:

                continue

            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": pdf_path.name,
                        "document_name": pdf_path.stem,
                        "page": page_number + 1,
                        "content_type": "text",
                    },
                )
            )

        except Exception as exc:

            print(
                f"Text extraction failed "
                f"on page "
                f"{page_number + 1}: "
                f"{exc}"
            )

    return documents


# ============================================================
# TABLE EXTRACTION
# ============================================================

def extract_table_documents(
    pdf_path: Path,
) -> list[Document]:

    documents: list[Document] = []

    try:

        tables = camelot.read_pdf(
            str(pdf_path),
            pages="all",
            flavor="stream",
        )

        print(
            f"Tables detected: "
            f"{len(tables)}"
        )

        for index, table in enumerate(
            tables
        ):

            try:

                table_text = (
                    table.df.to_string(
                        index=False
                    )
                )

                table_text = clean_text(
                    table_text
                )

                if not table_text:

                    continue

                documents.append(
                    Document(
                        page_content=(
                            "TABLE CONTENT\n\n"
                            + table_text
                        ),
                        metadata={
                            "source": pdf_path.name,
                            "document_name": pdf_path.stem,
                            "content_type": "table",
                            "table_index": index,
                        },
                    )
                )

            except Exception as exc:

                print(
                    f"Table {index} "
                    f"processing failed: "
                    f"{exc}"
                )

    except Exception as exc:

        print(
            f"Table extraction failed: "
            f"{exc}"
        )

    return documents


# ============================================================
# IMAGE EXTRACTION
# ============================================================

def extract_image_documents(
    pdf_path: Path,
    pdf: pymupdf.Document,
) -> list[Document]:
    """
    Extract meaningful images actually placed on PDF pages.

    Pipeline:

        PDF page
          -> visible image blocks
          -> size filtering
          -> duplicate filtering
          -> EasyOCR
          -> BLIP when useful
          -> multimodal Document
    """

    documents: list[Document] = []

    # --------------------------------------------------------
    # IMAGE FILTERS
    # --------------------------------------------------------

    MIN_IMAGE_WIDTH = 150

    MIN_IMAGE_HEIGHT = 100

    MIN_IMAGE_AREA = 30_000

    MIN_DISPLAY_WIDTH = 80.0

    MIN_DISPLAY_HEIGHT = 60.0

    MIN_DISPLAY_AREA = 5_000.0

    BLIP_IF_OCR_SHORTER_THAN = 80

    processed_images = 0

    skipped_small = 0

    skipped_duplicate = 0

    failed_images = 0

    total_visible_blocks = 0

    # --------------------------------------------------------
    # IMAGE DUPLICATE TRACKING
    # --------------------------------------------------------

    seen_fingerprints: set[str] = set()

    # --------------------------------------------------------
    # PROCESS EACH PAGE
    # --------------------------------------------------------

    for page_number in range(
        len(pdf)
    ):

        try:

            page = pdf[
                page_number
            ]

            # ------------------------------------------------
            # Use page layout dictionary
            # ------------------------------------------------

            page_dict = page.get_text(
                "dict"
            )

            image_blocks = [
                block
                for block in page_dict.get(
                    "blocks",
                    [],
                )
                if block.get(
                    "type"
                ) == 1
            ]

            total_visible_blocks += len(
                image_blocks
            )

        except Exception as exc:

            print(
                f"Could not inspect image blocks "
                f"on page {page_number + 1}: "
                f"{exc}"
            )

            continue

        if not image_blocks:

            continue

        print(
            f"Page {page_number + 1}: "
            f"{len(image_blocks)} visible "
            f"image blocks found."
        )

        # ----------------------------------------------------
        # PROCESS IMAGES
        # ----------------------------------------------------

        for image_index, block in enumerate(
            image_blocks,
            1,
        ):

            try:

                width = int(
                    block.get(
                        "width",
                        0,
                    )
                    or 0
                )

                height = int(
                    block.get(
                        "height",
                        0,
                    )
                    or 0
                )

                bbox = block.get(
                    "bbox",
                    (
                        0,
                        0,
                        0,
                        0,
                    ),
                )

                # ------------------------------------------------
                # DISPLAY SIZE
                # ------------------------------------------------

                if len(bbox) == 4:

                    x0, y0, x1, y1 = bbox

                    display_width = abs(
                        float(x1) - float(x0)
                    )

                    display_height = abs(
                        float(y1) - float(y0)
                    )

                else:

                    display_width = 0.0

                    display_height = 0.0

                display_area = (
                    display_width
                    * display_height
                )

                # ------------------------------------------------
                # SMALL IMAGE FILTER
                # ------------------------------------------------

                if (
                    width
                    < MIN_IMAGE_WIDTH
                    or height
                    < MIN_IMAGE_HEIGHT
                    or width * height
                    < MIN_IMAGE_AREA
                    or display_width
                    < MIN_DISPLAY_WIDTH
                    or display_height
                    < MIN_DISPLAY_HEIGHT
                    or display_area
                    < MIN_DISPLAY_AREA
                ):

                    skipped_small += 1

                    print(
                        f"Skipping small image "
                        f"{image_index} "
                        f"on page "
                        f"{page_number + 1}: "
                        f"embedded={width}x{height}, "
                        f"displayed="
                        f"{display_width:.0f}x"
                        f"{display_height:.0f}"
                    )

                    continue

                # ------------------------------------------------
                # IMAGE BYTES
                # ------------------------------------------------

                image_bytes = block.get(
                    "image"
                )

                if not image_bytes:

                    xref = block.get(
                        "xref"
                    )

                    if xref:

                        base_image = (
                            pdf.extract_image(
                                xref
                            )
                        )

                        image_bytes = (
                            base_image.get(
                                "image"
                            )
                        )

                if not image_bytes:

                    print(
                        f"No image bytes available "
                        f"for image {image_index} "
                        f"on page "
                        f"{page_number + 1}."
                    )

                    continue

                # ------------------------------------------------
                # IMAGE FINGERPRINT
                # ------------------------------------------------

                fingerprint = hashlib.sha1(
                    image_bytes
                ).hexdigest()

                if fingerprint in seen_fingerprints:

                    skipped_duplicate += 1

                    print(
                        f"Skipping duplicate image "
                        f"{image_index} "
                        f"on page "
                        f"{page_number + 1}."
                    )

                    continue

                seen_fingerprints.add(
                    fingerprint
                )

                processed_images += 1

                print(
                    f"Processing image "
                    f"{processed_images} "
                    f"on page "
                    f"{page_number + 1} "
                    f"[{width}x{height}]..."
                )

                # ------------------------------------------------
                # PIL IMAGE
                # ------------------------------------------------

                image = Image.open(
                    io.BytesIO(
                        image_bytes
                    )
                ).convert(
                    "RGB"
                )

                image.load()

                # ------------------------------------------------
                # OCR FIRST
                # ------------------------------------------------

                ocr_text = extract_ocr_text(
                    image.copy()
                )

                # ------------------------------------------------
                # BLIP ONLY WHEN NEEDED
                # ------------------------------------------------

                caption = ""

                if len(
                    ocr_text.strip()
                ) < BLIP_IF_OCR_SHORTER_THAN:

                    caption = (
                        generate_image_caption(
                            image.copy()
                        )
                    )

                # ------------------------------------------------
                # COMBINE CONTENT
                # ------------------------------------------------

                combined_parts = [
                    "IMAGE CONTENT",
                    f"PDF: {pdf_path.name}",
                    f"Page: {page_number + 1}",
                    (
                        f"Image dimensions: "
                        f"{width}x{height}"
                    ),
                ]

                if caption:

                    combined_parts.append(
                        "IMAGE DESCRIPTION\n\n"
                        + caption
                    )

                if ocr_text:

                    combined_parts.append(
                        "OCR TEXT\n\n"
                        + ocr_text
                    )

                # ------------------------------------------------
                # NOTHING USEFUL
                # ------------------------------------------------

                if (
                    not caption
                    and not ocr_text
                ):

                    print(
                        f"No useful information "
                        f"extracted from image "
                        f"{image_index}; "
                        f"skipping."
                    )

                    continue

                combined_content = clean_text(
                    "\n\n".join(
                        combined_parts
                    )
                )

                if not combined_content:

                    continue

                # ------------------------------------------------
                # DOCUMENT
                # ------------------------------------------------

                documents.append(
                    Document(
                        page_content=combined_content,
                        metadata={
                            "source": pdf_path.name,
                            "document_name": pdf_path.stem,
                            "page": page_number + 1,
                            "content_type": "image",
                            "image_index": processed_images,
                            "image_width": width,
                            "image_height": height,
                            "display_width": round(
                                display_width,
                                2,
                            ),
                            "display_height": round(
                                display_height,
                                2,
                            ),
                            "image_fingerprint": fingerprint,
                        },
                    )
                )

                print(
                    "Image processed successfully."
                )

            except Exception as exc:

                failed_images += 1

                print(
                    f"Image processing failed "
                    f"for page "
                    f"{page_number + 1}, "
                    f"image {image_index}: "
                    f"{exc}"
                )

                continue

    # --------------------------------------------------------
    # IMAGE STATISTICS
    # --------------------------------------------------------

    print()

    print(
        f"Visible image blocks found : "
        f"{total_visible_blocks}"
    )

    print(
        f"Meaningful images processed: "
        f"{processed_images}"
    )

    print(
        f"Small images skipped       : "
        f"{skipped_small}"
    )

    print(
        f"Duplicate images skipped   : "
        f"{skipped_duplicate}"
    )

    print(
        f"Image failures             : "
        f"{failed_images}"
    )

    return documents


# ============================================================
# LOAD SINGLE PDF
# ============================================================

def load_pdf(
    pdf_path: Path,
) -> list[Document]:

    print()
    print("=" * 70)
    print(
        f"PROCESSING: "
        f"{pdf_path.name}"
    )
    print("=" * 70)

    try:

        pdf = pymupdf.open(
            str(pdf_path)
        )

    except Exception as exc:

        print(
            f"Could not open PDF "
            f"{pdf_path.name}: "
            f"{exc}"
        )

        return []

    try:

        # ----------------------------------------------------
        # TEXT
        # ----------------------------------------------------

        text_docs = (
            extract_text_documents(
                pdf_path,
                pdf,
            )
        )

        # ----------------------------------------------------
        # IMAGES
        # ----------------------------------------------------

        image_docs = (
            extract_image_documents(
                pdf_path,
                pdf,
            )
        )

        # ----------------------------------------------------
        # TABLES
        # ----------------------------------------------------

        table_docs = (
            extract_table_documents(
                pdf_path
            )
        )

    finally:

        pdf.close()

    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    print()

    print(
        f"Text documents  : "
        f"{len(text_docs)}"
    )

    print(
        f"Image documents : "
        f"{len(image_docs)}"
    )

    print(
        f"Table documents : "
        f"{len(table_docs)}"
    )

    return (
        text_docs
        + image_docs
        + table_docs
    )


# ============================================================
# STATISTICS
# ============================================================

def display_statistics(
    documents: list[Document],
) -> None:

    print()
    print("=" * 70)
    print(
        "DOCUMENT STATISTICS"
    )
    print("=" * 70)

    print(
        f"Total documents: "
        f"{len(documents)}"
    )

    content_types: dict[
        str,
        int,
    ] = {}

    for doc in documents:

        content_type = (
            doc.metadata.get(
                "content_type",
                "unknown",
            )
        )

        content_types[
            content_type
        ] = (
            content_types.get(
                content_type,
                0,
            )
            + 1
        )

    print()

    for key, value in (
        content_types.items()
    ):

        print(
            f"{key}: {value}"
        )


# ============================================================
# CHROMADB INGESTION
# ============================================================

def ingest_documents(
    documents: list[Document],
) -> None:

    print()
    print("=" * 70)
    print(
        "INDEXING INTO CHROMADB"
    )
    print("=" * 70)

    if not documents:

        print(
            "No documents available "
            "for indexing."
        )

        return

    try:

        total_chunks = add_documents(
            documents,
            batch_size=EMBEDDING_BATCH_SIZE,
        )

        print()

        print(
            f"Chunks indexed: "
            f"{total_chunks}"
        )

    except Exception as exc:

        print()
        print("=" * 70)
        print(
            "CHROMADB INGESTION FAILED"
        )
        print("=" * 70)

        print(
            f"Error: {exc}"
        )

        raise


# ============================================================
# INCREMENTAL PDF PROCESSING
# ============================================================

def load_incremental_pdfs(
    pdf_files: list[Path],
) -> tuple[
    list[Document],
    dict,
]:
    """
    Process only NEW and MODIFIED PDFs.

    UNCHANGED PDFs are skipped.

    Returns:

        documents
        processing information
    """

    manifest = load_manifest()

    (
        new_files,
        changed_files,
        unchanged_files,
    ) = get_pdf_changes(
        pdf_files,
        manifest,
    )

    # --------------------------------------------------------
    # INGESTION PLAN
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "INGESTION PLAN"
    )
    print("=" * 70)

    print(
        f"New PDFs     : "
        f"{len(new_files)}"
    )

    print(
        f"Changed PDFs : "
        f"{len(changed_files)}"
    )

    print(
        f"Unchanged PDFs: "
        f"{len(unchanged_files)}"
    )

    documents: list[Document] = []

    successfully_processed: list[Path] = []

    # ========================================================
    # NEW PDFs
    # ========================================================

    for pdf_path in new_files:

        print()
        print("=" * 70)
        print(
            f"NEW PDF: "
            f"{pdf_path.name}"
        )
        print("=" * 70)

        try:

            docs = load_pdf(
                pdf_path
            )

            if docs:

                documents.extend(
                    docs
                )

                successfully_processed.append(
                    pdf_path
                )

            else:

                print(
                    f"No documents extracted "
                    f"from {pdf_path.name}."
                )

        except Exception as exc:

            print()
            print(
                f"ERROR processing "
                f"{pdf_path.name}: "
                f"{exc}"
            )

            continue

    # ========================================================
    # MODIFIED PDFs
    # ========================================================

    for pdf_path in changed_files:

        print()
        print("=" * 70)
        print(
            f"MODIFIED PDF: "
            f"{pdf_path.name}"
        )
        print("=" * 70)

        try:

            # ------------------------------------------------
            # DELETE OLD CHUNKS
            # ------------------------------------------------

            delete_pdf_from_chromadb(
                pdf_path.name
            )

            # ------------------------------------------------
            # PROCESS NEW VERSION
            # ------------------------------------------------

            docs = load_pdf(
                pdf_path
            )

            if docs:

                documents.extend(
                    docs
                )

                successfully_processed.append(
                    pdf_path
                )

            else:

                print(
                    f"No documents extracted "
                    f"from modified PDF "
                    f"{pdf_path.name}."
                )

        except Exception as exc:

            print()
            print(
                f"ERROR processing modified "
                f"PDF {pdf_path.name}: "
                f"{exc}"
            )

            raise

    # ========================================================
    # UPDATE MANIFEST
    # ========================================================

    for pdf_path in successfully_processed:

        try:

            file_hash = calculate_file_hash(
                pdf_path
            )

            manifest[
                pdf_path.name
            ] = {
                "sha256": file_hash,
                "size": pdf_path.stat().st_size,
                "modified_time": pdf_path.stat().st_mtime,
            }

        except Exception as exc:

            print(
                f"WARNING: Could not update "
                f"manifest for "
                f"{pdf_path.name}: "
                f"{exc}"
            )

    # ========================================================
    # HANDLE REMOVED PDFs
    # ========================================================

    current_pdf_names = {
        pdf.name
        for pdf in pdf_files
    }

    removed_files = []

    for filename in list(
        manifest.keys()
    ):

        if filename not in current_pdf_names:

            removed_files.append(
                filename
            )

    for filename in removed_files:

        print()
        print(
            f"WARNING: PDF no longer exists "
            f"in rag_data: {filename}"
        )

        # ----------------------------------------------------
        # IMPORTANT
        # ----------------------------------------------------
        #
        # We remove the manifest entry but DO NOT automatically
        # delete the ChromaDB chunks.
        #
        # This prevents accidental deletion if a PDF was only
        # temporarily moved from rag_data.
        #
        # ----------------------------------------------------

        del manifest[
            filename
        ]

    # ========================================================
    # SAVE MANIFEST
    # ========================================================

    save_manifest(
        manifest
    )

    return (
        documents,
        {
            "new": new_files,
            "changed": changed_files,
            "unchanged": unchanged_files,
            "processed": successfully_processed,
        },
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print()
    print("=" * 70)
    print(
        "MULTIMODAL INCREMENTAL RAG INGESTION"
    )
    print("=" * 70)

    print(
        f"Python executable: "
        f"{sys.executable}"
    )

    print(
        f"Project root: "
        f"{PROJECT_ROOT}"
    )

    print(
        f"RAG directory: "
        f"{RAG_DIRECTORY}"
    )

    print(
        f"Manifest: "
        f"{MANIFEST_FILE}"
    )

    # ========================================================
    # CHECK RAG DIRECTORY
    # ========================================================

    if not RAG_DIRECTORY.exists():

        print()
        print(
            "ERROR: RAG directory "
            "not found:"
        )

        print(
            RAG_DIRECTORY
        )

        return

    # ========================================================
    # FIND PDFs
    # ========================================================

    pdf_files = sorted(
        [
            pdf
            for pdf in RAG_DIRECTORY.iterdir()
            if (
                pdf.is_file()
                and pdf.suffix.lower()
                in SUPPORTED_EXTENSIONS
            )
        ]
    )

    if not pdf_files:

        print()
        print(
            "ERROR: No PDF files found."
        )

        print()

        print(
            "Put your PDF files inside:"
        )

        print(
            RAG_DIRECTORY
        )

        return

    print()

    print(
        f"PDF files found: "
        f"{len(pdf_files)}"
    )

    # ========================================================
    # INCREMENTAL PROCESSING
    # ========================================================

    (
        documents,
        change_info,
    ) = load_incremental_pdfs(
        pdf_files
    )

    # ========================================================
    # NOTHING TO PROCESS
    # ========================================================

    if not documents:

        print()
        print("=" * 70)
        print(
            "NO NEW OR MODIFIED PDFs"
        )
        print("=" * 70)

        print(
            "All PDFs are already indexed."
        )

        print()

        print(
            "ChromaDB was not modified."
        )

        print()

        print(
            "Manifest:"
        )

        print(
            MANIFEST_FILE
        )

        return

    # ========================================================
    # STATISTICS
    # ========================================================

    display_statistics(
        documents
    )

    # ========================================================
    # CHROMADB
    # ========================================================

    ingest_documents(
        documents
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print(
        "INCREMENTAL INGESTION COMPLETE"
    )
    print("=" * 70)

    print(
        f"New PDFs processed     : "
        f"{len(change_info['new'])}"
    )

    print(
        f"Modified PDFs processed: "
        f"{len(change_info['changed'])}"
    )

    print(
        f"Unchanged PDFs skipped : "
        f"{len(change_info['unchanged'])}"
    )

    print(
        f"Documents generated    : "
        f"{len(documents)}"
    )

    print()

    print(
        "Manifest saved to:"
    )

    print(
        MANIFEST_FILE
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()