"""
AWS Assistant - Incremental Qdrant RAG Ingestion

Features:
- PDF text extraction with PyMuPDF
- Table extraction with Camelot
- Image extraction
- EasyOCR
- BLIP image captioning
- Image deduplication
- SHA256-based incremental ingestion
- Existing manifest compatibility
- Qdrant Cloud vector storage
- New PDF ingestion
- Modified PDF replacement
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import camelot
import easyocr
import numpy as np
import pymupdf
import torch

from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor
from langchain_core.documents import Document


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAG_DIRECTORY = PROJECT_ROOT / "rag_data"
MANIFEST_FILE = PROJECT_ROOT / "rag_manifest.json"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# QDRANT RAG
# ============================================================

from backend.rag import (
    add_documents,
    delete_documents_by_source,
)


# ============================================================
# CONFIGURATION
# ============================================================

SUPPORTED_EXTENSIONS = {".pdf"}

INDEX_BATCH_SIZE = 64

MIN_IMAGE_WIDTH = 80
MIN_IMAGE_HEIGHT = 80

MAX_IMAGE_SIZE = 768

OCR_MIN_TEXT_LENGTH_FOR_CAPTION = 80

BLIP_MODEL_NAME = (
    "Salesforce/blip-image-captioning-base"
)


# ============================================================
# ML MODELS
# ============================================================

blip_processor = None
blip_model = None
ocr_reader = None


# ============================================================
# MODEL INITIALIZATION
# ============================================================

def initialize_blip() -> bool:
    """Load BLIP image captioning model."""

    global blip_processor, blip_model

    if (
        blip_processor is not None
        and blip_model is not None
    ):
        return True

    try:
        print("Loading BLIP image captioning model...")

        blip_processor = (
            BlipProcessor.from_pretrained(
                BLIP_MODEL_NAME
            )
        )

        blip_model = (
            BlipForConditionalGeneration
            .from_pretrained(
                BLIP_MODEL_NAME
            )
        )

        blip_model.to("cpu")
        blip_model.eval()

        print("✓ BLIP loaded successfully.")

        return True

    except Exception as exc:

        print(
            f"⚠ BLIP initialization failed: {exc}"
        )

        blip_processor = None
        blip_model = None

        return False


def initialize_ocr() -> bool:
    """Load EasyOCR."""

    global ocr_reader

    if ocr_reader is not None:
        return True

    try:

        print("Loading EasyOCR model...")

        ocr_reader = easyocr.Reader(
            ["en"],
            gpu=False,
            verbose=False,
        )

        print("✓ EasyOCR loaded successfully.")

        return True

    except Exception as exc:

        print(
            f"⚠ EasyOCR initialization failed: {exc}"
        )

        ocr_reader = None

        return False


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text: str) -> str:
    """Clean extracted text."""

    if not text:
        return ""

    text = text.replace(
        "\x00",
        " ",
    )

    text = re.sub(
        r"[ \t]+",
        " ",
        text,
    )

    text = re.sub(
        r"\n\s*\n+",
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
    """Calculate SHA256 hash."""

    sha256 = hashlib.sha256()

    with file_path.open(
        "rb"
    ) as file:

        while True:

            chunk = file.read(
                1024 * 1024
            )

            if not chunk:
                break

            sha256.update(chunk)

    return sha256.hexdigest()


# ============================================================
# MANIFEST
# ============================================================

def load_manifest() -> Dict:
    """
    Load existing manifest.

    Supports BOTH:

    Old/current format:
    {
        "ec2.pdf": {
            "sha256": "...",
            "size": 123,
            "modified_time": 123
        }
    }

    New format:
    {
        "version": 1,
        "files": {
            "ec2.pdf": {
                "hash": "...",
                ...
            }
        }
    }
    """

    if not MANIFEST_FILE.exists():
        return {
            "version": 1,
            "files": {},
        }

    try:

        with MANIFEST_FILE.open(
            "r",
            encoding="utf-8",
        ) as file:

            raw_manifest = json.load(file)

        # ----------------------------------------------------
        # New format
        # ----------------------------------------------------

        if (
            isinstance(raw_manifest, dict)
            and "files" in raw_manifest
        ):

            return raw_manifest

        # ----------------------------------------------------
        # Existing old format
        # ----------------------------------------------------

        converted_files = {}

        if isinstance(
            raw_manifest,
            dict,
        ):

            for filename, data in raw_manifest.items():

                if not isinstance(
                    data,
                    dict,
                ):
                    continue

                converted_files[
                    filename
                ] = {
                    "sha256": data.get(
                        "sha256"
                    ),
                    "size": data.get(
                        "size"
                    ),
                    "modified_time": data.get(
                        "modified_time"
                    ),
                }

        return {
            "version": 1,
            "files": converted_files,
        }

    except Exception as exc:

        print(
            f"⚠ Could not load manifest: {exc}"
        )

        return {
            "version": 1,
            "files": {},
        }


def save_manifest(
    manifest: Dict,
) -> None:
    """Save manifest."""

    MANIFEST_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_file = (
        MANIFEST_FILE.with_suffix(
            ".tmp"
        )
    )

    with temporary_file.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            manifest,
            file,
            indent=4,
            ensure_ascii=False,
        )

    temporary_file.replace(
        MANIFEST_FILE
    )


# ============================================================
# CHANGE DETECTION
# ============================================================

def get_pdf_changes(
    pdf_files: List[Path],
    manifest: Dict,
) -> Tuple[
    List[Path],
    List[Path],
    List[Path],
    List[str],
]:
    """
    Determine new, modified, unchanged and removed PDFs.

    Existing SHA256 manifest is respected.
    """

    new_files = []
    modified_files = []
    unchanged_files = []

    current_names = set()

    manifest_files = manifest.get(
        "files",
        {},
    )

    for pdf_path in pdf_files:

        name = pdf_path.name

        current_names.add(name)

        current_hash = calculate_file_hash(
            pdf_path
        )

        previous = manifest_files.get(
            name
        )

        # ----------------------------------------------------
        # New
        # ----------------------------------------------------

        if previous is None:

            new_files.append(
                pdf_path
            )

            continue

        # ----------------------------------------------------
        # Support both hash field names
        # ----------------------------------------------------

        previous_hash = previous.get(
            "sha256"
        )

        if previous_hash is None:
            previous_hash = previous.get(
                "hash"
            )

        # ----------------------------------------------------
        # Unchanged
        # ----------------------------------------------------

        if previous_hash == current_hash:

            unchanged_files.append(
                pdf_path
            )

        # ----------------------------------------------------
        # Modified
        # ----------------------------------------------------

        else:

            modified_files.append(
                pdf_path
            )

    # --------------------------------------------------------
    # Removed
    # --------------------------------------------------------

    removed_files = [
        name
        for name in manifest_files
        if name not in current_names
    ]

    return (
        new_files,
        modified_files,
        unchanged_files,
        removed_files,
    )


# ============================================================
# QDRANT DELETE
# ============================================================

def delete_pdf_from_qdrant(
    pdf_name: str,
) -> None:
    """Delete all Qdrant chunks belonging to a PDF."""

    print(
        f"Removing old Qdrant vectors for "
        f"{pdf_name}..."
    )

    deleted = delete_documents_by_source(
        pdf_name
    )

    print(
        f"✓ Removed {deleted} old chunks."
    )


# ============================================================
# BLIP CAPTIONING
# ============================================================

def generate_image_caption(
    image: Image.Image,
) -> str:
    """Generate BLIP caption."""

    if not initialize_blip():
        return ""

    try:

        image = image.convert(
            "RGB"
        )

        width, height = image.size

        if max(
            width,
            height,
        ) > MAX_IMAGE_SIZE:

            scale = (
                MAX_IMAGE_SIZE
                / max(width, height)
            )

            image = image.resize(
                (
                    max(
                        1,
                        int(width * scale),
                    ),
                    max(
                        1,
                        int(height * scale),
                    ),
                ),
                Image.Resampling.LANCZOS,
            )

        inputs = blip_processor(
            images=image,
            return_tensors="pt",
        )

        with torch.no_grad():

            output = blip_model.generate(
                **inputs,
                max_new_tokens=30,
                num_beams=1,
            )

        caption = (
            blip_processor.decode(
                output[0],
                skip_special_tokens=True,
            )
        )

        return clean_text(
            caption
        )

    except Exception as exc:

        print(
            f"⚠ BLIP captioning failed: {exc}"
        )

        return ""


# ============================================================
# OCR
# ============================================================

def extract_ocr_text(
    image: Image.Image,
) -> str:
    """Extract text using EasyOCR."""

    if not initialize_ocr():
        return ""

    try:

        image = image.convert(
            "RGB"
        )

        image_array = np.array(
            image
        )

        results = ocr_reader.readtext(
            image_array,
            detail=0,
            paragraph=True,
        )

        text = "\n".join(
            str(item)
            for item in results
            if item
        )

        return clean_text(
            text
        )

    except Exception as exc:

        print(
            f"⚠ OCR failed: {exc}"
        )

        return ""


# ============================================================
# PDF TEXT
# ============================================================

def extract_text_documents(
    pdf_path: Path,
    pdf_document,
) -> List[Document]:
    """Extract normal PDF text."""

    documents = []

    for page_number, page in enumerate(
        pdf_document,
        start=1,
    ):

        try:

            text = page.get_text()

            text = clean_text(
                text
            )

            if not text:
                continue

            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": pdf_path.name,
                        "document_name": pdf_path.name,
                        "page": page_number,
                        "content_type": "text",
                    },
                )
            )

        except Exception as exc:

            print(
                f"⚠ Text extraction failed "
                f"for {pdf_path.name}, "
                f"page {page_number}: {exc}"
            )

    return documents


# ============================================================
# TABLE EXTRACTION
# ============================================================

def extract_table_documents(
    pdf_path: Path,
) -> List[Document]:
    """Extract tables using Camelot."""

    documents = []

    try:

        tables = camelot.read_pdf(
            str(pdf_path),
            pages="all",
            flavor="stream",
        )

        print(
            f"  Tables detected: {len(tables)}"
        )

        for table_index, table in enumerate(
            tables,
            start=1,
        ):

            try:

                dataframe = table.df

                if (
                    dataframe is None
                    or dataframe.empty
                ):
                    continue

                table_text = dataframe.to_csv(
                    index=False,
                    header=False,
                )

                table_text = clean_text(
                    table_text
                )

                if not table_text:
                    continue

                page_number = getattr(
                    table,
                    "page",
                    None,
                )

                try:

                    if page_number is not None:
                        page_number = int(
                            page_number
                        )

                except Exception:
                    pass

                documents.append(
                    Document(
                        page_content=table_text,
                        metadata={
                            "source": pdf_path.name,
                            "document_name": pdf_path.name,
                            "page": page_number,
                            "content_type": "table",
                            "table_index": table_index,
                        },
                    )
                )

            except Exception as exc:

                print(
                    f"⚠ Table {table_index} "
                    f"processing failed: {exc}"
                )

    except Exception as exc:

        print(
            f"⚠ Camelot failed for "
            f"{pdf_path.name}: {exc}"
        )

    return documents


# ============================================================
# IMAGE EXTRACTION
# ============================================================

def extract_image_documents(
    pdf_path: Path,
    pdf_document,
) -> List[Document]:
    """Extract and process PDF images."""

    documents = []

    processed_fingerprints = set()

    total_images = 0
    skipped_images = 0
    ocr_images = 0
    captioned_images = 0

    for page_number, page in enumerate(
        pdf_document,
        start=1,
    ):

        try:

            page_dict = page.get_text(
                "dict"
            )

            blocks = page_dict.get(
                "blocks",
                [],
            )

            for image_index, block in enumerate(
                blocks,
                start=1,
            ):

                if block.get(
                    "type"
                ) != 1:
                    continue

                total_images += 1

                try:

                    width = int(
                        block.get(
                            "width",
                            0,
                        )
                    )

                    height = int(
                        block.get(
                            "height",
                            0,
                        )
                    )

                    if (
                        width < MIN_IMAGE_WIDTH
                        or height < MIN_IMAGE_HEIGHT
                    ):

                        skipped_images += 1
                        continue

                    image_bytes = block.get(
                        "image"
                    )

                    # Fallback to xref
                    if not image_bytes:

                        xref = block.get(
                            "xref"
                        )

                        if xref:

                            image_info = (
                                pdf_document
                                .extract_image(
                                    xref
                                )
                            )

                            image_bytes = (
                                image_info.get(
                                    "image"
                                )
                            )

                    if not image_bytes:

                        skipped_images += 1
                        continue

                    fingerprint = (
                        hashlib.sha1(
                            image_bytes
                        ).hexdigest()
                    )

                    if (
                        fingerprint
                        in processed_fingerprints
                    ):
                        continue

                    processed_fingerprints.add(
                        fingerprint
                    )

                    image = Image.open(
                        io.BytesIO(
                            image_bytes
                        )
                    ).convert(
                        "RGB"
                    )

                    # ------------------------------------------------
                    # OCR
                    # ------------------------------------------------

                    ocr_text = extract_ocr_text(
                        image
                    )

                    caption = ""

                    if ocr_text:
                        ocr_images += 1

                    # ------------------------------------------------
                    # BLIP
                    # ------------------------------------------------

                    if (
                        len(ocr_text)
                        < OCR_MIN_TEXT_LENGTH_FOR_CAPTION
                    ):

                        caption = (
                            generate_image_caption(
                                image
                            )
                        )

                        if caption:
                            captioned_images += 1

                    # ------------------------------------------------
                    # Build content
                    # ------------------------------------------------

                    description_parts = []

                    if caption:

                        description_parts.append(
                            f"Image description: "
                            f"{caption}"
                        )

                    if ocr_text:

                        description_parts.append(
                            f"Extracted text: "
                            f"{ocr_text}"
                        )

                    if not description_parts:
                        continue

                    content = "\n".join(
                        description_parts
                    )

                    documents.append(
                        Document(
                            page_content=content,
                            metadata={
                                "source": pdf_path.name,
                                "document_name": pdf_path.name,
                                "page": page_number,
                                "content_type": "image",
                                "image_index": image_index,
                                "image_width": width,
                                "image_height": height,
                                "image_fingerprint": fingerprint,
                                "has_ocr": bool(
                                    ocr_text
                                ),
                                "has_caption": bool(
                                    caption
                                ),
                            },
                        )
                    )

                except Exception as exc:

                    print(
                        f"⚠ Image processing failed "
                        f"for {pdf_path.name}, "
                        f"page {page_number}: {exc}"
                    )

        except Exception as exc:

            print(
                f"⚠ Image block processing failed "
                f"for {pdf_path.name}, "
                f"page {page_number}: {exc}"
            )

    print(
        f"  Images found: {total_images}, "
        f"skipped: {skipped_images}, "
        f"OCR: {ocr_images}, "
        f"captioned: {captioned_images}"
    )

    return documents


# ============================================================
# LOAD PDF
# ============================================================

def load_pdf(
    pdf_path: Path,
) -> List[Document]:
    """Extract text, tables and images."""

    print(
        f"\nProcessing: {pdf_path.name}"
    )

    pdf_document = None

    try:

        pdf_document = pymupdf.open(
            str(pdf_path)
        )

        print(
            f"  Pages: {len(pdf_document)}"
        )

        text_documents = (
            extract_text_documents(
                pdf_path,
                pdf_document,
            )
        )

        image_documents = (
            extract_image_documents(
                pdf_path,
                pdf_document,
            )
        )

        pdf_document.close()
        pdf_document = None

        table_documents = (
            extract_table_documents(
                pdf_path
            )
        )

        documents = (
            text_documents
            + table_documents
            + image_documents
        )

        print(
            f"  Extracted: "
            f"{len(text_documents)} text, "
            f"{len(table_documents)} tables, "
            f"{len(image_documents)} images"
        )

        print(
            f"  Total documents: "
            f"{len(documents)}"
        )

        return documents

    except Exception:

        if pdf_document is not None:

            try:
                pdf_document.close()
            except Exception:
                pass

        raise


# ============================================================
# STATISTICS
# ============================================================

def display_statistics(
    documents: List[Document],
) -> None:
    """Display document statistics."""

    counts = {
        "text": 0,
        "table": 0,
        "image": 0,
    }

    total_characters = 0

    for document in documents:

        content_type = document.metadata.get(
            "content_type",
            "text",
        )

        if content_type in counts:
            counts[content_type] += 1

        total_characters += len(
            document.page_content
        )

    print("\n" + "=" * 60)
    print("INGESTION STATISTICS")
    print("=" * 60)

    print(
        f"Text documents   : "
        f"{counts['text']}"
    )

    print(
        f"Table documents  : "
        f"{counts['table']}"
    )

    print(
        f"Image documents  : "
        f"{counts['image']}"
    )

    print(
        f"Total documents  : "
        f"{len(documents)}"
    )

    print(
        f"Total characters : "
        f"{total_characters:,}"
    )

    print("=" * 60)


# ============================================================
# QDRANT INGESTION
# ============================================================

def ingest_documents(
    documents: List[Document],
) -> int:
    """Add documents to Qdrant."""

    if not documents:
        return 0

    print(
        f"\nAdding {len(documents)} documents "
        f"to Qdrant..."
    )

    added = add_documents(
        documents,
        batch_size=INDEX_BATCH_SIZE,
    )

    print(
        f"✓ Successfully indexed "
        f"{added} documents."
    )

    return added


# ============================================================
# PDF DISCOVERY
# ============================================================

def discover_pdfs() -> List[Path]:
    """Find PDFs in rag_data."""

    if not RAG_DIRECTORY.exists():

        print(
            f"⚠ RAG directory does not exist: "
            f"{RAG_DIRECTORY}"
        )

        return []

    return sorted(
        [
            path
            for path in RAG_DIRECTORY.iterdir()
            if (
                path.is_file()
                and path.suffix.lower()
                in SUPPORTED_EXTENSIONS
            )
        ]
    )


# ============================================================
# PROCESS INCREMENTAL PDFs
# ============================================================

def process_incremental_pdfs(
    pdf_files: List[Path],
    manifest: Dict,
):
    """Process only new and modified PDFs."""

    (
        new_files,
        modified_files,
        unchanged_files,
        removed_files,
    ) = get_pdf_changes(
        pdf_files,
        manifest,
    )

    print("\n" + "=" * 60)
    print("INCREMENTAL ANALYSIS")
    print("=" * 60)

    print(
        f"New PDFs       : {len(new_files)}"
    )

    print(
        f"Modified PDFs  : {len(modified_files)}"
    )

    print(
        f"Unchanged PDFs : {len(unchanged_files)}"
    )

    print(
        f"Removed PDFs   : {len(removed_files)}"
    )

    processed_count = 0
    failed_count = 0
    total_indexed = 0

    # ========================================================
    # NEW FILES
    # ========================================================

    for pdf_path in new_files:

        try:

            documents = load_pdf(
                pdf_path
            )

            if not documents:

                print(
                    f"⚠ No documents extracted "
                    f"from {pdf_path.name}"
                )

                continue

            display_statistics(
                documents
            )

            indexed = ingest_documents(
                documents
            )

            manifest["files"][
                pdf_path.name
            ] = {
                "sha256": calculate_file_hash(
                    pdf_path
                ),
                "size": pdf_path.stat().st_size,
                "modified_time": pdf_path.stat().st_mtime,
                "processed_documents": indexed,
            }

            save_manifest(
                manifest
            )

            processed_count += 1
            total_indexed += indexed

        except Exception as exc:

            failed_count += 1

            print(
                f"✗ Failed to ingest "
                f"{pdf_path.name}: {exc}"
            )

    # ========================================================
    # MODIFIED FILES
    # ========================================================

    for pdf_path in modified_files:

        try:

            print(
                f"\nModified PDF detected: "
                f"{pdf_path.name}"
            )

            # Extract first so an extraction failure
            # doesn't delete the existing vectors.
            documents = load_pdf(
                pdf_path
            )

            if not documents:

                print(
                    f"⚠ No documents extracted "
                    f"from {pdf_path.name}"
                )

                continue

            display_statistics(
                documents
            )

            # Remove previous version.
            delete_pdf_from_qdrant(
                pdf_path.name
            )

            # Index new version.
            indexed = ingest_documents(
                documents
            )

            manifest["files"][
                pdf_path.name
            ] = {
                "sha256": calculate_file_hash(
                    pdf_path
                ),
                "size": pdf_path.stat().st_size,
                "modified_time": pdf_path.stat().st_mtime,
                "processed_documents": indexed,
            }

            save_manifest(
                manifest
            )

            processed_count += 1
            total_indexed += indexed

        except Exception as exc:

            failed_count += 1

            print(
                f"✗ Failed to update "
                f"{pdf_path.name}: {exc}"
            )

    # ========================================================
    # REMOVED FILES
    # ========================================================

    for removed_name in removed_files:

        print(
            f"\nRemoved from rag_data: "
            f"{removed_name}"
        )

        # Preserve your original behavior:
        # remove from manifest but don't automatically
        # delete Qdrant vectors.
        manifest["files"].pop(
            removed_name,
            None,
        )

    if removed_files:
        save_manifest(
            manifest
        )

    return (
        processed_count,
        len(unchanged_files),
        len(removed_files),
        total_indexed,
        failed_count,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("AWS ASSISTANT - QDRANT RAG INGESTION")
    print("=" * 60)

    print(
        f"Project root : {PROJECT_ROOT}"
    )

    print(
        f"RAG directory: {RAG_DIRECTORY}"
    )

    print(
        f"Manifest     : {MANIFEST_FILE}"
    )

    # --------------------------------------------------------
    # Discover PDFs
    # --------------------------------------------------------

    pdf_files = discover_pdfs()

    if not pdf_files:

        print(
            "\n⚠ No PDF files found."
        )

        return

    print(
        f"\nFound {len(pdf_files)} PDF file(s)."
    )

    for pdf_path in pdf_files:

        print(
            f"  - {pdf_path.name}"
        )

    # --------------------------------------------------------
    # Manifest
    # --------------------------------------------------------

    manifest = load_manifest()

    # --------------------------------------------------------
    # Incremental processing
    # --------------------------------------------------------

    (
        processed_count,
        unchanged_count,
        removed_count,
        total_indexed,
        failed_count,
    ) = process_incremental_pdfs(
        pdf_files,
        manifest,
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)

    print(
        f"PDFs processed : "
        f"{processed_count}"
    )

    print(
        f"PDFs unchanged : "
        f"{unchanged_count}"
    )

    print(
        f"PDFs removed   : "
        f"{removed_count}"
    )

    print(
        f"Vectors indexed: "
        f"{total_indexed}"
    )

    print(
        f"Failures       : "
        f"{failed_count}"
    )

    if (
        processed_count == 0
        and failed_count == 0
    ):

        print(
            "\n✓ No new or modified PDFs."
        )

        print(
            "✓ Existing Qdrant vectors "
            "were left untouched."
        )

    elif failed_count == 0:

        print(
            "\n✓ Qdrant ingestion completed."
        )

    else:

        print(
            "\n⚠ Ingestion completed "
            "with failures."
        )

    print("=" * 60)


if __name__ == "__main__":
    main()