"""Gate the committed review PDF against submission-level invariants."""

from pathlib import Path

from pypdf import PdfReader


PDF = Path(__file__).with_name("crop_alignment_wacv_2027.pdf")
EXPECTED_TITLE = (
    "Crop Alignment: A Structural-Null Test for Closed-Loop Labels in Segmentation "
    "Benchmarks"
)


def normalized(text: str) -> str:
    return " ".join(text.split())


def main() -> int:
    reader = PdfReader(PDF)
    pages = [normalized(page.extract_text() or "") for page in reader.pages]
    full_text = " ".join(pages)
    failures = []

    if EXPECTED_TITLE not in full_text:
        failures.append("expected paper title is missing")

    reference_pages = [i + 1 for i, text in enumerate(pages) if "References" in text]
    if not reference_pages or reference_pages[0] != 9:
        failures.append(f"references must begin on page 9, found {reference_pages}")

    if "github.com/kacytran1122" in full_text.lower():
        failures.append("identity-bearing repository URL leaked into review PDF")

    for identity in ("first author", "second author", "institution1"):
        if identity in full_text.lower():
            failures.append(f"identity leaked into review PDF: {identity}")

    if "*****" in full_text:
        print("warning: WACV paper ID is still the ***** placeholder")

    if failures:
        print("submission check: FAIL")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print(
        f"submission check: PASS ({len(pages)} PDF pages; "
        "8 main-content pages; review PDF anonymous)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
