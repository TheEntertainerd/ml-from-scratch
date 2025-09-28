import glob
import os
import re
from typing import List, Tuple

import genanki

HERE = os.path.dirname(__file__)
BASIC_PATTERN = os.path.join(HERE, "*/basic.md")
ADVANCED_PATTERN = os.path.join(HERE, "*/advanced.md")

# A card is (front text, back text, tags)
Card = Tuple[str, str, List[str]]


def parse_md(path: str) -> List[Card]:
    """Parse a markdown file into a list of (front, back, tags) cards."""
    with open(path, encoding="utf-8") as f:
        sections = f.read().split("---")
    cards: List[Card] = []
    for sec in sections:
        mF = re.search(r"\*\*Front:\*\*\s*(.*?)$", sec, re.M)
        mB = re.search(r"\*\*Back:\*\*\s*([\s\S]*?)(?=\n\*\*Tags:\*\*|\Z)", sec, re.DOTALL)
        mT = re.search(r"\*\*Tags:\*\*\s*(.+)", sec)
        if mF and mB:
            front = mF.group(1).strip()
            back = mB.group(1).strip().replace("\n", "<br>")
            tags = mT.group(1).split() if mT else []
            cards.append((front, back, tags))
    return cards


def build_subdecks(level: str, decks_list: List[genanki.Deck], deck_id_base: int) -> None:
    """Build subdecks (basic or advanced) and append them to decks_list."""
    pattern = BASIC_PATTERN if level == "basic" else ADVANCED_PATTERN

    for idx, md_path in enumerate(sorted(glob.glob(pattern))):
        topic = os.path.basename(os.path.dirname(md_path)).replace("_", " ").title()
        deck_name = f"Machine Learning::{level.title()}::{topic}"
        deck_id = deck_id_base + idx + 1

        deck = genanki.Deck(deck_id, deck_name)
        model = genanki.Model(
            1607392319,
            "Simple Model",
            fields=[{"name": "Front"}, {"name": "Back"}],
            templates=[
                {
                    "name": "Card 1",
                    "qfmt": "{{Front}}",
                    "afmt": '{{FrontSide}}<hr id="answer">{{Back}}',
                }
            ],
            css="""
                .card {
                font-family: arial;
                font-size: 20px;
                text-align: center;
                color: black;
                background-color: white;
                }
            """,
        )

        for front, back, tags in parse_md(md_path):
            note = genanki.Note(model=model, fields=[front, back], tags=tags)
            deck.add_note(note)

        decks_list.append(deck)


if __name__ == "__main__":
    all_decks: List[genanki.Deck] = []
    build_subdecks("basic", all_decks, 1000000000)
    build_subdecks("advanced", all_decks, 2000000000)
    OUTPUT_DIR = os.path.join(HERE, "apkg")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    genanki.Package(all_decks).write_to_file(os.path.join(OUTPUT_DIR, "machine_learning.apkg"))
    print("Wrote all decks to", os.path.join(OUTPUT_DIR, "machine_learning.apkg"))
