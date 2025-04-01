"""Update file extensions in internal links in converted Markdown notebooks."""

import argparse
import re
from pathlib import Path


def update_notebook_links(
    path: Path, from_extension="ipynb", to_extension="md"
) -> None:
    contents = path.read_text()
    updated_contents = re.sub(
        rf"([0-9]+[a-z]?[_A-z]+)\.{from_extension}",
        rf"\1.{to_extension}",
        contents,
    )
    path.write_text(updated_contents)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("directory", type=Path)
    parser.add_argument("--glob-pattern", default="*.md")
    parser.add_argument("--from-extension", default="ipynb")
    parser.add_argument("--to-extension", default="md")
    args = parser.parse_args()
    for path in sorted(args.directory.glob(args.glob_pattern)):
        update_notebook_links(path, args.from_extension, args.to_extension)
