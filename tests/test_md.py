"""Tests to ensure README.md and docs/domains.md stay in sync with actual domains, modules, and datasets."""

import re
from pathlib import Path

import pytest

from discogen import create_config
from discogen.utils import get_domains, get_modules

ROOT: Path = Path(__file__).resolve().parent.parent
README_PATH: Path = ROOT / "README.md"
DOMAINS_MD_PATH: Path = ROOT / "docs" / "domains.md"


def _read_file(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Helpers to parse README.md table
# ---------------------------------------------------------------------------


def _find_task_domains_heading(lines: list[str]) -> int:
    """Return the line index of the '## Task Domains' heading."""
    for i, line in enumerate(lines):
        if re.match(r"^##\s+Task Domains\s*$", line):
            return i
    raise AssertionError("Could not find '## Task Domains' heading in README.md")


def _collect_table_rows(lines: list[str], start_idx: int) -> list[str]:
    """Collect markdown table data rows starting after *start_idx*."""
    table_rows: list[str] = []
    in_table: bool = False
    header_seen: bool = False
    for line in lines[start_idx + 1 :]:
        stripped: str = line.strip()
        if not stripped:
            if in_table:
                break  # blank line ends the table
            continue
        if stripped.startswith("|"):
            if not header_seen:
                header_seen = True  # this is the header row
                in_table = True
                continue
            if re.match(r"^\|[\s:*-]+\|", stripped):
                continue  # separator row
            table_rows.append(stripped)
        elif in_table:
            break  # non-table line ends the table
    return table_rows


def _parse_readme_table(readme: str) -> dict[str, dict[str, str | set[str]]]:
    """Parse the Task Domains table from README.md.

    Returns a dict mapping domain name -> {"modules": set, "datasets_raw": str, "description": str}.
    """
    lines: list[str] = readme.split("\n")
    start_idx: int = _find_task_domains_heading(lines)
    table_rows: list[str] = _collect_table_rows(lines, start_idx)
    assert table_rows, "Could not find any data rows in the Task Domains table in README.md"

    result: dict[str, dict[str, str | set[str]]] = {}
    for row in table_rows:
        cols: list[str] = [c.strip() for c in row.split("|")]
        cols = [c for c in cols if c]
        if len(cols) < 4:
            continue

        domain_name: str = re.sub(r"\*\*(.+)\*\*", r"\1", cols[0]).strip()
        modules_raw: str = cols[1].strip()
        modules_set: set[str] = {m.strip() for m in modules_raw.split(",") if m.strip()}
        result[domain_name] = {"modules": modules_set, "datasets_raw": cols[2].strip(), "description": cols[3].strip()}
    return result


# ---------------------------------------------------------------------------
# Helpers to parse docs/domains.md
# ---------------------------------------------------------------------------


def _parse_domains_md(text: str) -> dict[str, dict[str, set[str]]]:
    """Parse docs/domains.md into a dict mapping domain -> {"modules": set, "datasets": set}.

    Expects sections like:
        ## DomainName
        ...
        ### Modules
        `mod1`, `mod2`
        ### Datasets
        `ds1`, `ds2`
    """
    # Split on ## headings (level 2) that are domain sections
    domain_sections: list[str] = re.split(r"\n## (?!#)", text)
    result: dict[str, dict[str, set[str]]] = {}

    for section in domain_sections[1:]:  # skip preamble before first ##
        lines: list[str] = section.strip().split("\n")
        domain_name: str = lines[0].strip()

        # Extract modules
        modules: set[str] = set()
        datasets: set[str] = set()

        current_subsection: str | None = None
        for line in lines[1:]:
            if line.startswith("### Modules"):
                current_subsection = "modules"
                continue
            elif line.startswith("### Datasets"):
                current_subsection = "datasets"
                continue
            elif line.startswith("### "):
                current_subsection = None
                continue

            if current_subsection and line.strip():
                # Extract backtick-quoted items
                items: list[str] = re.findall(r"`([^`]+)`", line)
                if current_subsection == "modules":
                    modules.update(items)
                elif current_subsection == "datasets":
                    datasets.update(items)

        if domain_name:
            result[domain_name] = {"modules": modules, "datasets": datasets}

    return result


# ---------------------------------------------------------------------------
# Helpers for dataset verification in README
# ---------------------------------------------------------------------------


def _group_datasets_by_prefix(datasets: list[str]) -> tuple[dict[str, list[str]], list[str]]:
    """Split datasets into prefixed groups (containing '/') and non-prefixed lists."""
    prefixed_groups: dict[str, list[str]] = {}
    non_prefixed: list[str] = []
    for ds in datasets:
        if "/" in ds:
            prefix: str = ds.split("/")[0]
            prefixed_groups.setdefault(prefix, []).append(ds)
        else:
            non_prefixed.append(ds)
    return prefixed_groups, non_prefixed


def _extract_readme_counts(datasets_cell: str) -> dict[str, int]:
    """Extract all ``N keyword`` count patterns from a README table cell."""
    count_patterns: list[tuple[str, str]] = re.findall(r"(\d+)\s+([A-Za-z_]+)", datasets_cell)
    return {name.lower(): int(n) for n, name in count_patterns}


def _check_prefixed_groups(
    prefixed_groups: dict[str, list[str]],
    cell_lower: str,
    datasets_cell: str,
    readme_counts: dict[str, int],
    domain: str,
) -> None:
    """Assert that every prefixed dataset group appears in the README cell with correct count."""
    for prefix, members in prefixed_groups.items():
        prefix_lower: str = prefix.lower()
        assert prefix_lower in cell_lower, (
            f"Dataset group '{prefix}' ({len(members)} datasets) for domain {domain} "
            f"not mentioned in README.md table. Cell content: {datasets_cell}"
        )
        if prefix_lower in readme_counts:
            assert readme_counts[prefix_lower] == len(members), (
                f"Dataset count mismatch for group '{prefix}' in domain {domain}. "
                f"README says {readme_counts[prefix_lower]}, code has {len(members)}. "
                f"Datasets: {members}"
            )


def _check_non_prefixed_datasets(
    non_prefixed: list[str], cell_lower: str, datasets_cell: str, readme_counts: dict[str, int], domain: str
) -> None:
    """Assert that non-prefixed datasets appear in the README cell, either as counts or names."""
    # Group by stripping trailing digits
    # e.g. LibriBrainSherlock1, LibriBrainSherlock2 -> group "LibriBrainSherlock"
    stem_groups: dict[str, list[str]] = {}
    ungrouped: list[str] = []
    for ds in non_prefixed:
        stem_match: re.Match[str] | None = re.match(r"^(.+?)\d+[a-z]*$", ds)
        if stem_match:
            stem: str = stem_match.group(1)
            stem_groups.setdefault(stem, []).append(ds)
        else:
            ungrouped.append(ds)

    # Check stem groups: either "N StemName" count pattern, or each member listed
    for stem, members in stem_groups.items():
        stem_lower: str = stem.lower()
        if stem_lower in readme_counts:
            assert readme_counts[stem_lower] == len(members), (
                f"Dataset count mismatch for group '{stem}' in domain {domain}. "
                f"README says {readme_counts[stem_lower]}, code has {len(members)}. "
                f"Datasets: {members}"
            )
        else:
            for ds in members:
                assert ds.lower() in cell_lower, (
                    f"Dataset '{ds}' for domain {domain} not mentioned in README.md table. "
                    f"Cell content: {datasets_cell}"
                )

    # Check ungrouped non-prefixed datasets individually
    for ds in ungrouped:
        assert ds.lower() in cell_lower, (
            f"Dataset '{ds}' for domain {domain} not mentioned in README.md table. Cell content: {datasets_cell}"
        )


# ===========================================================================
# Tests for README.md
# ===========================================================================


class TestReadme:
    """Verify that README.md Task Domains table matches actual code domains, modules, and datasets."""

    @pytest.fixture(autouse=True)
    def _load(self) -> None:
        self.readme: str = _read_file(README_PATH)
        self.table: dict[str, dict[str, str | set[str]]] = _parse_readme_table(self.readme)

    def test_all_domains_present(self) -> None:
        """Every domain returned by get_domains() must appear in the README table."""
        domains: set[str] = set(get_domains())
        readme_domains: set[str] = set(self.table.keys())
        missing: set[str] = domains - readme_domains
        assert not missing, f"Domains missing from README.md table: {missing}"

    def test_no_extra_domains(self) -> None:
        """README table should not list domains that don't exist in code."""
        domains: set[str] = set(get_domains())
        readme_domains: set[str] = set(self.table.keys())
        extra: set[str] = readme_domains - domains
        assert not extra, f"README.md table lists domains not in code: {extra}"

    @pytest.mark.parametrize("domain, modules", get_modules().items())
    def test_modules_match(self, domain: str, modules: list[str]) -> None:
        """Modules listed in README table must match actual modules for each domain."""
        assert domain in self.table, f"Domain {domain} missing from README.md table"
        readme_modules: set[str] = self.table[domain]["modules"]  # type: ignore[assignment]
        code_modules: set[str] = set(modules)
        assert code_modules == readme_modules, (
            f"Module mismatch for {domain} in README.md.\n"
            f"  In code but not README: {code_modules - readme_modules}\n"
            f"  In README but not code: {readme_modules - code_modules}"
        )

    @pytest.mark.parametrize("domain", get_domains())
    def test_datasets_mentioned(self, domain: str) -> None:
        """Dataset counts and group names in README table must match the actual datasets.

        The README uses summary descriptions like "4 MinAtar, 7 Brax, 2 Craftax"
        for prefixed datasets (MinAtar/Asterix, Brax/Ant, etc.).

        For non-prefixed datasets, the README may list abbreviated names like
        "11 synthetic functions (Ackley, Branin, ...)" or full names like
        "CIFAR10, CIFAR10C, ...".

        We verify:
        - Prefixed groups: group name appears and count matches.
        - Non-prefixed datasets: each dataset name (case-insensitive) appears in the cell.
        - Total dataset count matches any leading total count in the cell.
        """
        assert domain in self.table, f"Domain {domain} missing from README.md table"
        config: dict[str, list[str]] = create_config(domain)
        datasets: list[str] = list(config["train_task_id"])
        datasets_cell: str = self.table[domain]["datasets_raw"]  # type: ignore[assignment]
        cell_lower: str = datasets_cell.lower()

        prefixed_groups, non_prefixed = _group_datasets_by_prefix(datasets)
        readme_counts: dict[str, int] = _extract_readme_counts(datasets_cell)

        _check_prefixed_groups(prefixed_groups, cell_lower, datasets_cell, readme_counts, domain)
        _check_non_prefixed_datasets(non_prefixed, cell_lower, datasets_cell, readme_counts, domain)

        # Verify total dataset count if cell starts with "N ..." and all datasets are non-prefixed
        total_count: int = len(datasets)
        total_match: re.Match[str] | None = re.match(r"^(\d+)\s+", datasets_cell)
        if total_match and not prefixed_groups:
            readme_total: int = int(total_match.group(1))
            assert readme_total == total_count, (
                f"Total dataset count mismatch for domain {domain}. README says {readme_total}, code has {total_count}."
            )


# ===========================================================================
# Tests for docs/domains.md
# ===========================================================================


class TestDomainsMd:
    """Verify that docs/domains.md matches actual code domains, modules, and datasets."""

    @pytest.fixture(autouse=True)
    def _load(self) -> None:
        self.text: str = _read_file(DOMAINS_MD_PATH)
        self.parsed: dict[str, dict[str, set[str]]] = _parse_domains_md(self.text)

    def test_all_domains_present(self) -> None:
        """Every domain returned by get_domains() must have a section in docs/domains.md."""
        domains: set[str] = set(get_domains())
        md_domains: set[str] = set(self.parsed.keys())
        missing: set[str] = domains - md_domains
        assert not missing, f"Domains missing from docs/domains.md: {missing}"

    def test_no_extra_domains(self) -> None:
        """docs/domains.md should not list domains that don't exist in code."""
        domains: set[str] = set(get_domains())
        md_domains: set[str] = set(self.parsed.keys())
        extra: set[str] = md_domains - domains
        assert not extra, f"docs/domains.md lists domains not in code: {extra}"

    @pytest.mark.parametrize("domain, modules", get_modules().items())
    def test_modules_match(self, domain: str, modules: list[str]) -> None:
        """Modules listed in docs/domains.md must match actual modules for each domain."""
        assert domain in self.parsed, f"Domain {domain} missing from docs/domains.md"
        md_modules: set[str] = self.parsed[domain]["modules"]
        code_modules: set[str] = set(modules)
        assert code_modules == md_modules, (
            f"Module mismatch for {domain} in docs/domains.md.\n"
            f"  In code but not docs/domains.md: {code_modules - md_modules}\n"
            f"  In docs/domains.md but not code: {md_modules - code_modules}"
        )

    @pytest.mark.parametrize("domain", get_domains())
    def test_datasets_match(self, domain: str) -> None:
        """Datasets listed in docs/domains.md must match actual datasets for each domain."""
        assert domain in self.parsed, f"Domain {domain} missing from docs/domains.md"
        config: dict[str, list[str]] = create_config(domain)
        code_datasets: set[str] = set(config["train_task_id"])
        md_datasets: set[str] = self.parsed[domain]["datasets"]

        # Normalize: docs may use slash-separated names like MinAtar/Asterix
        assert code_datasets == md_datasets, (
            f"Dataset mismatch for {domain} in docs/domains.md.\n"
            f"  In code but not docs/domains.md: {code_datasets - md_datasets}\n"
            f"  In docs/domains.md but not code: {md_datasets - code_datasets}"
        )
