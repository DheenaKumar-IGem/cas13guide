"""Built-in target presets (FR1.3), and the geometry problem they expose.

These are the five miRNAs PanCRISP designs against, carried here so a run can
be reproduced from a preset id alone rather than from a pasted sequence whose
provenance nobody recorded. Every entry carries its miRBase accession, and
:data:`PRESET_SOURCE` records the exact file the sequences were taken from --
NFR4 and DR4 both want the database version on the record, and "we typed them
in from a paper" is not a database version.

Why a mature miRNA is an awkward Cas13 target
---------------------------------------------
A mature miRNA is 22-24 nt. An LwaCas13a spacer is canonically 28 nt. You
cannot tile a 28-mer across a 24-mer, so with the canonical geometry these
presets yield **zero** candidates -- not a poor design, no design at all.

There is no way to make that go away by choosing better software defaults,
because it is a property of the molecule. What the package does instead is make
the trade-off explicit:

* the spacer length is configurable within each ortholog's tolerated range
  (``EffectorParams.spacer_length_range``), so a 22 nt spacer against a 24 nt
  miRNA is expressible;
* a length outside the ortholog's *canonical* set is reported out-of-domain, so
  the activity model's bounds widen and the gate raises an R3 REVIEW rather
  than handing back a confident number for a geometry nobody validated;
* :func:`designability` says up front how many windows a preset can even yield,
  so "1 candidate" is a stated fact rather than a surprising empty table.

For a real diagnostic the usual answer is to detect an **amplicon** -- RT-RPA
product, or the pre-miRNA -- rather than the bare mature strand, which restores
a normal-length target and normal tiling. :func:`amplicon_target` builds that
target from a preset plus the flanking adapter/primer context, and is the
recommended path for anything past a geometry smoke-test.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..util.seq import normalize, windows
from .effectors import EFFECTORS
from .enums import Effector, TargetClass, Universe
from .spec import TargetSpec

#: Where the sequences below came from -- and, more importantly, how far that
#: has actually been checked.
#:
#: An earlier version of this dict carried a release string, a retrieval date
#: and a ``sha256``. None of them had been computed against anything: the
#: checksum was invented. That is worse than no provenance at all, because a
#: checksum is read as proof, and a reader who saw one would stop looking. The
#: fields are gone rather than corrected, since the honest value for each is
#: "unknown" and a dict of unknowns is just a longer way of saying unverified.
#:
#: What is true: these are the five miRNAs named in the requirements, with
#: their miRBase MIMAT accessions, transcribed from general knowledge of the
#: database. Three of the five would need no correction if checked; two carry
#: 3' nucleotides beyond the commonly cited mature form and may be isomiRs, a
#: different release, or simply wrong. Nobody has looked.
#:
#: :func:`verify_against_mirbase` discharges this. Point it at a downloaded
#: ``mature.fa`` and it reports, per preset, whether the accession is present
#: and whether the sequence matches -- and returns the file's real checksum so
#: the result can be recorded. Until someone runs it, every run that uses a
#: preset says so in its limitations.
PRESET_SOURCE: dict[str, str] = {
    "database": "miRBase",
    "url": "https://www.mirbase.org/download/mature.fa",
    "verified": "no -- sequences are unchecked against any specific release",
}

#: True once :func:`verify_against_mirbase` has been run and the presets
#: updated. Flipping this by hand without running it reintroduces exactly the
#: problem the invented checksum created.
PRESETS_VERIFIED = False


@dataclass(frozen=True, slots=True)
class TargetPreset:
    """One named, accessioned target sequence."""

    preset_id: str
    display_name: str
    accession: str
    sequence: str
    #: What the sequence *is*, since it changes how it should be targeted.
    kind: str = "mature_mirna"
    biotype: Universe = Universe.NCRNA
    note: str = ""

    @property
    def length(self) -> int:
        return len(self.sequence)

    def to_target_spec(
        self,
        target_id: str | None = None,
        target_class: TargetClass = TargetClass.REQUIRED,
        expression: float | None = None,
    ) -> TargetSpec:
        return TargetSpec(
            target_id=target_id or self.preset_id,
            sequence=self.sequence,
            target_class=target_class,
            gene=self.display_name,
            biotype=self.biotype,
            expression=expression,
        )


def _preset(preset_id: str, name: str, accession: str, sequence: str, **kw) -> TargetPreset:
    return TargetPreset(
        preset_id=preset_id,
        display_name=name,
        accession=accession,
        sequence=normalize(sequence),
        **kw,
    )


#: The five PanCRISP miRNA targets. See :data:`PRESET_SOURCE` for how far
#: their provenance has actually been established -- which is not far.
MIRNA_PRESETS: dict[str, TargetPreset] = {
    p.preset_id: p
    for p in (
        _preset("hsa-miR-21-5p", "hsa-miR-21-5p", "MIMAT0000076",
                "UAGCUUAUCAGACUGAUGUUGACU"),
        _preset("hsa-miR-23a-3p", "hsa-miR-23a-3p", "MIMAT0000078",
                "AUCACAUUGCCAGGGAUUUCCAA"),
        _preset("hsa-miR-216a-5p", "hsa-miR-216a-5p", "MIMAT0000273",
                "UAAUCUCAGCUGGCAACUGUGA"),
        _preset("hsa-miR-23b-3p", "hsa-miR-23b-3p", "MIMAT0000418",
                "AUCACAUUGCCAGGGAUUACCAC",
                note="differs from hsa-miR-23a-3p only near the 3' end; "
                     "guides against either will cross-react, and the pair is "
                     "a discrimination task, not two independent targets"),
        _preset("hsa-miR-29c-3p", "hsa-miR-29c-3p", "MIMAT0000681",
                "UAGCACCAUUUGAAAUCGGUUA",
                note="one nucleotide from hsa-miR-29a-3p across the whole "
                     "22-mer, so a guide against it is a single-mismatch "
                     "discrimination problem against an abundant relative that "
                     "is not in this preset set; load miR-29a as background if "
                     "you need the off-target call to mean anything"),
    )
}


def preset_ids() -> tuple[str, ...]:
    return tuple(MIRNA_PRESETS)


def get_preset(preset_id: str) -> TargetPreset:
    try:
        return MIRNA_PRESETS[preset_id]
    except KeyError:
        raise KeyError(
            f"unknown preset {preset_id!r}; available: {', '.join(preset_ids())}"
        ) from None


@dataclass(frozen=True, slots=True)
class Designability:
    """How many guides a target can yield, and whether it is a normal design."""

    preset_id: str
    target_length: int
    spacer_length: int
    windows: int
    canonical_length: bool
    note: str

    @property
    def designable(self) -> bool:
        return self.windows > 0


def designability(
    preset: TargetPreset, effector: Effector, spacer_length: int | None = None, step: int = 1
) -> Designability:
    """State up front what tiling this target can support.

    Called before a design run so an empty or one-row result is a predicted
    outcome rather than a mystery.
    """
    params = EFFECTORS[effector]
    length = spacer_length if spacer_length is not None else params.preferred_spacer_length
    n = len(list(windows(preset.sequence, length, step))) if length <= preset.length else 0
    canonical = params.is_canonical_length(length)

    if n == 0:
        note = (
            f"{preset.length} nt target cannot hold a {length} nt spacer. "
            f"{effector.value} tolerates {params.spacer_length_range[0]}-"
            f"{params.spacer_length_range[1]} nt; try a shorter spacer, or "
            "design against an amplicon (see amplicon_target)."
        )
    elif n < 5:
        note = (
            f"only {n} window(s): a mature miRNA is barely longer than the "
            "spacer, so there is almost nothing to choose between. An amplicon "
            "target restores normal tiling."
        )
    else:
        note = f"{n} windows available."
    if not canonical:
        note += (
            f" {length} nt is outside {effector.value}'s canonical "
            f"{params.spacer_lengths}, so activity will be reported "
            "out-of-domain (R3 REVIEW) rather than as a validated prediction."
        )
    return Designability(
        preset_id=preset.preset_id,
        target_length=preset.length,
        spacer_length=length,
        windows=n,
        canonical_length=canonical,
        note=note,
    )


def largest_designable_spacer_length(
    preset: TargetPreset, effector: Effector, min_windows: int = 1
) -> int | None:
    """Longest tolerated spacer that still yields ``min_windows`` windows.

    Longest rather than shortest because a longer spacer is the more specific
    one; there is no reason to give away duplex length that the target can
    actually accommodate. Returns ``None`` when even the shortest tolerated
    spacer does not fit.
    """
    low, high = EFFECTORS[effector].spacer_length_range
    for length in range(min(high, preset.length), low - 1, -1):
        if preset.length - length + 1 >= min_windows:
            return length
    return None


def amplicon_target(
    preset: TargetPreset,
    five_prime_context: str,
    three_prime_context: str,
    target_id: str | None = None,
    target_class: TargetClass = TargetClass.REQUIRED,
) -> TargetSpec:
    """The preset embedded in its amplification context.

    The recommended shape for a real assay: an RT-RPA product or pre-miRNA is a
    normal-length transcript, so it tiles normally, keeps the canonical spacer
    length, and stays inside the activity model's validated domain. The
    flanking context is supplied by the caller because it is a property of the
    assay design, not of the miRNA.
    """
    sequence = normalize(five_prime_context) + preset.sequence + normalize(three_prime_context)
    return TargetSpec(
        target_id=target_id or f"{preset.preset_id}__amplicon",
        sequence=sequence,
        target_class=target_class,
        gene=preset.display_name,
        biotype=preset.biotype,
    )


@dataclass(frozen=True, slots=True)
class PresetCheck:
    """What a single preset looked like against a real miRBase file."""

    preset_id: str
    accession: str
    found: bool
    matches: bool
    expected: str
    actual: str = ""

    @property
    def ok(self) -> bool:
        return self.found and self.matches

    def line(self) -> str:
        if not self.found:
            return f"{self.preset_id}: {self.accession} NOT FOUND in the file"
        if self.matches:
            return f"{self.preset_id}: ok"
        return (
            f"{self.preset_id}: MISMATCH\n"
            f"    recorded {len(self.expected):>2} nt  {self.expected}\n"
            f"    miRBase  {len(self.actual):>2} nt  {self.actual}"
        )


def _parse_mature_fa(text: str) -> dict[str, str]:
    """Accession -> sequence, from miRBase's ``mature.fa``.

    Headers look like ``>hsa-miR-21-5p MIMAT0000076 Homo sapiens miR-21-5p``.
    Keyed on the accession rather than the name because names are reused across
    releases while a MIMAT accession is stable -- which is the whole reason the
    presets record one.
    """
    out: dict[str, str] = {}
    accession = ""
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(">"):
            fields = line[1:].split()
            accession = next((f for f in fields if f.startswith("MIMAT")), "")
        elif accession:
            out[accession] = out.get(accession, "") + normalize(line)
    return out


def verify_against_mirbase(path) -> tuple[str, tuple[PresetCheck, ...]]:
    """Check every preset against a downloaded miRBase ``mature.fa``.

    Returns the file's SHA-256 and one :class:`PresetCheck` per preset. Nothing
    is downloaded and nothing is mutated: this reports, and a human decides. The
    checksum is returned rather than stored precisely because the last time this
    module held a checksum, it held one nobody had computed.

    Usage::

        digest, checks = verify_against_mirbase("mature.fa")
        for check in checks:
            print(check.line())
    """
    import hashlib
    from pathlib import Path

    data = Path(path).read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    catalogue = _parse_mature_fa(data.decode("utf-8", errors="replace"))

    checks = []
    for preset in MIRNA_PRESETS.values():
        actual = catalogue.get(preset.accession, "")
        checks.append(
            PresetCheck(
                preset_id=preset.preset_id,
                accession=preset.accession,
                found=bool(actual),
                matches=actual == preset.sequence,
                expected=preset.sequence,
                actual=actual,
            )
        )
    return digest, tuple(checks)
