"""Tests for the repository boundary guardrail (issue #113).

The guardrail is a ratchet, which makes it easy to test the wrong thing.  The
real tree is green by construction, so a checker that returned 0 unconditionally
would satisfy every helper assertion here and every other test in the repository.
Two things guard against that:

``RatchetFixtureTest`` builds throwaway git repositories that contain the
violations the policy exists to reject -- a new experiment runner, a new surface
path, an edit to a frozen file, a stale freeze entry -- and asserts the checker
fails on each.  The fixture repos are real, because the checker reads the index
and the worktree through git rather than through the filesystem.

``RepositorySatisfiesBoundaryTest`` runs the checker end-to-end against the real
repository.  That is the only invocation CI performs: the guardrail runner
``run_reproduction_guardrails.py`` is not wired into any workflow, so a boundary
test that only called the checker's helpers would never fail in CI.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "tools" / "reproduction" / "check_repository_boundary.py"

spec = importlib.util.spec_from_file_location(
    "check_repository_boundary", SCRIPT
)
assert spec is not None and spec.loader is not None
boundary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(boundary)


def _run_git(cwd: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(cwd), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


class SurfaceClassificationTest(unittest.TestCase):
    """R0's surface is an explicit inventory, so it is worth pinning exactly."""

    def test_surface_group_classification(self) -> None:
        cases = {
            "tools/relational_emergence/__init__.py": "package",
            "tools/relational_emergence/v2/run.py": "package",
            "tests/analysis/test_relational_emergence_v2.py": "tests",
            "outputs/reports/relational_emergence/README.md": "reports",
            "outputs/reports/relational_emergence/v2/development_evidence.json": "reports",
            "reproduction/adr/0005-phase1-data-generating-process.md": "decision_records",
            "reproduction/adr/0010-phase1-v2-development-boundary.md": "decision_records",
            "CONTEXT.md": "glossary",
        }
        for path, expected in cases.items():
            with self.subTest(path=path):
                self.assertEqual(boundary.surface_group(path), expected)

    def test_supported_paths_are_not_surface(self) -> None:
        """The inventory must not swallow the stable system or its neighbours."""
        cases = (
            "tools/reproduction/check_reproduction_docs.py",
            "tools/ontology_probe/README.md",
            "tests/reproduction/test_check_reproduction_docs.py",
            "tests/analysis/test_ontology_probe_metrics.py",
            # ADRs 0001-0004 document the reproduction standard, not Phase I.
            "reproduction/adr/0004-ra-sgg-predcls-v14-protocol-mismatch.md",
            "outputs/reports/ontology_probe/README.md",
        )
        for path in cases:
            with self.subTest(path=path):
                self.assertIsNone(boundary.surface_group(path))


class ShapeRuleTest(unittest.TestCase):
    def test_generated_artifact_rule(self) -> None:
        for path in (
            "outputs/reports/whatever/table.csv",
            "outputs/reports/whatever/evidence.json",
            "outputs/reports/a/b/plot.png",
        ):
            with self.subTest(path=path):
                self.assertIsNotNone(boundary.generated_artifact_finding(path))
        for path in (
            "outputs/reports/ontology_probe/README.md",
            "reproduction/evidence/artifact_manifest.json",
            "outputs/analysis/relational_emergence/phase1a/rows.json",
        ):
            with self.subTest(path=path):
                self.assertIsNone(boundary.generated_artifact_finding(path))

    def test_experiment_runner_rule(self) -> None:
        """The issue's filename globs miss run_level2.py; the tree clause catches it."""
        for path in (
            "tools/whatever/diagnose_codes.py",
            "scripts/run_phase2.py",
            "tools/whatever/summarise_phase2.py",
            "tools/whatever/relational_emergence/experiments/run_level2.py",
            "tools/whatever/relational_emergence/v3/run.py",
        ):
            with self.subTest(path=path):
                self.assertIsNotNone(boundary.experiment_runner_finding(path))

    def test_supported_tools_are_not_runners(self) -> None:
        """R3 must not fire on the supported entry points it shares a prefix with."""
        for path in (
            "tools/reproduction/run_reproduction_guardrails.py",
            "tools/reproduction/run_evidence_gate_checks.py",
            "tools/analysis/compute_collapse_metrics.py",
            "tools/relational_emergence/experiments/__init__.py",
        ):
            with self.subTest(path=path):
                self.assertIsNone(boundary.experiment_runner_finding(path))

    def test_retired_decision_record_rule(self) -> None:
        self.assertIsNotNone(
            boundary.retired_decision_record_finding("reproduction/adr/0007-emergence-evidence-criteria.md")
        )
        self.assertIsNotNone(boundary.retired_decision_record_finding("CONTEXT.md"))
        self.assertIsNone(
            boundary.retired_decision_record_finding("reproduction/adr/0003-usg-par-official-algorithm-alignment.md")
        )


class ImportRuleTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)

    def _scan(self, source: str) -> list[str]:
        path = self.tmp / "candidate.py"
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(source)
        return boundary.research_import_lines(path)

    def test_detects_import_forms(self) -> None:
        for source in (
            "import tools.relational_emergence\n",
            "from tools.relational_emergence.v2 import data\n",
            "import tools.relational_emergence.simulator.state as state\n",
            "import importlib\nimportlib.import_module('tools.relational_emergence')\n",
            "mod = __import__('tools.relational_emergence.audits')\n",
        ):
            with self.subTest(source=source.strip()):
                self.assertTrue(self._scan(source), source)

    def test_ignores_prose_and_unrelated_imports(self) -> None:
        """A docstring or comment naming the package is not an import."""
        for source in (
            "# tools.relational_emergence is discussed here\n",
            '"""See tools.relational_emergence for the research instrument."""\n',
            "from tools.ontology_probe import common\n",
            "import tools\n",
        ):
            with self.subTest(source=source.strip()):
                self.assertEqual(self._scan(source), [], source)

    def test_syntax_error_is_a_finding_not_an_exit(self) -> None:
        self.assertTrue(self._scan("def broken(:\n"))


class TempRepoTest(unittest.TestCase):
    """A throwaway git repository, so R0 is exercised through git, not mocked."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.git("init", "-q")
        self.git("config", "user.email", "boundary@example.invalid")
        self.git("config", "user.name", "Boundary Test")

    def git(self, *args: str) -> str:
        result = _run_git(self.tmp, *args)
        self.assertEqual(result.returncode, 0, result.stderr)
        return result.stdout

    def write(self, relative: str, text: str) -> Path:
        path = self.tmp / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        return path

    def commit(self) -> None:
        self.git("add", "-A")
        self.git("-c", "commit.gpgsign=false", "commit", "-q", "-m", "fixture")

    def run_checker(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, str(SCRIPT), "--root", str(self.tmp), *args],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

    def freeze_current_tree(self) -> dict:
        """Freeze whatever is committed right now, as the bootstrap would."""
        baseline = json.loads(self.run_checker("--print-baseline").stdout)
        path = self.tmp / boundary.BASELINE_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(baseline, indent=2, sort_keys=True) + "\n")
        return baseline

    def current_sha256(self, relative: str) -> str:
        blobs = boundary.tracked_blobs(self.tmp)
        oid = blobs[relative]
        return boundary.blob_digests(self.tmp, [oid])[oid][0]


class RatchetFixtureTest(TempRepoTest):
    """RED: the violations the policy exists to reject must fail the check."""

    def test_clean_surface_is_accepted(self) -> None:
        self.write("tools/relational_emergence/simulator/state.py", "x = 1\n")
        self.write("CONTEXT.md", "glossary\n")
        self.commit()
        self.freeze_current_tree()

        result = self.run_checker()

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_new_experiment_runner_is_rejected(self) -> None:
        self.write("tools/relational_emergence/simulator/state.py", "x = 1\n")
        self.commit()
        self.freeze_current_tree()
        # An experiment-only script that no supported component imports.
        self.write("tools/research/diagnose_thing.py", "print('one-off')\n")
        self.commit()

        result = self.run_checker()

        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn("R3", result.stdout)

    def test_new_surface_path_is_rejected(self) -> None:
        self.write("tools/relational_emergence/v2/run.py", "x = 1\n")
        self.commit()
        self.freeze_current_tree()
        self.write("tools/relational_emergence/v3/run.py", "x = 1\n")
        self.commit()

        result = self.run_checker()

        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn("R0 new research-surface path", result.stdout)

    def test_modifying_a_frozen_file_is_rejected(self) -> None:
        target = self.write("tools/relational_emergence/v2/run.py", "x = 1\n")
        self.commit()
        self.freeze_current_tree()
        with open(target, "w", encoding="utf-8", newline="\n") as handle:
            handle.write("x = 2\n")
        self.commit()

        result = self.run_checker()

        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn("frozen research file modified", result.stdout)

    def test_unstaged_edit_to_a_frozen_file_is_rejected(self) -> None:
        """The pre-PR check runs before the commit, which is when the edit exists."""
        target = self.write("CONTEXT.md", "glossary\n")
        self.commit()
        self.freeze_current_tree()
        with open(target, "w", encoding="utf-8", newline="\n") as handle:
            handle.write("glossary, edited\n")

        result = self.run_checker()

        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn("working tree", result.stdout)

    def test_authorized_migration_permits_the_edit(self) -> None:
        self.write("CONTEXT.md", "glossary\n")
        self.commit()
        baseline = self.freeze_current_tree()
        from_sha = baseline["frozen_surface"]["CONTEXT.md"]["sha256"]

        self.write("CONTEXT.md", "glossary, edited\n")
        self.commit()
        to_sha = self.current_sha256("CONTEXT.md")

        migration = self.tmp / boundary.MIGRATION_PATH
        migration.parent.mkdir(parents=True, exist_ok=True)
        with open(migration, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(
                json.dumps(
                    {
                        "authorized_changes": [
                            {
                                "path": "CONTEXT.md",
                                "from_sha256": from_sha,
                                "to_sha256": to_sha,
                                "reason": "recorded migration",
                                "approved_by": "test",
                                "approved_at": "2026-09-23",
                            }
                        ]
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )

        result = self.run_checker()

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("authorized migration", result.stdout)

    def test_unrecorded_edit_is_not_authorized(self) -> None:
        """An authorization for one edit must not license a different one."""
        self.write("CONTEXT.md", "glossary\n")
        self.commit()
        self.freeze_current_tree()

        self.write("CONTEXT.md", "glossary, edited once\n")
        self.commit()
        to_sha = self.current_sha256("CONTEXT.md")
        self.write("CONTEXT.md", "glossary, edited twice\n")
        self.commit()

        migration = self.tmp / boundary.MIGRATION_PATH
        migration.parent.mkdir(parents=True, exist_ok=True)
        with open(migration, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(
                json.dumps(
                    {
                        "authorized_changes": [
                            {
                                "path": "CONTEXT.md",
                                "from_sha256": "0" * 64,
                                "to_sha256": to_sha,
                                "reason": "authorizes a superseded revision only",
                                "approved_by": "test",
                                "approved_at": "2026-09-23",
                            }
                        ]
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )

        result = self.run_checker()

        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn("frozen research file modified", result.stdout)

    def test_stale_freeze_entry_fails_until_allowed(self) -> None:
        self.write("tools/relational_emergence/v2/run.py", "x = 1\n")
        self.commit()
        self.freeze_current_tree()
        self.git("rm", "-q", "tools/relational_emergence/v2/run.py")
        self.commit()

        strict = self.run_checker()
        allowed = self.run_checker("--allow-stale")

        self.assertEqual(strict.returncode, 1, strict.stdout)
        self.assertIn("stale freeze entry", strict.stdout)
        self.assertEqual(allowed.returncode, 0, allowed.stdout)

    def test_core_importing_research_is_rejected(self) -> None:
        self.write("tools/relational_emergence/simulator/state.py", "x = 1\n")
        self.commit()
        self.freeze_current_tree()
        self.write("src/exp.py", "from tools.relational_emergence import simulator\n")
        self.commit()

        result = self.run_checker()

        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn("R1", result.stdout)

    def test_missing_baseline_is_rejected(self) -> None:
        self.write("tools/relational_emergence/simulator/state.py", "x = 1\n")
        self.commit()

        result = self.run_checker()

        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn("missing baseline", result.stdout)

    def test_json_report_is_written(self) -> None:
        self.write("CONTEXT.md", "glossary\n")
        self.commit()
        self.freeze_current_tree()
        report_path = self.tmp / "report.json"

        result = self.run_checker("--json", str(report_path))

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["frozen_entries"], 1)


class RepositorySatisfiesBoundaryTest(unittest.TestCase):
    """The invocation CI performs. Everything else here could be green while this fails."""

    def test_repository_satisfies_its_own_boundary(self) -> None:
        result = subprocess.run(
            [sys.executable, str(SCRIPT)],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


class FrozenSurfaceSnapshotTest(unittest.TestCase):
    """The freeze must still describe the tree it claims to describe.

    The baseline records a blob hash per research-surface path at the
    preservation commit.  If main's content for one of those paths has drifted
    since, the preservation record no longer describes what is in the repository
    and an extraction built from it would silently carry the wrong bytes.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.baseline = json.loads(
            (REPO_ROOT / boundary.BASELINE_PATH).read_text(encoding="utf-8")
        )
        cls.frozen = cls.baseline["frozen_surface"]

    def test_preserved_commit_is_reachable_history(self) -> None:
        """The preservation record must be verifiable from the branch itself.

        Keyed on the recorded commit rather than the tag.  A tag is an alias that
        exists only once someone publishes it, so requiring it would make this
        check depend on an outward-facing step rather than on the record; the
        commit, by contrast, is an ancestor of this branch and is therefore
        present in any full clone.  That is what lets CI verify the freeze
        without the tag having been pushed.
        """
        commit = self.baseline["preserved_commit"]
        self.assertTrue(commit, "the baseline records no preserved commit")

        resolved = _run_git(REPO_ROOT, "rev-parse", "--verify", f"{commit}^{{commit}}")
        self.assertEqual(
            resolved.returncode, 0, f"preserved commit {commit} is not a commit object"
        )
        self.assertEqual(resolved.stdout.strip(), commit)

        reachable = _run_git(REPO_ROOT, "merge-base", "--is-ancestor", commit, "HEAD")
        self.assertEqual(
            reachable.returncode,
            0,
            "the preserved commit is not an ancestor of HEAD, so a full clone of "
            "this branch could not verify the freeze against it",
        )

        # The tag is a convenience alias for the same commit.  It is checked when
        # present, but not required: requiring it would gate this on publication.
        tag = self.baseline["preserved_tag"]
        if tag:
            # --verify so an absent tag is an error rather than a literal echoed
            # back on stdout, which is what bare rev-parse does with an
            # unresolvable revision.
            tagged = _run_git(REPO_ROOT, "rev-parse", "--verify", f"{tag}^{{commit}}")
            if tagged.returncode == 0:
                self.assertEqual(
                    tagged.stdout.strip(),
                    commit,
                    f"tag {tag!r} points somewhere other than the preserved commit",
                )

    def test_frozen_surface_matches_the_preserved_commit(self) -> None:
        commit = self.baseline["preserved_commit"]
        listing = _run_git(REPO_ROOT, "ls-tree", "-r", "-z", commit)
        self.assertEqual(listing.returncode, 0, listing.stderr)

        committed: dict[str, str] = {}
        for record in listing.stdout.split("\0"):
            if not record:
                continue
            meta, _, path = record.partition("\t")
            fields = meta.split()
            if len(fields) == 3 and fields[1] == "blob":
                committed[path] = fields[2]

        missing = sorted(path for path in self.frozen if path not in committed)
        self.assertEqual(missing, [], f"frozen paths absent from {commit}")

        digests = boundary.blob_digests(
            REPO_ROOT, [committed[path] for path in self.frozen]
        )
        mismatched = sorted(
            path
            for path, entry in self.frozen.items()
            if digests[committed[path]][0] != entry["sha256"]
        )
        self.assertEqual(
            mismatched,
            [],
            "the preservation record does not describe the preserved commit any more",
        )


MIGRATION_DIR = REPO_ROOT / "reproduction" / "migration"
BASELINE_PATH = REPO_ROOT / boundary.BASELINE_PATH
# Gitignored and local-only by decision (issue #113): no remote research
# repository exists, so the extraction is present only on the machine that built
# it. Tests that need it skip elsewhere; see MigrationRecordTest.
EXTRACTION = REPO_ROOT / ".research-extraction" / "Relational_Representation_Research"


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _head_blob_sha256(repo: Path) -> dict[str, str]:
    """Return ``{path: sha256}`` for every blob at ``repo``'s HEAD."""
    listing = _run_git(repo, "ls-tree", "-r", "-z", "HEAD").stdout
    blobs: dict[str, str] = {}
    for record in listing.split("\0"):
        if not record:
            continue
        meta, _, path = record.partition("\t")
        fields = meta.split()
        if len(fields) == 3 and fields[1] == "blob" and path:
            blobs[path] = fields[2]
    digests = boundary.blob_digests(repo, list(blobs.values()))
    return {path: digests[oid][0] for path, oid in blobs.items()}


class MigrationRecordTest(unittest.TestCase):
    """The records must describe the freeze, not a surface that has since moved."""

    def setUp(self) -> None:
        self.manifest = _read_json(MIGRATION_DIR / "relational_emergence_manifest.json")
        self.baseline = _read_json(BASELINE_PATH)

    def test_paths_file_selects_exactly_the_frozen_surface(self) -> None:
        """The filter-repo directive list and the freeze must not drift apart.

        If they do, either the extraction copied something the ratchet does not
        guard, or the ratchet guards something the extraction left behind.
        """
        selected = set()
        for line in (
            MIGRATION_DIR / "relational_emergence_paths.txt"
        ).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "==>" in line:
                continue
            selected.add(line)

        self.assertEqual(selected, set(self.baseline["frozen_surface"]))

    def test_manifest_agrees_with_the_freeze(self) -> None:
        self.assertEqual(
            self.manifest["preserved_commit"], self.baseline["preserved_commit"]
        )
        self.assertEqual(self.manifest["preserved_tag"], self.baseline["preserved_tag"])

        groups = self.manifest["extraction"]["selected_path_groups"]
        self.assertEqual(
            sum(len(paths) for paths in groups.values()),
            len(self.baseline["frozen_surface"]),
        )

    def test_removal_is_recorded_as_deferred(self) -> None:
        """Nothing leaves main until the extraction verifies and someone approves."""
        self.assertEqual(self.manifest["removal_from_main"]["status"], "deferred")
        self.assertFalse(
            (self.manifest["destination"].get("status") or "").startswith("created")
        )


class ExtractionRecordTest(unittest.TestCase):
    """The records must still describe the extracted tree on disk.

    These skip where the extraction is absent -- which is the case in CI, since
    the path is gitignored and there is no remote. That is a real limit worth
    stating plainly: only the machine holding the extraction re-checks these
    bytes. What CI can check is covered by FrozenSurfaceSnapshotTest, which
    verifies the freeze against the preserved commit.
    """

    @classmethod
    def setUpClass(cls) -> None:
        if not (EXTRACTION / ".git").exists():
            raise unittest.SkipTest(
                "no local extraction; it is gitignored and local-only by "
                "decision (issue #113)"
            )

    def test_transformed_manifest_describes_the_extraction(self) -> None:
        record = _read_json(MIGRATION_DIR / "relational_emergence_extracted.json")

        tip = _run_git(EXTRACTION, "rev-parse", "HEAD").stdout.strip()
        self.assertEqual(record["extracted_tip"], tip)

        actual = _head_blob_sha256(EXTRACTION)
        self.assertEqual(record["file_count"], len(actual))
        self.assertEqual(
            {entry["path"]: entry["sha256"] for entry in record["files"]},
            actual,
            "the transformed manifest no longer describes the extracted tree",
        )

    def test_extracted_tree_names_no_main_only_paths(self) -> None:
        """The extraction has to stand alone.

        The package moved to ``src/relational_research/`` and ``tests/_optional.py``
        exists here but was not among the frozen paths.  Either left unrepointed
        would make the extracted suite depend on a file that exists only in this
        repository -- the thing issue #113's exit criterion forbids.
        """
        listing = _run_git(EXTRACTION, "ls-files", "-z").stdout
        offenders: list[str] = []
        for relative in listing.split("\0"):
            if not relative:
                continue
            path = EXTRACTION / relative
            if path.suffix not in {".py", ".md", ".txt"}:
                continue
            if "tools.relational_emergence" in path.read_text(
                encoding="utf-8", errors="replace"
            ):
                offenders.append(relative)

        self.assertEqual(
            offenders,
            [],
            "the extraction still names the SGG repository's layout: "
            + ", ".join(offenders),
        )

    def test_raw_record_is_the_one_the_transformed_manifest_cites(self) -> None:
        raw_bytes = (
            MIGRATION_DIR / "relational_emergence_raw_verification.json"
        ).read_bytes()
        transformed = _read_json(MIGRATION_DIR / "relational_emergence_extracted.json")

        self.assertEqual(
            hashlib.sha256(raw_bytes).hexdigest(),
            transformed["raw_verification"]["sha256"],
            "the transformed manifest cites a different raw record than the one on disk",
        )

        raw = json.loads(raw_bytes.decode("utf-8"))
        self.assertEqual(raw["mismatched"], [])
        self.assertEqual(raw["checked"], raw["matched"])

        # The raw stage must describe an earlier state than the rewrite: it
        # compares source blobs, which rewriting deliberately invalidates.
        ancestry = _run_git(
            EXTRACTION, "merge-base", "--is-ancestor", raw["extracted_tip"], "HEAD"
        )
        self.assertEqual(
            ancestry.returncode,
            0,
            "the raw record postdates the link rewriting; it must be taken first",
        )


if __name__ == "__main__":
    unittest.main()
