# Copyright 2018-2026
# Institute of Neuroscience and Medicine (INM-1), Forschungszentrum Jülich GmbH

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Release helper for siibra-python. Requires packaging (and tomli on Python < 3.11).

    python scripts/release.py check [--tag vX.Y.Z]  verify release metadata (also run in CI)
    python scripts/release.py prepare [BUMP]        commit the version bump
    python scripts/release.py tag                   after the release PR is merged: tag main
    python scripts/release.py sync                  rewrite README requirements from pyproject

prepare works from the current branch: on main it creates release/v<version> for the
bump, on any other branch it commits there. tag only runs on main, in sync with the
remote. Nothing is pushed: the script prints the git push command for each step.

Versions follow README.rst: X.Y.Z for releases, X.Y.Z-alpha.T and X.Y.Z-beta.T for
prereleases, and each version needs the siibra-configurations tag siibra-<version>.
An explicit version may use any PEP 440 spelling (1.1.0a0, v1.1.0-alpha.0, ...);
it is written in the form above.
BUMP is one of (default: the next prerelease of the same kind, or patch):
    alpha   1.0.1-alpha.23 -> 1.0.1-alpha.24,  1.0.1 -> 1.0.2-alpha.0
    beta    1.0.1-alpha.23 -> 1.0.1-beta.0,  1.0.1-beta.0 -> 1.0.1-beta.1
    final   1.0.1-alpha.23 or 1.0.1-beta.1 -> 1.0.1
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import subprocess
import sys
from pathlib import Path

try:
    from packaging.version import InvalidVersion, Version

    if sys.version_info >= (3, 11):
        import tomllib
    else:
        import tomli as tomllib
except ImportError as err:
    sys.exit(
        f"scripts/release.py needs the '{err.name}' library: pip install {err.name}"
    )

ROOT = Path(__file__).resolve().parent.parent
VERSION_FILE = ROOT / "siibra" / "VERSION"
CITATION_FILE = ROOT / "CITATION.cff"
CODEMETA_FILE = ROOT / "codemeta.json"
PYPROJECT_FILE = ROOT / "pyproject.toml"
README_FILE = ROOT / "README.rst"
REPO_URL = "https://github.com/FZJ-INM1-BDA/siibra-python"
CONFIG_REPO_URL = "https://github.com/FZJ-INM1-BDA/siibra-configurations"
REMOTE, BASE_BRANCH = "origin", "main"
CHECK_CMDS = [
    [sys.executable, "-m", "flake8", "./siibra", "./e2e", "./test"],
    [sys.executable, "-m", "pytest", "-q"],
]

CFF_VERSION_RE = re.compile(r"""^(version:\s*["']?)([^"'\s]+)""", re.MULTILINE)
CFF_DATE_RE = re.compile(r"""^(date-released:\s*["']?)([^"'\s]+)""", re.MULTILINE)


class ReleaseError(Exception):
    pass


def run(*cmd: str, capture: bool = False) -> str:
    if not capture:
        print("$", " ".join(cmd), flush=True)
    result = subprocess.run(
        cmd, cwd=ROOT, check=True, text=True, capture_output=capture
    )
    return result.stdout.rstrip() if capture else ""


def git(*args: str, capture: bool = False) -> str:
    return run("git", *args, capture=capture)


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write(path: Path, text: str) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as fh:  # LF, also on Windows
        fh.write(text)


# --- versions ----------------------------------------------------------------

# packaging understands every PEP 440 spelling and orders versions correctly
# (1.0.1-alpha.24 < 1.0.1-beta.0 < 1.0.1). siibra accepts only X.Y.Z, X.Y.Z-alpha.T
# and X.Y.Z-beta.T, and always writes them in that form, because the version must
# match the siibra-configurations tag siibra-<version> character for character.
PRERELEASES = {"a": "alpha", "b": "beta"}
FORM = "X.Y.Z, X.Y.Z-alpha.T or X.Y.Z-beta.T"


def parse(text: str) -> Version:
    try:
        version = Version(text)
    except InvalidVersion:
        raise ReleaseError(f"{text!r} is not a valid version") from None
    if (
        len(version.release) != 3
        or version.epoch
        or version.post is not None
        or version.dev is not None
        or version.local
        or (version.pre is not None and version.pre[0] not in PRERELEASES)
    ):
        raise ReleaseError(f"{text!r} is not of the form {FORM}")
    return version


def canonical(version: Version) -> str:
    """siibra's spelling of a version, e.g. 1.0.1-alpha.24 rather than 1.0.1a24."""
    x, y, z = version.release
    if version.pre is None:
        return f"{x}.{y}.{z}"
    kind, number = version.pre
    return f"{x}.{y}.{z}-{PRERELEASES[kind]}.{number}"


def bump(current: str, spec: str | None = None) -> str:
    cur = parse(current)
    x, y, z = cur.release
    if spec is None:
        spec = PRERELEASES[cur.pre[0]] if cur.pre else "patch"
    if spec in ("alpha", "beta"):
        kind = spec[0]
        if cur.pre and cur.pre[0] == kind:  # next number of the same kind
            new = Version(f"{x}.{y}.{z}{kind}{cur.pre[1] + 1}")
        elif cur.pre:  # alpha -> beta; beta -> alpha is rejected as not newer
            new = Version(f"{x}.{y}.{z}{kind}0")
        else:  # after a release, start prereleases of the next patch
            new = Version(f"{x}.{y}.{z + 1}{kind}0")
    elif spec == "final":
        if not cur.pre:
            raise ReleaseError(f"{current} is already a final release")
        new = Version(f"{x}.{y}.{z}")
    elif spec == "patch":
        new = Version(f"{x}.{y}.{z + 1}")
    elif spec == "minor":
        new = Version(f"{x}.{y + 1}.0")
    elif spec == "major":
        new = Version(f"{x + 1}.0.0")
    else:
        try:
            new = parse(spec)
        except ReleaseError as err:
            raise ReleaseError(
                f"{err}; use alpha, beta, final, patch, minor, major or a version"
            ) from None
    if new <= cur:
        raise ReleaseError(f"{canonical(new)} is not newer than {current}")
    return canonical(new)


# --- metadata ----------------------------------------------------------------


def config_tag_exists(version: str) -> bool:
    ref = f"refs/tags/siibra-{version}"
    return bool(git("ls-remote", "--tags", CONFIG_REPO_URL, ref, capture=True))


README_REQS_RE = re.compile(
    r"(\.\. requirements-start\n)(.*?)(\n\.\. requirements-end)", re.S
)


def readme_with_requirements() -> str:
    """README.rst text with the requirements block regenerated from pyproject.toml."""
    reqs = tomllib.loads(read(PYPROJECT_FILE)).get("project", {}).get("dependencies")
    if reqs is None:
        raise ReleaseError("pyproject.toml: [project] dependencies not found")
    readme = read(README_FILE)
    if not README_REQS_RE.search(readme):
        raise ReleaseError(
            "README.rst has no requirements-start/requirements-end markers"
        )
    block = "\n" + "\n".join(f"- {req}" for req in reqs) + "\n"
    return README_REQS_RE.sub(
        lambda m: m.group(1) + block + m.group(3), readme, count=1
    )


def sync_readme() -> bool:
    """Rewrite the README requirements from pyproject.toml; return True if changed."""
    new = readme_with_requirements()
    if new == read(README_FILE):
        return False
    write(README_FILE, new)
    return True


def check_for_problems(tag: str | None = None) -> list:
    version = read(VERSION_FILE).strip()
    try:
        expected = canonical(parse(version))
    except ReleaseError as err:
        return [f"siibra/VERSION: {err}"]
    if version != expected:
        return [f"siibra/VERSION is {version!r}; write it as {expected!r}"]
    cff = read(CITATION_FILE)
    cff_version, cff_date = (
        m.group(2) if m else None
        for m in (CFF_VERSION_RE.search(cff), CFF_DATE_RE.search(cff))
    )
    meta = json.loads(read(CODEMETA_FILE))
    checks = [
        ("CITATION.cff version", cff_version, f"v{version}"),
        ("codemeta.json version", meta.get("version"), f"v{version}"),
        ("codemeta.json datePublished", str(meta.get("datePublished"))[:10], cff_date),
        ("codemeta.json dateModified", meta.get("dateModified"), cff_date),
        (
            "codemeta.json downloadUrl",
            meta.get("downloadUrl"),
            f"{REPO_URL}/archive/refs/tags/v{version}.zip",
        ),
    ]
    if tag:
        checks.append(("tag", tag, f"v{version}"))
    found = [
        f"{label} is {actual!r}, expected {expected!r}"
        for label, actual, expected in checks
        if actual != expected
    ]
    if readme_with_requirements() != read(README_FILE):
        found.append(
            "README.rst requirements differ from pyproject.toml; "
            "run 'python scripts/release.py sync'"
        )
    if not config_tag_exists(version):
        found.append(f"siibra-configurations has no tag siibra-{version}")
    return found


def update_metadata(version: str, today: str) -> None:
    sync_readme()
    old = read(VERSION_FILE)
    write(VERSION_FILE, old.replace(old.strip(), version))

    cff = CFF_VERSION_RE.sub(rf"\g<1>v{version}", read(CITATION_FILE), count=1)
    write(CITATION_FILE, CFF_DATE_RE.sub(rf"\g<1>{today}", cff, count=1))

    meta = json.loads(read(CODEMETA_FILE))
    # Keep a time suffix such as "T00:00:00.000Z" if datePublished has one.
    meta["datePublished"] = today + str(meta.get("datePublished", ""))[10:]
    meta["dateModified"] = today
    meta["version"] = f"v{version}"
    meta["downloadUrl"] = f"{REPO_URL}/archive/refs/tags/v{version}.zip"
    write(CODEMETA_FILE, json.dumps(meta, indent=2, ensure_ascii=False) + "\n")


# --- commands ----------------------------------------------------------------


def ensure_clean() -> None:
    if changes := git("status", "--porcelain", capture=True):
        raise ReleaseError(
            f"Uncommitted changes; commit or stash them first:\n{changes}"
        )


def current_branch() -> str:
    branch = git("branch", "--show-current", capture=True)
    if not branch:
        raise ReleaseError("HEAD is detached; check out a branch first.")
    return branch


def ensure_in_sync(branch: str) -> None:
    """Fetch (read-only) and require the local branch to match the remote one."""
    git("fetch", REMOTE, "--tags")
    if git("rev-parse", branch, capture=True) != git(
        "rev-parse", f"{REMOTE}/{branch}", capture=True
    ):
        raise ReleaseError(
            f"{branch} differs from {REMOTE}/{branch}. "
            "Pull or push first so that both match."
        )


def check(tag: str | None) -> None:
    if found := check_for_problems(tag):
        raise ReleaseError(
            "Release metadata check failed:\n  - " + "\n  - ".join(found)
        )
    print("Release metadata is consistent.")


def prepare(spec: str | None) -> None:
    ensure_clean()
    here = current_branch()
    on_main = here == BASE_BRANCH
    if on_main:
        ensure_in_sync(BASE_BRANCH)
    else:
        git("fetch", REMOTE, "--tags")
    current = read(VERSION_FILE).strip()
    version = bump(current, spec)
    tag = f"v{version}"
    branch = f"release/{tag}" if on_main else here
    if not config_tag_exists(version):
        raise ReleaseError(
            f"siibra-configurations has no tag siibra-{version}. "
            "Tag the configuration first."
        )
    if git("tag", "--list", tag, capture=True):
        raise ReleaseError(f"Tag {tag} already exists.")
    if on_main:
        if git("ls-remote", "--heads", REMOTE, branch, capture=True):
            raise ReleaseError(f"{branch} already exists on {REMOTE}.")
        if git("branch", "--list", branch, capture=True):
            raise ReleaseError(
                f"{branch} already exists locally. Push it with "
                f"'git push --set-upstream {REMOTE} {branch}', "
                f"or delete it with 'git branch -D {branch}'."
            )

    print(f"\nPreparing {current} -> {version} on {branch}\n", flush=True)
    if on_main:
        git("checkout", "-b", branch)
    try:
        update_metadata(version, dt.date.today().isoformat())
        for cmd in CHECK_CMDS:
            run(*cmd)
        check(tag)
        git("add", "siibra/VERSION", "CITATION.cff", "codemeta.json", "README.rst")
        git("commit", "-m", f"Release {tag}")
    except BaseException:  # also Ctrl+C: leave the repository as it was
        print("\nAborting: discarding the changes.", file=sys.stderr)
        if on_main:
            git("checkout", "--force", BASE_BRANCH)
            git("branch", "-D", branch)
        else:  # the tree was clean before, so this only drops the script's edits
            git("reset", "--hard", "HEAD")
        raise
    print(
        f"\nCommitted the release on {branch}. Review it with 'git show', then push\n"
        f"it and open a PR into {BASE_BRANCH}:\n"
        f"    git push --set-upstream {REMOTE} {branch}\n"
        f"Once the PR is merged, run 'python scripts/release.py tag' on {BASE_BRANCH}."
    )


def tag_release() -> None:
    ensure_clean()
    if (here := current_branch()) != BASE_BRANCH:
        raise ReleaseError(
            f"Releases are only tagged on {BASE_BRANCH}, but {here} is checked out."
        )
    ensure_in_sync(BASE_BRANCH)
    tag = "v" + read(VERSION_FILE).strip()
    if git("tag", "--list", tag, capture=True):
        if git("ls-remote", "--tags", REMOTE, f"refs/tags/{tag}", capture=True):
            raise ReleaseError(
                f"{tag} already exists on {REMOTE}. Has the release PR been merged?"
            )
        raise ReleaseError(
            f"{tag} exists locally but is not pushed. Push it with "
            f"'git push {REMOTE} {tag}', or delete it with 'git tag -d {tag}'."
        )
    check(tag)
    git("tag", "--annotate", tag, "--message", f"Release {tag}")
    print(
        f"\nCreated tag {tag} on {BASE_BRANCH}. Pushing it starts the release\n"
        f"workflow, which checks the release without publishing it:\n"
        f"    git push {REMOTE} {tag}\n"
        f"Once the checks pass, publish the GitHub release, which uploads to PyPI:\n"
        f"    {REPO_URL}/releases/new?tag={tag}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="siibra-python release helper")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check").add_argument(
        "--tag", help="tag to check against, e.g. v1.0.1"
    )
    sub.add_parser("prepare").add_argument(
        "bump",
        nargs="?",
        help="alpha, beta, final, patch, minor, major or an explicit version "
        "(default: the next prerelease of the same kind, or patch)",
    )
    sub.add_parser("tag")
    sub.add_parser("sync")
    args = parser.parse_args()
    try:
        if args.command == "check":
            in_ci_on_tag = os.environ.get("GITHUB_REF_TYPE") == "tag"
            check(
                args.tag
                or (os.environ.get("GITHUB_REF_NAME") if in_ci_on_tag else None)
            )
        elif args.command == "prepare":
            prepare(args.bump)
        elif args.command == "sync":
            print(
                "README.rst updated."
                if sync_readme()
                else "README.rst already up to date."
            )
        elif args.command == "tag":
            tag_release()
        else:
            raise ReleaseError(f"Unknown command: {args.command}")
    except ReleaseError as err:
        print(f"\nError: {err}", file=sys.stderr)
        return 1
    except subprocess.CalledProcessError as err:
        print(
            f"{err.stderr or ''}\nError: command failed: {' '.join(err.cmd)}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
