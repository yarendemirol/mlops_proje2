from pathlib import Path


def test_repo_has_expected_files():
    repo_root = Path(__file__).resolve().parents[1]

    assert (repo_root / "README.md").is_file()
    assert (repo_root / "tests").is_dir()
    assert (repo_root / ".github" / "workflows" / "ci.yml").is_file()
