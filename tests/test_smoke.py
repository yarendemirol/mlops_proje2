from pathlib import Path

def test_repo_has_expected_files():
    repo_root = Path(__file__).resolve().parents[1]
    
    readme_exists = (repo_root / "README.md").is_file() or (repo_root / "readme.md").is_file()
    
    assert readme_exists, "Ana dizinde README.md dosyası bulunamadı!"
    assert (repo_root / "tests").is_dir()
