import subprocess
import tempfile
import unittest
import zipfile
from pathlib import Path

from chestxray8.package_project import build_zip


class PackageProjectTest(unittest.TestCase):
    def test_build_zip_excludes_data_models_and_agents(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "src").mkdir()
            (root / "data").mkdir()
            (root / "models").mkdir()
            (root / "reports").mkdir()
            (root / "README.md").write_text("# demo\n", encoding="utf-8")
            (root / ".gitignore").write_text("data/\n", encoding="utf-8")
            (root / "src" / "app.py").write_text("print('ok')\n", encoding="utf-8")
            (root / "data" / "image.png").write_bytes(b"raw")
            (root / "models" / "best_model.keras").write_bytes(b"model")
            (root / "Agents.md").write_text("local notes\n", encoding="utf-8")
            (root / "reports" / "final_report.docx").write_bytes(b"report")

            subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True)
            subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=root, check=True)
            subprocess.run(["git", "config", "user.name", "Test"], cwd=root, check=True)
            subprocess.run(["git", "add", "README.md", "src/app.py", ".gitignore"], cwd=root, check=True)
            subprocess.run(
                ["git", "add", "-f", "data/image.png", "models/best_model.keras", "Agents.md"],
                cwd=root,
                check=True,
            )
            subprocess.run(["git", "commit", "-m", "init"], cwd=root, check=True, capture_output=True)

            output = root / "dist" / "submission.zip"
            archived = build_zip(root, output, includes=["reports/final_report.docx"])

            self.assertIn("README.md", archived)
            self.assertIn(".gitignore", archived)
            self.assertIn("src/app.py", archived)
            self.assertIn("reports/final_report.docx", archived)
            self.assertIn("SUBMISSION_MANIFEST.txt", archived)
            self.assertNotIn("Agents.md", archived)
            self.assertNotIn("data/image.png", archived)
            self.assertNotIn("models/best_model.keras", archived)

            with zipfile.ZipFile(output) as zf:
                names = set(zf.namelist())
            self.assertEqual(names, set(archived))


if __name__ == "__main__":
    unittest.main()
