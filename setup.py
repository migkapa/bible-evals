from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).parent
LONG_DESCRIPTION = (ROOT / "README.md").read_text(encoding="utf-8")


setup(
    name="bible-eval",
    version="0.1.0",
    description="Benchmark LLM verbatim fidelity to public-domain Bible texts.",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    license="MIT",
    license_files=["LICENSE"],
    package_dir={"": "src"},
    packages=find_packages("src"),
    python_requires=">=3.9",
    install_requires=[
        "PyYAML>=6.0.1",
    ],
    extras_require={
        # Consistent, fast fuzzy similarity. Without it the scorer falls back to
        # difflib, so install this extra for reproducible token-sort ratios.
        "fuzzy": ["rapidfuzz>=3.0.0"],
        "dev": [
            "pytest>=8.0.0",
            "ruff>=0.6.0",
            "rapidfuzz>=3.0.0",
        ],
    },
    entry_points={"console_scripts": ["bible-eval=bible_eval.cli:main"]},
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Text Processing :: Linguistic",
    ],
)
