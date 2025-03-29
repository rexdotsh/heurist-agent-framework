from setuptools import setup

setup(
    name="heurist-core",
    version="0.1.0",
    packages=[".", "components", "workflows", "tools", "utils", "heurist_image"],
    install_requires=[
        "openai>=1.40.8",
        "requests>=2.31.0",
        "numpy>=1.26.3",
        "scikit-learn>=1.3.2",
        "psycopg2-binary>=2.9.9",
        "smolagents>=1.9.2",
        "python-dotenv>=1.0.0",
        "pyyaml>=6.0.1",
        "tenacity>=8.5.0",
    ],
    python_requires=">=3.8",
)
