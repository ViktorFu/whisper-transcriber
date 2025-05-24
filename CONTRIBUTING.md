# Contributing to Whisper Audio/Video Transcriber

🎉 Thank you for your interest in contributing to the Whisper Audio/Video Transcriber project! We welcome contributions from the community and are excited to see what you can bring to this project.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Issue Reporting](#issue-reporting)

## 🤝 Code of Conduct

This project adheres to a code of conduct that ensures a welcoming and inclusive environment for all contributors. By participating, you are expected to uphold this code.

### Our Standards

- **Be respectful** and considerate of others
- **Be collaborative** and help each other
- **Be constructive** in discussions and feedback
- **Focus on the project** and its goals
- **Accept responsibility** for mistakes and learn from them

## 🛠️ How to Contribute

There are many ways to contribute to this project:

### 🐛 Bug Reports
- Search existing issues first
- Use the bug report template
- Include system information and steps to reproduce
- Provide sample files when possible

### ✨ Feature Requests
- Check if the feature already exists or is planned
- Use the feature request template
- Explain the use case and expected behavior
- Consider implementation complexity

### 📝 Documentation
- Fix typos and improve clarity
- Add examples and use cases
- Update API documentation
- Translate documentation (if applicable)

### 💻 Code Contributions
- Bug fixes
- Performance improvements
- New features
- Code refactoring
- Test coverage improvements

## 🚀 Development Setup

### Prerequisites

- Python 3.8 or higher
- Git
- FFmpeg (for audio/video processing)
- CUDA-compatible GPU (optional, for acceleration)

### Environment Setup

1. **Fork and clone the repository**
```bash
git clone https://github.com/your-username/whisper-transcriber.git
cd whisper-transcriber
```

2. **Create a virtual environment**
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate
```

3. **Install development dependencies**
```bash
pip install -r requirements-dev.txt
```

4. **Install PyTorch with CUDA support** (optional but recommended)
```bash
# For CUDA 11.8
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 torchaudio==2.1.2+cu118 --index-url https://download.pytorch.org/whl/cu118
```

5. **Install pre-commit hooks**
```bash
pre-commit install
```

### Verify Installation

```bash
# Run basic tests
python -m pytest tests/

# Check code formatting
black --check src/
isort --check-only src/

# Run type checking
mypy src/

# Test the application
python -m src.main --help
```

## 🔄 Pull Request Process

### Before Submitting

1. **Create a feature branch**
```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

2. **Make your changes**
   - Follow coding standards
   - Add tests for new functionality
   - Update documentation as needed
   - Ensure all tests pass

3. **Commit your changes**
```bash
git add .
git commit -m "feat: add new feature description"
# Use conventional commit format
```

### Conventional Commits

We use conventional commits for clear and consistent commit messages:

- `feat:` new features
- `fix:` bug fixes
- `docs:` documentation changes
- `style:` code style changes (formatting, etc.)
- `refactor:` code refactoring
- `perf:` performance improvements
- `test:` adding or updating tests
- `chore:` maintenance tasks

Examples:
```bash
git commit -m "feat: add parallel processing for Demucs"
git commit -m "fix: resolve GPU memory leak in transcription"
git commit -m "docs: update installation instructions"
```

### Submitting the Pull Request

1. **Push your branch**
```bash
git push origin feature/your-feature-name
```

2. **Create a Pull Request**
   - Use the PR template
   - Provide clear description of changes
   - Reference related issues
   - Include screenshots/examples if applicable

3. **Address Review Feedback**
   - Respond to comments promptly
   - Make requested changes
   - Update tests and documentation

## 📏 Coding Standards

### Python Style Guide

We follow PEP 8 with some project-specific conventions:

- **Line length**: 88 characters (Black default)
- **Import formatting**: Use isort
- **Type hints**: Required for public APIs
- **Docstrings**: Google style for all public functions

### Code Quality Tools

```bash
# Format code
black src/
isort src/

# Lint code
flake8 src/
mypy src/

# Security check
bandit -r src/
```

### Project Structure

```
src/
├── core/              # Core functionality
│   ├── audio_processor.py
│   ├── transcriber.py
│   └── utils.py
├── gui/               # User interface
│   └── interface.py
└── main.py           # Application entry point
```

### Best Practices

- **Single Responsibility**: Each class/function should have one clear purpose
- **Error Handling**: Use appropriate exception handling
- **Performance**: Consider memory usage and processing speed
- **Compatibility**: Support different platforms and hardware configurations
- **Logging**: Use proper logging instead of print statements for debugging

## 🧪 Testing Guidelines

### Test Structure

```
tests/
├── unit/             # Unit tests
├── integration/      # Integration tests
├── fixtures/         # Test data and fixtures
└── conftest.py      # Pytest configuration
```

### Writing Tests

```python
import pytest
from src.core.utils import parse_time_str

def test_parse_time_str_valid_format():
    """Test parsing valid time string formats."""
    assert parse_time_str("01:23:45.678") == 5025.678
    assert parse_time_str("23:45") == 1425.0

def test_parse_time_str_invalid_format():
    """Test parsing invalid time string formats."""
    with pytest.raises(ValueError):
        parse_time_str("invalid")
```

### Running Tests

```bash
# Run all tests
python -m pytest

# Run with coverage
python -m pytest --cov=src

# Run specific test file
python -m pytest tests/unit/test_utils.py

# Run with verbose output
python -m pytest -v
```

### Test Coverage

- Aim for >90% test coverage
- Test both success and failure cases
- Include edge cases and boundary conditions
- Mock external dependencies (GPU, file system, etc.)

## 📚 Documentation

### Docstring Format

Use Google-style docstrings:

```python
def transcribe_audio(audio_path: str, model: str = "turbo") -> str:
    """
    Transcribe audio file using Whisper.
    
    Args:
        audio_path: Path to the audio file
        model: Whisper model name to use
        
    Returns:
        Path to generated SRT file
        
    Raises:
        FileNotFoundError: If audio file doesn't exist
        RuntimeError: If transcription fails
        
    Example:
        >>> srt_path = transcribe_audio("audio.mp3", "turbo")
        >>> print(f"SRT saved to: {srt_path}")
    """
```

### API Documentation

- Document all public functions and classes
- Include usage examples
- Explain parameters and return values
- Note any side effects or requirements

### README Updates

When adding new features:
- Update the feature list
- Add usage examples
- Update installation instructions if needed
- Include performance benchmarks if applicable

## 🐛 Issue Reporting

### Before Creating an Issue

1. **Search existing issues** to avoid duplicates
2. **Check the documentation** for solutions
3. **Test with the latest version**
4. **Gather system information**

### Bug Report Template

```markdown
**Bug Description**
A clear description of the bug.

**To Reproduce**
Steps to reproduce the behavior:
1. Go to '...'
2. Click on '....'
3. See error

**Expected Behavior**
What you expected to happen.

**System Information**
- OS: [e.g., Windows 10, Ubuntu 20.04]
- Python version: [e.g., 3.9.7]
- PyTorch version: [e.g., 2.1.2+cu118]
- GPU: [e.g., RTX 4060 8GB]

**Additional Context**
Add any other context about the problem here.
```

### Feature Request Template

```markdown
**Feature Description**
A clear description of the feature you'd like to see.

**Use Case**
Explain why this feature would be useful.

**Proposed Implementation**
If you have ideas about how to implement this feature.

**Additional Context**
Any other context or screenshots about the feature.
```

## 🏷️ Release Process

### Version Numbering

We follow semantic versioning (SemVer):
- `MAJOR.MINOR.PATCH`
- `MAJOR`: Breaking changes
- `MINOR`: New features (backward compatible)
- `PATCH`: Bug fixes (backward compatible)

### Release Checklist

- [ ] Update version in `setup.py`
- [ ] Update `CHANGELOG.md`
- [ ] Run full test suite
- [ ] Update documentation
- [ ] Create release notes
- [ ] Tag the release
- [ ] Build and publish packages

## 🙏 Recognition

Contributors will be recognized in:
- `README.md` contributors section
- Release notes
- Project documentation

Thank you for contributing to making audio transcription accessible and powerful for everyone! 🎉

---

**Questions?** Feel free to open an issue or reach out to the maintainers. We're here to help! 💬 