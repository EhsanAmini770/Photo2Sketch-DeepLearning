# Contributing to Photo to Sketch Conversion

Thank you for your interest in contributing to this project! This document provides guidelines for contributing to the Photo to Sketch Conversion project.

## 🤝 How to Contribute

### Reporting Issues

If you find a bug or have a suggestion for improvement:

1. **Check existing issues** to avoid duplicates
2. **Create a new issue** with a clear title and description
3. **Include relevant details**:
   - Python version
   - PyTorch version
   - GPU/CPU information
   - Error messages (if any)
   - Steps to reproduce

### Submitting Changes

1. **Fork the repository**
2. **Create a feature branch** from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes** following the coding standards
4. **Test your changes** thoroughly
5. **Commit with clear messages**:
   ```bash
   git commit -m "Add: Brief description of your changes"
   ```
6. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```
7. **Create a Pull Request** with a detailed description

## 📝 Coding Standards

### Python Code Style

- Follow **PEP 8** style guidelines
- Use **type hints** where appropriate
- Add **docstrings** for functions and classes
- Keep functions **focused and small**
- Use **meaningful variable names**

### Example:

```python
def preprocess_image(img_path: str, img_size: int, device: torch.device) -> torch.Tensor:
    """
    Load and preprocess an image for model inference.
    
    Args:
        img_path: Path to the input image
        img_size: Target size for resizing
        device: PyTorch device for tensor placement
        
    Returns:
        Preprocessed image tensor
    """
    # Implementation here
    pass
```

### Documentation

- Update **README.md** if adding new features
- Add **inline comments** for complex logic
- Include **usage examples** for new functions
- Update **requirements.txt** if adding dependencies

## 🧪 Testing

Before submitting changes:

1. **Test basic functionality**:
   ```bash
   python example.py --input_photo sample_image.jpg
   ```

2. **Test training** (if applicable):
   ```bash
   python train.py --num_epochs 1  # Quick test
   ```

3. **Check imports and dependencies**:
   ```bash
   python -c "import torch, torchvision, PIL, numpy, matplotlib"
   ```

## 🎯 Areas for Contribution

### High Priority
- **Performance optimizations**
- **Memory usage improvements**
- **Better error handling**
- **Additional evaluation metrics**

### Medium Priority
- **Support for different image formats**
- **Batch processing improvements**
- **Web interface development**
- **Mobile deployment support**

### Low Priority
- **Code refactoring**
- **Documentation improvements**
- **Additional visualization tools**
- **Example notebooks**

## 🔧 Development Setup

1. **Clone your fork**:
   ```bash
   git clone https://github.com/yourusername/photo-to-sketch.git
   cd photo-to-sketch
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Install in development mode**:
   ```bash
   pip install -e .
   ```

## 📋 Pull Request Checklist

Before submitting a PR, ensure:

- [ ] Code follows project style guidelines
- [ ] All tests pass
- [ ] Documentation is updated
- [ ] Commit messages are clear
- [ ] No unnecessary files are included
- [ ] Changes are focused and atomic

## 🚀 Feature Requests

For new features:

1. **Open an issue** first to discuss the idea
2. **Explain the use case** and benefits
3. **Consider implementation complexity**
4. **Wait for maintainer feedback** before starting work

## 📞 Getting Help

If you need help:

- **Check the documentation** in `docs/`
- **Look at existing issues** for similar problems
- **Create a new issue** with the "question" label
- **Be specific** about what you're trying to achieve

## 🏆 Recognition

Contributors will be:

- **Listed in the README** (if significant contribution)
- **Mentioned in release notes**
- **Credited in academic citations** (if applicable)

## 📄 License

By contributing, you agree that your contributions will be licensed under the same MIT License that covers the project.

---

Thank you for helping make this project better! 🎉
