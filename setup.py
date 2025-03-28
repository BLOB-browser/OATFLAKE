from setuptools import setup
import os
import sys
from pathlib import Path

def find_qt_frameworks():
    """Find Qt framework paths"""
    try:
        import PyQt6
        qt_root = Path(PyQt6.__file__).parent / "Qt6/lib"
        if qt_root.exists():
            return [str(qt_root)]
        return []
    except Exception as e:
        print(f"Error finding Qt frameworks: {e}")
        return []

OPTIONS = {
    'argv_emulation': False,
    'iconfile': 'assets/icon.icns',
    'packages': [
        'PyQt6',
        'pkg_resources',
        'jaraco',
        'httpx',
        'fastapi',
        'uvicorn',
        'scripts',
        'desktop',
        'app'
    ],
    'includes': [
        'jaraco.text',
        'jaraco.collections',
        'PyQt6.QtCore',
        'PyQt6.QtGui',
        'PyQt6.QtWidgets',
    ],
    'excludes': ['tkinter', 'test', 'distutils'],
    'resources': ['assets'],
    'frameworks': find_qt_frameworks(),
    'plist': {
        'CFBundleName': "Blob",
        'CFBundleDisplayName': "Blob",
        'CFBundleIdentifier': "com.blob.app",
        'CFBundleVersion': "0.1.0",
        'CFBundleShortVersionString': "0.1.0",
        'LSMinimumSystemVersion': "10.10",
        'NSHighResolutionCapable': True,
    }
}

setup(
    name="Blob",
    version="0.1.0",
    app=['run.py'],
    data_files=[('assets', ['assets/icon.icns', 'assets/background_dark.png'])],
    options={'py2app': OPTIONS},
    setup_requires=['py2app'],
)
