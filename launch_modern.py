"""
Simple launcher for the Ultra-Modern PyQt6 AGI System
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from PyQt6.QtWidgets import QApplication
from modern_gui_pyqt6 import ModernAGIGUI

def main():
    """Launch the modern AGI system"""
    app = QApplication(sys.argv)

    # Set application properties
    app.setApplicationName("Advanced AI Self-Improvement System v2.0")
    app.setApplicationVersion("2.0")
    app.setOrganizationName("AI Research Lab")

    # Create and show the modern GUI
    window = ModernAGIGUI()
    window.show()

    # Start event loop
    sys.exit(app.exec())

if __name__ == "__main__":
    main()