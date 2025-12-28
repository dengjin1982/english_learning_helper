#!/usr/bin/env python3
"""
English Learning Helper - GUI Application (Qt)
A graphical interface for the English Learning Helper tool using PyQt5.
"""

import os
import sys
import re
import random
import math
from datetime import datetime
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QLineEdit, QTextEdit, QFileDialog, QMessageBox,
    QGroupBox, QGridLayout, QSpinBox, QCheckBox, QStatusBar, QProgressBar,
    QSplitter, QListWidget, QListWidgetItem, QTreeWidget, QTreeWidgetItem,
    QTabWidget, QDialog, QShortcut, QComboBox
)
from PyQt5.QtGui import QClipboard
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer, QSize, QDir, QSortFilterProxyModel, QModelIndex
from PyQt5.QtWidgets import QFileSystemModel, QListView, QTreeView
from PyQt5.QtGui import QFont, QTextCharFormat, QColor, QSyntaxHighlighter, QTextCursor, QKeySequence, QIcon, QPixmap, QPainter, QPen, QBrush
from PyQt5.QtWidgets import QTextEdit

# Import the core functionality
from english_learner import (
    VocabularyLoader, SentenceGenerator, FillInBlankGenerator, WordTracker,
    process_vocabulary_files, check_new_words_and_process, clear_word_tracker
)


class SyntaxHighlighter(QSyntaxHighlighter):
    """Syntax highlighter for the results text editor."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.highlighting_rules = []
        
        # Header format (blue)
        header_format = QTextCharFormat()
        header_format.setForeground(QColor("#569cd6"))
        header_format.setFontWeight(QFont.Bold)
        self.highlighting_rules.append((r'^=+$|^SECTION \d+|^TEST|^VOCABULARY LEARNING', header_format))
        
        # Important/Duplicate format (orange/red)
        important_format = QTextCharFormat()
        important_format.setForeground(QColor("#f48771"))
        important_format.setFontWeight(QFont.Bold)
        self.highlighting_rules.append((r'\[IMPORTANT|NOTE:|appeared \d+ time', important_format))
        
        # Word format (cyan)
        word_format = QTextCharFormat()
        word_format.setForeground(QColor("#4ec9b0"))
        self.highlighting_rules.append((r'^\d+\. word:', word_format))
        
        # Explanation format (tan)
        explanation_format = QTextCharFormat()
        explanation_format.setForeground(QColor("#ce9178"))
        self.highlighting_rules.append((r'^explanation:', explanation_format))
        
        # Example format (yellow)
        example_format = QTextCharFormat()
        example_format.setForeground(QColor("#dcdcaa"))
        self.highlighting_rules.append((r'^example:|^\s+•', example_format))
        
        # Test format (purple)
        test_format = QTextCharFormat()
        test_format.setForeground(QColor("#c586c0"))
        self.highlighting_rules.append((r'^Question \d+:|^TEST -', test_format))
        
        # Answer format (green)
        answer_format = QTextCharFormat()
        answer_format.setForeground(QColor("#6a9955"))
        self.highlighting_rules.append((r'^ANSWERS:|^Question \d+: [A-Za-z]+', answer_format))
    
    def highlightBlock(self, text):
        """Apply syntax highlighting to a block of text."""
        for pattern, format in self.highlighting_rules:
            expression = re.compile(pattern, re.IGNORECASE | re.MULTILINE)
            for match in expression.finditer(text):
                start, end = match.span()
                self.setFormat(start, end - start, format)


class FileFirstSortProxy(QSortFilterProxyModel):
    """Proxy model that sorts files before directories."""
    
    def _is_directory(self, source_model, source_index):
        """Check if an item is a directory, handling different model types."""
        # Check if it's a QFileSystemModel
        if isinstance(source_model, QFileSystemModel):
            return source_model.isDir(source_index)
        
        # For other models, try to get file info from the index
        # QFileDialog uses QFileSystemModel internally, but might wrap it
        # Try to get the file path and check if it's a directory
        try:
            file_path = source_model.filePath(source_index) if hasattr(source_model, 'filePath') else None
            if file_path:
                return os.path.isdir(file_path)
        except:
            pass
        
        # Fallback: check if the item has children (directories often do)
        # This is not perfect but better than crashing
        return source_model.hasChildren(source_index)
    
    def lessThan(self, left: QModelIndex, right: QModelIndex) -> bool:
        """Sort files before directories, then alphabetically."""
        source_model = self.sourceModel()
        if not source_model:
            return False
        
        # Validate indices belong to this proxy model
        if not left.isValid() or not right.isValid():
            return False
        
        if left.model() != self or right.model() != self:
            return False
        
        try:
            left_source = self.mapToSource(left)
            right_source = self.mapToSource(right)
            
            if not left_source.isValid() or not right_source.isValid():
                return False
            
            left_is_dir = self._is_directory(source_model, left_source)
            right_is_dir = self._is_directory(source_model, right_source)
            
            # Files come before directories
            if left_is_dir != right_is_dir:
                # If left is file and right is dir, left comes first (return True)
                # If left is dir and right is file, right comes first (return False)
                return not left_is_dir
            
            # Both are files or both are directories - get data from source model
            left_data = source_model.data(left_source, Qt.DisplayRole) if isinstance(source_model, QFileSystemModel) else left.data()
            right_data = source_model.data(right_source, Qt.DisplayRole) if isinstance(source_model, QFileSystemModel) else right.data()
            
            if not left_data or not right_data:
                return False
            
            left_str = str(left_data).lower()
            right_str = str(right_data).lower()
            
            # If both are files, prioritize .html and .txt files
            if not left_is_dir and not right_is_dir:
                left_is_html_txt = left_str.endswith(('.html', '.htm', '.txt'))
                right_is_html_txt = right_str.endswith(('.html', '.htm', '.txt'))
                
                if left_is_html_txt != right_is_html_txt:
                    return left_is_html_txt  # .html/.txt files come first
            
            # Both same type, sort alphabetically
            return left_str < right_str
        except Exception:
            # If anything fails, fall back to alphabetical sorting using proxy data
            try:
                left_data = left.data()
                right_data = right.data()
                if left_data and right_data:
                    return str(left_data).lower() < str(right_data).lower()
            except:
                pass
        
        return False


class CheckNewThread(QThread):
    """Thread for checking new words and generating materials without blocking UI."""
    
    progress = pyqtSignal(str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    
    def __init__(self, input_files, output_file, examples_per_word, test_batch_size,
                 words_per_section, tracker_file, ai_provider='google', use_batch=False):
        super().__init__()
        self.input_files = input_files if isinstance(input_files, list) else [input_files]
        self.output_file = output_file
        self.examples_per_word = examples_per_word
        self.test_batch_size = test_batch_size
        self.words_per_section = words_per_section
        self.tracker_file = tracker_file
        self.ai_provider = ai_provider
        self.use_batch = use_batch
    
    def run(self):
        """Run the check-new processing in background thread."""
        try:
            result = check_new_words_and_process(
                file_paths=self.input_files,
                tracker_file=self.tracker_file,
                output=self.output_file,
                examples_per_word=self.examples_per_word,
                test_batch_size=self.test_batch_size,
                words_per_section=self.words_per_section,
                progress_callback=self.progress.emit,  # Use GUI progress signal
                ai_provider=self.ai_provider
            )
            self.finished.emit(result)
        except Exception as e:
            import traceback
            self.error.emit(f"{str(e)}\n\n{traceback.format_exc()}")


class ProcessingThread(QThread):
    """Thread for processing vocabulary file without blocking UI."""
    
    progress = pyqtSignal(str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    
    def __init__(self, input_files, output_file, examples_per_word, test_batch_size,
                 words_per_section, track_new_words, tracker_file, ai_provider='google', use_batch=False):
        super().__init__()
        self.input_files = input_files if isinstance(input_files, list) else [input_files]
        self.output_file = output_file
        self.examples_per_word = examples_per_word
        self.test_batch_size = test_batch_size
        self.words_per_section = words_per_section
        self.track_new_words = track_new_words
        self.tracker_file = tracker_file
        self.ai_provider = ai_provider
        self.use_batch = use_batch
    
    def run(self):
        """Run the processing in background thread."""
        try:
            result = self._process_files()
            self.finished.emit(result)
        except Exception as e:
            import traceback
            self.error.emit(f"{str(e)}\n\n{traceback.format_exc()}")
    
    def _process_files(self):
        """Process one or more vocabulary files using shared processing function."""
        # Use shared processing function from english_learner.py
        result = process_vocabulary_files(
            file_paths=self.input_files,
            examples_per_word=self.examples_per_word,
            output=self.output_file,
            test_batch_size=self.test_batch_size,
            words_per_section=self.words_per_section,
            track_new_words=self.track_new_words,
            tracker_file=self.tracker_file,
            progress_callback=self.progress.emit,  # Use GUI progress signal
            ai_provider=self.ai_provider,
            use_batch_processing=self.use_batch
        )
        
        # Return result in format expected by GUI
        return {
            'word_data': result['word_data'],
            'all_tests': result['all_tests'],
            'test_batches': result['test_batches'],
            'new_words': result['new_words'],
            'duplicate_words': result['duplicate_words'],
            'total_words': result['total_words'],
            'output_file': result['output']
        }
    
    # _save_results_to_file method removed - now using shared process_vocabulary_files() function


class FindDialog(QDialog):
    """Find dialog for searching text in results."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Find")
        self.setModal(False)  # Non-modal so user can still interact with main window
        self.setFixedSize(450, 140)
        
        # Style the dialog to match dark theme
        self.setStyleSheet("""
            QDialog {
                background-color: #2d2d2d;
                color: #d4d4d4;
            }
            QLabel {
                color: #d4d4d4;
            }
            QLineEdit {
                background-color: #1e1e1e;
                color: #d4d4d4;
                border: 1px solid #3e3e3e;
                padding: 4px;
            }
            QPushButton {
                background-color: #0e639c;
                color: white;
                border: none;
                padding: 6px 12px;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #1177bb;
            }
            QPushButton:pressed {
                background-color: #0a4d75;
            }
        """)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Search input
        input_layout = QHBoxLayout()
        input_layout.addWidget(QLabel("Find:"))
        self.find_input = QLineEdit()
        self.find_input.setPlaceholderText("Enter search text...")
        self.find_input.textChanged.connect(self.on_text_changed)
        self.find_input.returnPressed.connect(self.find_next)
        input_layout.addWidget(self.find_input)
        layout.addLayout(input_layout)
        
        # Status label
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #888; font-size: 9pt;")
        layout.addWidget(self.status_label)
        
        # Buttons
        button_layout = QHBoxLayout()
        self.find_next_btn = QPushButton("Find Next")
        self.find_next_btn.clicked.connect(self.find_next)
        self.find_prev_btn = QPushButton("Find Previous")
        self.find_prev_btn.clicked.connect(self.find_previous)
        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.close)
        
        button_layout.addWidget(self.find_next_btn)
        button_layout.addWidget(self.find_prev_btn)
        button_layout.addStretch()
        button_layout.addWidget(self.close_btn)
        layout.addLayout(button_layout)
        
        # Store reference to text editor
        self.text_editor = None
        self.last_search_text = ""
        self.last_position = 0
    
    def on_text_changed(self, text):
        """Handle text change in find input."""
        # Clear status when text changes
        self.status_label.setText("")
    
    def set_text_editor(self, text_editor):
        """Set the text editor to search in."""
        self.text_editor = text_editor
    
    def find_next(self):
        """Find next occurrence (case-insensitive)."""
        if not self.text_editor:
            return
        
        search_text = self.find_input.text()
        if not search_text:
            return
        
        # Get current cursor position
        cursor = self.text_editor.textCursor()
        start_pos = cursor.position()
        
        # Search forward from current position (case-insensitive)
        text = self.text_editor.toPlainText()
        text_lower = text.lower()
        search_lower = search_text.lower()
        
        found_pos = text_lower.find(search_lower, start_pos)
        
        if found_pos == -1:
            # Wrap around - search from beginning
            found_pos = text_lower.find(search_lower, 0)
            if found_pos == -1:
                # Not found anywhere
                self.status_label.setText("Not found")
                QMessageBox.information(self, "Find", f"'{search_text}' not found.")
                return
        
        if found_pos != -1:
            # Move cursor to found position
            cursor.setPosition(found_pos)
            cursor.setPosition(found_pos + len(search_text), QTextCursor.KeepAnchor)
            self.text_editor.setTextCursor(cursor)
            self.text_editor.ensureCursorVisible()
            
            # Highlight the found text
            self.highlight_found_text(found_pos, len(search_text))
            
            # Update status - count total occurrences
            total_count = text_lower.count(search_lower)
            current_num = text_lower[:found_pos].count(search_lower) + 1
            self.status_label.setText(f"Found {current_num} of {total_count}")
            
            # Keep dialog on top
            self.raise_()
            self.activateWindow()
        else:
            self.status_label.setText("Not found")
    
    def find_previous(self):
        """Find previous occurrence (case-insensitive)."""
        if not self.text_editor:
            return
        
        search_text = self.find_input.text()
        if not search_text:
            return
        
        # Get current cursor position
        cursor = self.text_editor.textCursor()
        start_pos = cursor.position() - 1
        
        # Search backward from current position (case-insensitive)
        text = self.text_editor.toPlainText()
        text_lower = text.lower()
        search_lower = search_text.lower()
        
        # Search backward up to start_pos
        found_pos = text_lower.rfind(search_lower, 0, max(0, start_pos))
        
        if found_pos == -1:
            # Wrap around - search from end
            found_pos = text_lower.rfind(search_lower)
            if found_pos == -1:
                # Not found anywhere
                self.status_label.setText("Not found")
                QMessageBox.information(self, "Find", f"'{search_text}' not found.")
                return
        
        if found_pos != -1:
            # Move cursor to found position
            cursor.setPosition(found_pos)
            cursor.setPosition(found_pos + len(search_text), QTextCursor.KeepAnchor)
            self.text_editor.setTextCursor(cursor)
            self.text_editor.ensureCursorVisible()
            
            # Highlight the found text
            self.highlight_found_text(found_pos, len(search_text))
            
            # Update status - count total occurrences
            total_count = text_lower.count(search_lower)
            current_num = text_lower[:found_pos].count(search_lower) + 1
            self.status_label.setText(f"Found {current_num} of {total_count}")
            
            # Keep dialog on top
            self.raise_()
            self.activateWindow()
        else:
            self.status_label.setText("Not found")
    
    def highlight_found_text(self, position, length):
        """Highlight the found text."""
        if not self.text_editor:
            return
        
        cursor = self.text_editor.textCursor()
        cursor.setPosition(position)
        cursor.setPosition(position + length, QTextCursor.KeepAnchor)
        
        # Create highlight format (bright yellow for find results)
        highlight_format = QTextCharFormat()
        highlight_format.setBackground(QColor("#ffeb3b"))  # Bright yellow background
        highlight_format.setForeground(QColor("#000000"))  # Black text
        highlight_format.setFontWeight(QFont.Bold)
        
        # Apply highlight using ExtraSelection
        extra_selection = QTextEdit.ExtraSelection()
        extra_selection.cursor = cursor
        extra_selection.format = highlight_format
        
        # Clear any previous find highlights and set new one
        current_selections = self.text_editor.extraSelections()
        # Keep only non-find highlights (if any), add find highlight
        # Check background color by converting QBrush to QColor
        find_selections = []
        for s in current_selections:
            bg_color = s.format.background().color()
            if bg_color.name() != "#ffeb3b":  # Not a find highlight (yellow)
                find_selections.append(s)
        find_selections.append(extra_selection)
        self.text_editor.setExtraSelections(find_selections)
        
        # Clear highlight after 3 seconds
        QTimer.singleShot(3000, lambda: self.clear_find_highlight())
    
    def clear_find_highlight(self):
        """Clear find highlights."""
        if not self.text_editor:
            return
        current_selections = self.text_editor.extraSelections()
        # Remove only find highlights (yellow background)
        # Check background color by converting QBrush to QColor
        remaining = []
        for s in current_selections:
            bg_color = s.format.background().color()
            if bg_color.name() != "#ffeb3b":  # Not a find highlight (yellow)
                remaining.append(s)
        self.text_editor.setExtraSelections(remaining)
    
    def showEvent(self, event):
        """Focus the input when dialog is shown."""
        super().showEvent(event)
        self.find_input.setFocus()
        if self.find_input.text():
            self.find_input.selectAll()
    
    def keyPressEvent(self, event):
        """Handle keyboard events."""
        if event.key() == Qt.Key_Escape:
            self.close()
        else:
            super().keyPressEvent(event)


class EnglishLearnerGUI(QMainWindow):
    """Main GUI application window."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("English Learning Helper")
        self.setGeometry(100, 100, 1400, 900)
        
        # Processing thread
        self.processing_thread = None
        
        # Navigation data
        self.section_positions = {}  # Maps section names to line numbers
        self.current_file_path = None
        
        # Highlight management
        self.highlight_timer = QTimer()
        self.highlight_timer.setSingleShot(True)
        self.highlight_timer.timeout.connect(self.clear_highlight)
        
        # Find dialog
        self.find_dialog = None
        
        # Define reusable stylesheets
        self._init_stylesheets()
        
        # Setup UI
        self.setup_ui()
        
        # Set window icon
        self.setWindowIcon(self.create_app_icon())
        
        # Timer to update current position indicator
        self.position_timer = QTimer()
        self.position_timer.timeout.connect(self.update_current_position)
        self.position_timer.start(500)  # Update every 500ms
        
        # Setup keyboard shortcuts
        self.setup_shortcuts()
        
        # Install event filter to catch Ctrl+PageUp/PageDown even when child widgets have focus
        self.installEventFilter(self)
    
    def _show_status(self, message: str, status_type: str = "info", timeout: int = 0):
        """
        Show status message with color coding.
        
        Args:
            message: The status message to display
            status_type: Type of status - "success" (green), "error" (red), "warning" (yellow), "info" (default)
            timeout: Timeout in milliseconds (0 = no timeout)
        """
        # Use HTML formatting for colors in status label
        if status_type == "success":
            colored_msg = f'<span style="color: #4CAF50; font-weight: bold;">{message}</span>'
        elif status_type == "error":
            colored_msg = f'<span style="color: #f44336; font-weight: bold;">{message}</span>'
        elif status_type == "warning":
            colored_msg = f'<span style="color: #FFC107; font-weight: bold;">{message}</span>'
        else:  # info
            colored_msg = message
        
        # Set the HTML text in the label
        self.status_label.setText(colored_msg)
        
        # If timeout is specified, reset to "Ready" after timeout
        if timeout > 0:
            QTimer.singleShot(timeout, lambda: self._show_status("Ready", "info"))
    
    def _init_stylesheets(self):
        """Initialize reusable stylesheet variables."""
        # Common disabled state for all buttons
        disabled_style = """
            QPushButton:disabled {
                background-color: #555555;
                color: #999999;
            }
        """
        
        # Primary button disabled state (needs border color)
        primary_disabled_style = """
            QPushButton:disabled {
                background-color: #555555;
                color: #999999;
                border-color: #444444;
            }
        """
        
        # Primary button style (Check New Words - most prominent)
        self.style_primary_button = f"""
            QPushButton {{
                background-color: #2196F3;
                color: white;
                font-weight: bold;
                font-size: 14pt;
                padding: 12px 20px;
                border: 2px solid #1976D2;
                border-radius: 5px;
            }}
            QPushButton:hover {{
                background-color: #1976D2;
                border-color: #1565C0;
            }}
            QPushButton:pressed {{
                background-color: #1565C0;
            }}
            {primary_disabled_style}
        """
        
        # Secondary button style (Process Vocabulary File)
        self.style_secondary_button = f"""
            QPushButton {{
                background-color: #4CAF50;
                color: white;
                font-weight: normal;
                padding: 6px;
                font-size: 10pt;
            }}
            {disabled_style}
        """
        
        # Danger button style (Clear Tracker)
        self.style_danger_button = f"""
            QPushButton {{
                background-color: #f44336;
                color: white;
                font-weight: normal;
                padding: 6px;
                font-size: 10pt;
            }}
            {disabled_style}
        """
        
        # Purple button style (Load Results)
        self.style_purple_button = f"""
            QPushButton {{
                background-color: #9C27B0;
                color: white;
                padding: 6px;
            }}
            {disabled_style}
        """
        
        # Default button style (Browse buttons)
        self.style_default_button = disabled_style
    
    def create_app_icon(self):
        """Create a fancy icon for the application - a book with a lightbulb representing learning."""
        # Create icon with multiple sizes for different uses
        icon = QIcon()
        
        # Create sizes: 16x16, 32x32, 48x48, 64x64, 128x128, 256x256
        sizes = [16, 32, 48, 64, 128, 256]
        
        for size in sizes:
            pixmap = QPixmap(size, size)
            pixmap.fill(Qt.transparent)
            
            painter = QPainter(pixmap)
            painter.setRenderHint(QPainter.Antialiasing)
            
            center = size // 2
            pen_width = max(1, size // 32)
            
            # Draw background circle with gradient effect
            radius = int(size * 0.45)
            
            # Outer circle - deep blue
            painter.setBrush(QBrush(QColor("#0e639c")))
            painter.setPen(QPen(QColor("#0a4d75"), pen_width))
            painter.drawEllipse(center - radius, center - radius, radius * 2, radius * 2)
            
            # Inner highlight - lighter blue
            inner_radius = int(radius * 0.75)
            painter.setBrush(QBrush(QColor("#1177bb")))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(center - inner_radius, center - inner_radius, inner_radius * 2, inner_radius * 2)
            
            # Draw open book
            book_width = int(size * 0.55)
            book_height = int(size * 0.4)
            book_x = center - book_width // 2
            book_y = center - book_height // 2 + int(size * 0.08)
            
            # Left page
            left_page_width = int(book_width * 0.48)
            painter.setBrush(QBrush(QColor("#ffffff")))
            painter.setPen(QPen(QColor("#d0d0d0"), pen_width))
            painter.drawRoundedRect(book_x, book_y, left_page_width, book_height, 3, 3)
            
            # Right page
            right_page_x = book_x + left_page_width + int(size * 0.02)
            right_page_width = int(book_width * 0.48)
            painter.drawRoundedRect(right_page_x, book_y, right_page_width, book_height, 3, 3)
            
            # Book spine/binding
            spine_x = book_x + left_page_width - 1
            spine_width = int(size * 0.03)
            painter.setBrush(QBrush(QColor("#4a90e2")))
            painter.setPen(Qt.NoPen)
            painter.drawRect(spine_x, book_y, spine_width, book_height)
            
            # Draw text lines on pages
            line_y_start = book_y + int(book_height * 0.2)
            line_spacing = int(book_height * 0.18)
            painter.setPen(QPen(QColor("#6ba3e3"), max(1, size // 40)))
            
            # Lines on left page
            for i in range(3):
                y = line_y_start + i * line_spacing
                line_start = book_x + int(size * 0.05)
                line_end = book_x + left_page_width - int(size * 0.05)
                painter.drawLine(line_start, y, line_end, y)
            
            # Lines on right page
            for i in range(3):
                y = line_y_start + i * line_spacing
                line_start = right_page_x + int(size * 0.05)
                line_end = right_page_x + right_page_width - int(size * 0.05)
                painter.drawLine(line_start, y, line_end, y)
            
            # Draw lightbulb above book (represents learning/insight)
            bulb_size = int(size * 0.25)
            bulb_x = center
            bulb_y = center - int(size * 0.3)
            
            # Lightbulb body (yellow/gold)
            bulb_radius = bulb_size // 2
            painter.setBrush(QBrush(QColor("#ffd700")))
            painter.setPen(QPen(QColor("#ffa500"), pen_width))
            painter.drawEllipse(bulb_x - bulb_radius, bulb_y - bulb_radius, bulb_radius * 2, bulb_radius * 2)
            
            # Lightbulb base (darker)
            base_width = int(bulb_size * 0.4)
            base_height = int(bulb_size * 0.3)
            base_x = bulb_x - base_width // 2
            base_y = bulb_y + bulb_radius - int(base_height * 0.3)
            painter.setBrush(QBrush(QColor("#ffa500")))
            painter.setPen(QPen(QColor("#cc8500"), pen_width))
            painter.drawRoundedRect(base_x, base_y, base_width, base_height, 2, 2)
            
            # Light rays (sparkles)
            ray_length = int(bulb_size * 0.3)
            painter.setPen(QPen(QColor("#ffd700"), max(1, size // 48)))
            for angle in [0, 45, 90, 135, 180, 225, 270, 315]:
                rad = math.radians(angle)
                start_x = bulb_x + int(bulb_radius * 0.7 * math.cos(rad))
                start_y = bulb_y + int(bulb_radius * 0.7 * math.sin(rad))
                end_x = bulb_x + int((bulb_radius + ray_length) * math.cos(rad))
                end_y = bulb_y + int((bulb_radius + ray_length) * math.sin(rad))
                painter.drawLine(start_x, start_y, end_x, end_y)
            
            painter.end()
            
            # Add pixmap to icon
            icon.addPixmap(pixmap)
        
        # Optionally save icon to file (for Windows .ico file)
        # Uncomment the following lines if you want to save the icon
        # self.save_icon_to_file(icon)
        
        return icon
    
    def save_icon_to_file(self, icon, filename="app_icon.ico"):
        """Save the icon to an ICO file (optional)."""
        # Get the largest pixmap (256x256)
        pixmap = icon.pixmap(256, 256)
        if pixmap:
            pixmap.save(filename, "ICO")
            print(f"Icon saved to {filename}")
    
    def setup_shortcuts(self):
        """Setup keyboard shortcuts."""
        # Ctrl+F for find
        find_shortcut = QShortcut(QKeySequence("Ctrl+F"), self)
        find_shortcut.activated.connect(self.show_find_dialog)
        
        # F3 for find next (when find dialog is open)
        find_next_shortcut = QShortcut(QKeySequence("F3"), self)
        find_next_shortcut.activated.connect(self.find_next_from_dialog)
        
        # Shift+F3 for find previous
        find_prev_shortcut = QShortcut(QKeySequence("Shift+F3"), self)
        find_prev_shortcut.activated.connect(self.find_previous_from_dialog)
        
        # Ctrl+PageUp to switch to previous tab (wraps around)
        # Try multiple approaches to ensure it works
        ctrl_pageup_shortcut = QShortcut(QKeySequence(Qt.CTRL + Qt.Key_PageUp), self)
        ctrl_pageup_shortcut.setContext(Qt.ApplicationShortcut)  # Work application-wide
        ctrl_pageup_shortcut.activated.connect(self.switch_to_previous_tab)
        
        # Ctrl+PageDown to switch to next tab (wraps around)
        ctrl_pagedown_shortcut = QShortcut(QKeySequence(Qt.CTRL + Qt.Key_PageDown), self)
        ctrl_pagedown_shortcut.setContext(Qt.ApplicationShortcut)  # Work application-wide
        ctrl_pagedown_shortcut.activated.connect(self.switch_to_next_tab)
    
    def switch_to_previous_tab(self):
        """Switch to previous tab (wraps around)."""
        current_index = self.tab_widget.currentIndex()
        previous_index = (current_index - 1) % self.tab_widget.count()
        self.tab_widget.setCurrentIndex(previous_index)
    
    def switch_to_next_tab(self):
        """Switch to next tab (wraps around)."""
        current_index = self.tab_widget.currentIndex()
        next_index = (current_index + 1) % self.tab_widget.count()
        self.tab_widget.setCurrentIndex(next_index)
    
    def eventFilter(self, obj, event):
        """Event filter to catch Ctrl+PageUp/PageDown globally."""
        if event.type() == event.KeyPress:
            # Check for Ctrl+PageUp (previous tab)
            if event.modifiers() == Qt.ControlModifier and event.key() == Qt.Key_PageUp:
                self.switch_to_previous_tab()
                return True
            # Check for Ctrl+PageDown (next tab)
            elif event.modifiers() == Qt.ControlModifier and event.key() == Qt.Key_PageDown:
                self.switch_to_next_tab()
                return True
        return super().eventFilter(obj, event)
    
    def keyPressEvent(self, event):
        """Handle keyboard events for tab navigation."""
        # Check for Ctrl+PageUp (previous tab)
        if event.modifiers() == Qt.ControlModifier and event.key() == Qt.Key_PageUp:
            self.switch_to_previous_tab()
            event.accept()
            return
        # Check for Ctrl+PageDown (next tab)
        elif event.modifiers() == Qt.ControlModifier and event.key() == Qt.Key_PageDown:
            self.switch_to_next_tab()
            event.accept()
            return
        # Let other key events be handled normally
        super().keyPressEvent(event)
    
    def show_find_dialog(self):
        """Show the find dialog."""
        if self.find_dialog is None:
            self.find_dialog = FindDialog(self)
            self.find_dialog.set_text_editor(self.results_text)
        
        self.find_dialog.show()
        self.find_dialog.raise_()
        self.find_dialog.activateWindow()
    
    def find_next_from_dialog(self):
        """Find next from dialog (F3 shortcut)."""
        if self.find_dialog and self.find_dialog.isVisible():
            self.find_dialog.find_next()
    
    def find_previous_from_dialog(self):
        """Find previous from dialog (Shift+F3 shortcut)."""
        if self.find_dialog and self.find_dialog.isVisible():
            self.find_dialog.find_previous()
        
    def setup_ui(self):
        """Set up the user interface."""
        # Central widget with tabs
        self.tab_widget = QTabWidget()
        self.setCentralWidget(self.tab_widget)
        
        # Control Panel Tab
        control_panel = self.create_control_panel()
        self.tab_widget.addTab(control_panel, "Control Panel")
        
        # Results Panel Tab
        results_panel = self.create_results_panel()
        self.tab_widget.addTab(results_panel, "Results Panel")
        
        # Tokens Panel Tab
        tokens_panel = self.create_tokens_panel()
        self.tab_widget.addTab(tokens_panel, "Tokens Panel")
        
        # Status bar with HTML-capable label
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # Create a label for status messages that supports HTML
        self.status_label = QLabel("Ready")
        self.status_label.setTextFormat(Qt.RichText)  # Enable HTML/rich text
        self.status_bar.addWidget(self.status_label)
        
        # Progress bar (hidden by default)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.status_bar.addPermanentWidget(self.progress_bar)
        
        self._show_status("Ready", "info")
        
    def create_control_panel(self):
        """Create the control panel tab."""
        control_widget = QWidget()
        control_layout = QVBoxLayout(control_widget)
        control_layout.setSpacing(5)  # Reduced spacing
        control_layout.setContentsMargins(10, 10, 10, 10)
        
        # File selection group - renamed to Control Panel
        file_group = QGroupBox("Control Panel")
        file_layout = QGridLayout()
        file_layout.setSpacing(5)
        file_layout.setContentsMargins(8, 8, 8, 8)
        file_layout.setColumnStretch(0, 0)  # Don't stretch label column
        file_layout.setColumnStretch(1, 1)  # Stretch file edit column
        file_layout.setColumnStretch(2, 0)  # Don't stretch button column
        file_layout.setColumnMinimumWidth(0, 100)  # Minimum width for labels
        
        # Input file
        file_layout.addWidget(QLabel("Input File:"), 0, 0)
        self.input_file_edit = QLineEdit()
        self.input_file_edit.setPlaceholderText("Select vocabulary file(s)... (Hold Ctrl/Cmd to select multiple)")
        file_layout.addWidget(self.input_file_edit, 0, 1)
        self.browse_input_btn = QPushButton("Browse...")
        self.browse_input_btn.setStyleSheet(self.style_default_button)
        self.browse_input_btn.clicked.connect(self.browse_input_file)
        file_layout.addWidget(self.browse_input_btn, 0, 2)
        
        # Output file
        file_layout.addWidget(QLabel("Output File:"), 1, 0)
        self.output_file_edit = QLineEdit()
        self.output_file_edit.setPlaceholderText("Auto-generated if empty...")
        file_layout.addWidget(self.output_file_edit, 1, 1)
        self.browse_output_btn = QPushButton("Browse...")
        self.browse_output_btn.setStyleSheet(self.style_default_button)
        self.browse_output_btn.clicked.connect(self.browse_output_file)
        file_layout.addWidget(self.browse_output_btn, 1, 2)
        
        # Load results file
        file_layout.addWidget(QLabel("Results File:"), 2, 0)
        self.results_file_edit = QLineEdit()
        self.results_file_edit.setPlaceholderText("Select a results file to view...")
        file_layout.addWidget(self.results_file_edit, 2, 1)
        self.browse_results_btn = QPushButton("Load Results...")
        self.browse_results_btn.setStyleSheet(self.style_purple_button)
        self.browse_results_btn.clicked.connect(self.load_results_file)
        file_layout.addWidget(self.browse_results_btn, 2, 2)
        
        file_group.setLayout(file_layout)
        control_layout.addWidget(file_group)
        
        # Options group
        options_group = QGroupBox("Options")
        options_layout = QGridLayout()
        options_layout.setSpacing(5)  # Reduced spacing
        options_layout.setContentsMargins(8, 8, 8, 8)
        options_layout.setColumnStretch(0, 0)  # Don't stretch label column
        options_layout.setColumnStretch(1, 1)  # Stretch control column
        options_layout.setColumnMinimumWidth(0, 120)  # Minimum width for labels
        options_layout.setHorizontalSpacing(10)  # Reduced horizontal spacing between columns
        
        # Examples per word
        options_layout.addWidget(QLabel("Examples per word:"), 0, 0)
        self.examples_spin = QSpinBox()
        self.examples_spin.setMinimum(1)
        self.examples_spin.setMaximum(10)
        self.examples_spin.setValue(1)
        options_layout.addWidget(self.examples_spin, 0, 1)
        
        # Test batch size
        options_layout.addWidget(QLabel("Test batch size:"), 1, 0)
        self.test_batch_spin = QSpinBox()
        self.test_batch_spin.setMinimum(1)
        self.test_batch_spin.setMaximum(100)
        self.test_batch_spin.setValue(20)
        options_layout.addWidget(self.test_batch_spin, 1, 1)
        
        # Words per section - below test batch size row
        options_layout.addWidget(QLabel("Words per section:"), 2, 0)
        self.words_section_spin = QSpinBox()
        self.words_section_spin.setMinimum(1)
        self.words_section_spin.setMaximum(100)
        self.words_section_spin.setValue(20)
        options_layout.addWidget(self.words_section_spin, 2, 1)
        
        # AI Provider - below words per section
        options_layout.addWidget(QLabel("AI Provider:"), 3, 0)
        self.ai_provider_combo = QComboBox()
        self.ai_provider_combo.addItems(["Google AI (Gemini)", "OpenAI (ChatGPT)", "DeepSeek", "Anthropic (Claude)"])
        self.ai_provider_combo.setCurrentIndex(0)  # Default to Google AI
        options_layout.addWidget(self.ai_provider_combo, 3, 1)
        
        # Batch Processing - below AI provider
        options_layout.addWidget(QLabel("Batch Processing:"), 4, 0)
        self.use_batch_checkbox = QCheckBox("Enable batch processing (faster, multiple words per API call)")
        self.use_batch_checkbox.setChecked(False)  # Default to disabled
        options_layout.addWidget(self.use_batch_checkbox, 4, 1)
        
        # Tracker file - show absolute path (read-only label) - below batch processing
        options_layout.addWidget(QLabel("Tracker file:"), 5, 0)
        # Initial tracker file path - will be updated when input file is selected
        self.tracker_file_path = ""  # Will be set when input file is selected
        self.tracker_label = QLabel("(Select input file to see tracker file path)")
        self.tracker_label.setStyleSheet("color: #888888; font-style: italic;")
        self.tracker_label.setWordWrap(True)
        options_layout.addWidget(self.tracker_label, 5, 1)
        
        options_group.setLayout(options_layout)
        control_layout.addWidget(options_group)
        
        # Main button - Check New Words (centered)
        buttons_layout = QHBoxLayout()
        buttons_layout.setSpacing(5)
        buttons_layout.addStretch()
        
        # Check New Words - Most used button, make it stand out (center)
        self.check_new_btn = QPushButton("Check New Words")
        self.check_new_btn.setStyleSheet(self.style_primary_button)
        self.check_new_btn.clicked.connect(self.check_new_words)
        buttons_layout.addWidget(self.check_new_btn)
        
        buttons_layout.addStretch()
        control_layout.addLayout(buttons_layout)
        
        # Secondary buttons at the bottom - Process and Clear Tracker
        secondary_buttons_layout = QHBoxLayout()
        secondary_buttons_layout.setSpacing(5)
        secondary_buttons_layout.addStretch()
        
        # Process Vocabulary File - Less prominent, styled like Browse button
        self.process_btn = QPushButton("Process Vocabulary File")
        self.process_btn.setStyleSheet(self.style_default_button)
        self.process_btn.clicked.connect(self.process_file)
        secondary_buttons_layout.addWidget(self.process_btn)
        
        # Clear Tracker - Least prominent, styled like Browse button
        self.clear_tracker_btn = QPushButton("Clear Tracker")
        self.clear_tracker_btn.setStyleSheet(self.style_default_button)
        self.clear_tracker_btn.clicked.connect(self.clear_tracker)
        secondary_buttons_layout.addWidget(self.clear_tracker_btn)
        
        secondary_buttons_layout.addStretch()
        control_layout.addLayout(secondary_buttons_layout)
        
        # Add stretch to push everything to top
        control_layout.addStretch()
        
        return control_widget
    
    def create_tokens_panel(self):
        """Create the tokens panel tab to display words/tokens for manual AI provider input."""
        tokens_widget = QWidget()
        tokens_layout = QVBoxLayout(tokens_widget)
        tokens_layout.setContentsMargins(10, 10, 10, 10)
        tokens_layout.setSpacing(10)
        
        # Header
        header_label = QLabel("📋 Tokens for Manual AI Provider Input")
        header_label.setStyleSheet("""
            QLabel {
                font-size: 14pt;
                font-weight: bold;
                color: #4ec9b0;
                padding: 5px;
            }
        """)
        tokens_layout.addWidget(header_label)
        
        # Info label
        info_label = QLabel("Words/tokens extracted from your input file. Copy these to feed to an AI provider manually if your API quota is reached.")
        info_label.setStyleSheet("""
            QLabel {
                color: #d4d4d4;
                font-size: 9pt;
                padding: 5px;
            }
        """)
        info_label.setWordWrap(True)
        tokens_layout.addWidget(info_label)
        
        # Text edit for tokens (read-only, but allows selection and copying)
        self.tokens_text = QTextEdit()
        self.tokens_text.setReadOnly(True)
        self.tokens_text.setPlaceholderText("No tokens available. Process a vocabulary file or check for new words to see tokens here.")
        self.tokens_text.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #d4d4d4;
                border: 1px solid #3e3e42;
                border-radius: 4px;
                padding: 10px;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 11pt;
                selection-background-color: #094771;
            }
        """)
        tokens_layout.addWidget(self.tokens_text)
        
        # Buttons layout
        buttons_layout = QHBoxLayout()
        buttons_layout.addStretch()
        
        # Copy all button
        copy_button = QPushButton("📋 Copy All")
        copy_button.setStyleSheet("""
            QPushButton {
                background-color: #0e639c;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                min-width: 100px;
            }
            QPushButton:hover {
                background-color: #1177bb;
            }
            QPushButton:pressed {
                background-color: #0a4d75;
            }
        """)
        copy_button.clicked.connect(self.copy_all_tokens)
        buttons_layout.addWidget(copy_button)
        
        tokens_layout.addLayout(buttons_layout)
        
        return tokens_widget
    
    def extract_tokens_from_file(self, file_path: str) -> list:
        """Extract words/tokens from an input file for display in Tokens Panel."""
        try:
            loader = VocabularyLoader()
            # Use load_file with fetch_pronunciations=False to speed up token extraction
            words = loader.load_file(file_path, fetch_pronunciations=False)
            return words
        except Exception as e:
            print(f"Error extracting tokens: {e}")
            return []
    
    def update_tokens_panel(self, words: list, is_check_new: bool = False, examples_per_word: int = 1):
        """Update the Tokens Panel with extracted words/tokens.
        
        Args:
            words: List of words/tokens to display
            is_check_new: If True, format for check-new (with Google AI Studio prompt template)
            examples_per_word: Number of examples per word (used in prompt template)
        """
        if not words:
            self.tokens_text.setPlainText("No tokens found.")
            return
        
        words_list = [word for word in words if word.strip()]
        
        if is_check_new:
            # For check-new, format with Google AI Studio prompt template
            self._update_tokens_panel_for_google_ai(words_list, examples_per_word)
        else:
            # Regular format: multiple formats for convenience
            # Format 1: One word per line (easy to copy individual words)
            per_line_format = "\n".join(words_list)
            
            # Format 2: Comma-separated (easy to paste into prompts)
            comma_format = ", ".join(words_list)
            
            # Format 3: Python list format
            python_list_format = "[" + ", ".join([f'"{word}"' for word in words_list]) + "]"
            
            # Format 4: JSON array format
            import json
            json_format = json.dumps(words_list, indent=2)
            
            # Combine all formats with separators
            tokens_content = f"""=== One word per line (copy individual words) ===

{per_line_format}


=== Comma-separated (paste into AI prompts) ===

{comma_format}


=== Python list format ===

{python_list_format}


=== JSON array format ===

{json_format}


=== Summary ===
Total tokens: {len(words_list)}
"""
            self.tokens_text.setPlainText(tokens_content)
        
        # Switch to Tokens Panel tab automatically
        self.tab_widget.setCurrentIndex(2)  # Tokens Panel is the 3rd tab (index 2)
    
    def _update_tokens_panel_for_google_ai(self, words: list, examples_per_word: int):
        """Format tokens with Google AI Studio prompt template for manual AI input."""
        if not words:
            self.tokens_text.setPlainText("No new tokens found.")
            return
        
        # Build prompt template based on the template in english_learner.py lines 1096-1120
        # Format for Google AI Studio - can be copied directly
        word_list_str = ", ".join([f'"{word}"' for word in words])
        
        # Create numbered word list for the prompt
        numbered_words = "\n".join([f"{i+1}. {word}" for i, word in enumerate(words)])
        
        # Build examples section template based on examples_per_word (NO numbering for examples)
        examples_template = ""
        for i in range(examples_per_word):
            examples_template += "[Example sentence showing how native speakers use this word in their daily life - naturally USED in context, not just mentioned]\n"
        examples_template = examples_template.rstrip()
        
        # Create the prompt template
        prompt_template = f"""Provide a dictionary definition (like Oxford or Google Dictionary) and {examples_per_word} example sentence(s) for each of these vocabulary words: {word_list_str}.

Requirements:
- For EACH word, provide an EXPLANATION that is a proper dictionary definition in the style of Oxford or Google Dictionary
- Each definition should be clear, precise, and explain what the word means
- **CRITICAL FOR EXAMPLES**: Each EXAMPLE must show how native English speakers use this word in their DAILY LIFE - in real conversations, everyday situations, and natural speech
- Each EXAMPLE must be a REALISTIC example of how US English native speakers actually USE the word in daily conversations and everyday life
- Examples should reflect how people actually talk in real life - at work, with friends, in stores, at home, etc.
- CRITICAL: Each word must be USED NATURALLY in its example sentence, not just mentioned or talked about
- Examples should demonstrate the word being actively used in context - like "The poverty-stricken neighborhood needs help" NOT "She's talking about poverty-stricken"
- Examples should sound like actual dialogue from TV shows (like Friends, The Office, Breaking Bad), movies, or real everyday conversations
- Make examples natural and conversational - like something a real person would say in casual conversation during their daily life
- Use contractions, casual language, and natural speech patterns when appropriate
- Each example should show the word being used as it would naturally appear in speech or writing in daily life situations
- DO NOT create examples where someone is just mentioning or discussing the word - the word must be actively used in a real daily life context

After providing the explanations and examples, also create fill-in-the-blank test questions for each word based on the example sentences. For each test question:
- Create a sentence with a blank (_____) where the word should go
- Provide the answer (the word itself)
- Provide the full original sentence

CRITICAL FORMATTING REQUIREMENTS:
1. Use periods (.) NOT commas (,) for ALL numbering - this is very important!
2. First, provide ALL word explanations and examples in one block (words numbered 1, 2, 3, etc. using PERIODS)
3. Examples should NOT have numbers - just list them one per line without numbering
4. Then, provide ALL tests in a separate block (numbered 1, 2, 3, etc. using PERIODS) - but the test order should be DIFFERENT from the word order (shuffled/randomized)
5. Finally, provide ALL answers in a separate block (numbered 1, 2, 3, etc. using PERIODS) matching the test order

Format your response EXACTLY as follows (use PERIODS (.) for numbering, NOT commas (,)):

=== SECTION 1: WORD EXPLANATIONS AND EXAMPLES ===

1. WORD: word1
EXPLANATION:
[Dictionary definition of "word1"]

EXAMPLES:
{examples_template}

2. WORD: word2
EXPLANATION:
[Dictionary definition of "word2"]

EXAMPLES:
{examples_template}

[Continue for all words in order 1, 2, 3, ... - use PERIODS (.) not commas (,) for numbering]

=== SECTION 2: TESTS (IN DIFFERENT ORDER) ===

1. Question: [Sentence with _____ where a word should go]
Full sentence: [The complete original sentence]

2. Question: [Sentence with _____ where a word should go]
Full sentence: [The complete original sentence]

[Continue for all tests - order should be DIFFERENT from the word order above - use PERIODS (.) not commas (,) for numbering]

=== SECTION 3: ANSWERS (MATCHING TEST ORDER) ===

1. Answer: [word]
2. Answer: [word]
[Continue for all answers in the same order as the tests - use PERIODS (.) not commas (,) for numbering]

REMINDER: 
- Use periods (.) for word numbering, NOT commas (,)
- Examples should NOT have numbers - just list them one per line without any numbering
- Word numbering: 1. WORD:, 2. WORD:, 3. WORD: (with periods)
- Examples format: [example sentence] (no numbering, just one per line)

Words to process ({len(words)} total):
{numbered_words}
"""
        
        self.tokens_text.setPlainText(prompt_template)
    
    def copy_all_tokens(self):
        """Copy all tokens to clipboard."""
        text = self.tokens_text.toPlainText()
        if text:
            clipboard = QApplication.clipboard()
            clipboard.setText(text)
            self._show_status("Tokens copied to clipboard", "success", 2000)
        else:
            self._show_status("No tokens to copy", "warning", 2000)
    
    def create_results_panel(self):
        """Create the results panel tab with navigation and viewer."""
        results_widget = QWidget()
        results_layout = QHBoxLayout(results_widget)
        results_layout.setContentsMargins(0, 0, 0, 0)
        results_layout.setSpacing(0)
        
        # Splitter for navigation and results
        splitter = QSplitter(Qt.Horizontal)
        
        # Navigation panel (left side)
        nav_widget = QWidget()
        nav_layout = QVBoxLayout(nav_widget)
        nav_layout.setContentsMargins(5, 5, 5, 5)
        nav_layout.setSpacing(5)
        
        nav_label = QLabel("Navigation (Bird's Eye View)")
        nav_label.setStyleSheet("font-weight: bold; font-size: 12pt; padding: 5px;")
        nav_layout.addWidget(nav_label)
        
        self.nav_tree = QTreeWidget()
        self.nav_tree.setHeaderLabel("Sections")
        self.nav_tree.setMaximumWidth(300)
        self.nav_tree.setMinimumWidth(250)
        self.nav_tree.itemClicked.connect(self.navigate_to_section)
        self.nav_tree.setStyleSheet("""
            QTreeWidget {
                background-color: #252526;
                color: #d4d4d4;
                border: 1px solid #3e3e42;
            }
            QTreeWidget::item {
                padding: 5px;
                border-bottom: 1px solid #3e3e42;
            }
            QTreeWidget::item:hover {
                background-color: #2a2d2e;
            }
            QTreeWidget::item:selected {
                background-color: #094771;
                color: white;
            }
        """)
        nav_layout.addWidget(self.nav_tree)
        
        # Current position indicator
        self.position_label = QLabel("Current Position: Not loaded")
        self.position_label.setStyleSheet("padding: 5px; background-color: #1e1e1e; color: #4ec9b0; font-weight: bold;")
        self.position_label.setWordWrap(True)
        nav_layout.addWidget(self.position_label)
        
        splitter.addWidget(nav_widget)
        
        # Results area - Text editor style (right side)
        # Text editor with syntax highlighting
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(False)
        self.results_text.setFont(QFont("Consolas", 10))
        self.results_text.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #d4d4d4;
                selection-background-color: #264f78;
                selection-color: #ffffff;
            }
        """)
        
        # Apply syntax highlighting
        self.highlighter = SyntaxHighlighter(self.results_text.document())
        
        # Connect scroll signal to update position
        self.results_text.verticalScrollBar().valueChanged.connect(self.on_scroll)
        
        splitter.addWidget(self.results_text)
        
        # Set splitter proportions (20% nav, 80% results)
        splitter.setSizes([300, 1100])
        
        results_layout.addWidget(splitter)
        
        return results_widget
        
    def browse_input_file(self):
        """Browse for input file(s) - supports multiple file selection."""
        # Default directory
        default_dir = r"C:\Users\Admin\Documents"
        
        # Ensure directory exists, fallback to current directory if not
        if not os.path.exists(default_dir):
            default_dir = os.getcwd()
        
        # Create file dialog
        dialog = QFileDialog(self, "Select Vocabulary File(s) - Hold Ctrl/Cmd to select multiple", default_dir)
        dialog.setFileMode(QFileDialog.ExistingFiles)
        dialog.setNameFilter("HTML files (*.html *.htm);;All supported (*.html *.htm *.txt *.md);;Text files (*.txt);;Markdown files (*.md);;All files (*.*)")
        dialog.setOption(QFileDialog.DontUseNativeDialog, True)
        
        # Apply dark theme styling to file dialog
        dialog.setStyleSheet("""
            QFileDialog {
                background-color: #2b2b2b;
                color: #ffffff;
            }
            QFileDialog QListView, QFileDialog QTreeView {
                background-color: #1e1e1e;
                color: #d4d4d4;
                border: 1px solid #3e3e42;
                border-radius: 4px;
                padding: 5px;
            }
            QFileDialog QListView::item, QFileDialog QTreeView::item {
                padding: 5px;
                border-radius: 3px;
                color: #d4d4d4;
            }
            QFileDialog QListView::item:hover, QFileDialog QTreeView::item:hover {
                background-color: #2a2d2e;
                color: #ffffff;
            }
            QFileDialog QListView::item:selected, QFileDialog QTreeView::item:selected {
                background-color: #094771;
                color: #ffffff;
            }
            QFileDialog QLineEdit {
                padding: 8px;
                border: 2px solid #3e3e42;
                border-radius: 4px;
                background-color: #1e1e1e;
                color: #d4d4d4;
                font-size: 11pt;
            }
            QFileDialog QLineEdit:focus {
                border-color: #007acc;
                background-color: #252526;
            }
            QFileDialog QPushButton {
                background-color: #007acc;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
                min-width: 80px;
            }
            QFileDialog QPushButton:hover {
                background-color: #005a9e;
            }
            QFileDialog QPushButton:pressed {
                background-color: #004578;
            }
            QFileDialog QComboBox {
                padding: 6px;
                border: 2px solid #3e3e42;
                border-radius: 4px;
                background-color: #1e1e1e;
                color: #d4d4d4;
            }
            QFileDialog QComboBox:focus {
                border-color: #007acc;
            }
            QFileDialog QComboBox::drop-down {
                border: none;
            }
            QFileDialog QComboBox QAbstractItemView {
                background-color: #1e1e1e;
                color: #d4d4d4;
                border: 1px solid #3e3e42;
                selection-background-color: #094771;
            }
            QFileDialog QLabel {
                color: #d4d4d4;
            }
            QFileDialog QToolButton {
                background-color: #3e3e42;
                color: #d4d4d4;
                border: 1px solid #3e3e42;
                border-radius: 3px;
            }
            QFileDialog QToolButton:hover {
                background-color: #505050;
            }
        """)
        
        if dialog.exec_() == QFileDialog.Accepted:
            filenames = dialog.selectedFiles()
        else:
            filenames = []
        if filenames:
            # Display multiple files separated by semicolon
            file_list = "; ".join(filenames)
            self.input_file_edit.setText(file_list)
            
            # Update tracker file path based on first input file's directory
            if filenames and len(filenames) > 0:
                first_file_path = filenames[0]
                input_dir = os.path.dirname(first_file_path)
                tracker_file_path = os.path.join(input_dir, "processed_words.json")
                self.tracker_file_path = tracker_file_path
                self.tracker_label.setText(tracker_file_path)
            
            if len(filenames) > 1:
                self._show_status(f"Selected {len(filenames)} files", "success", timeout=3000)
            
    def browse_output_file(self):
        """Browse for output file."""
        # Default directory
        default_dir = r"C:\Users\Admin\Documents"
        
        # Ensure directory exists, fallback to current directory if not
        if not os.path.exists(default_dir):
            default_dir = os.getcwd()
        
        # Create file dialog with files-first sorting
        dialog = QFileDialog(self, "Save Output File", default_dir)
        dialog.setFileMode(QFileDialog.AnyFile)
        dialog.setAcceptMode(QFileDialog.AcceptSave)
        dialog.setNameFilter("Text files (*.txt);;All files (*.*)")
        dialog.setOption(QFileDialog.DontUseNativeDialog, True)
        
        # Apply dark theme styling to file dialog
        dialog.setStyleSheet("""
            QFileDialog {
                background-color: #2b2b2b;
                color: #ffffff;
            }
            QFileDialog QListView, QFileDialog QTreeView {
                background-color: #1e1e1e;
                color: #d4d4d4;
                border: 1px solid #3e3e42;
                border-radius: 4px;
                padding: 5px;
            }
            QFileDialog QListView::item, QFileDialog QTreeView::item {
                padding: 5px;
                border-radius: 3px;
                color: #d4d4d4;
            }
            QFileDialog QListView::item:hover, QFileDialog QTreeView::item:hover {
                background-color: #2a2d2e;
                color: #ffffff;
            }
            QFileDialog QListView::item:selected, QFileDialog QTreeView::item:selected {
                background-color: #094771;
                color: #ffffff;
            }
            QFileDialog QLineEdit {
                padding: 8px;
                border: 2px solid #3e3e42;
                border-radius: 4px;
                background-color: #1e1e1e;
                color: #d4d4d4;
                font-size: 11pt;
            }
            QFileDialog QLineEdit:focus {
                border-color: #007acc;
                background-color: #252526;
            }
            QFileDialog QPushButton {
                background-color: #007acc;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
                min-width: 80px;
            }
            QFileDialog QPushButton:hover {
                background-color: #005a9e;
            }
            QFileDialog QPushButton:pressed {
                background-color: #004578;
            }
            QFileDialog QComboBox {
                padding: 6px;
                border: 2px solid #3e3e42;
                border-radius: 4px;
                background-color: #1e1e1e;
                color: #d4d4d4;
            }
            QFileDialog QComboBox:focus {
                border-color: #007acc;
            }
            QFileDialog QComboBox::drop-down {
                border: none;
            }
            QFileDialog QComboBox QAbstractItemView {
                background-color: #1e1e1e;
                color: #d4d4d4;
                border: 1px solid #3e3e42;
                selection-background-color: #094771;
            }
            QFileDialog QLabel {
                color: #d4d4d4;
            }
            QFileDialog QToolButton {
                background-color: #3e3e42;
                color: #d4d4d4;
                border: 1px solid #3e3e42;
                border-radius: 3px;
            }
            QFileDialog QToolButton:hover {
                background-color: #505050;
            }
        """)
        
        if dialog.exec_() == QFileDialog.Accepted:
            filename = dialog.selectedFiles()[0] if dialog.selectedFiles() else ""
        else:
            filename = ""
        if filename:
            self.output_file_edit.setText(filename)
    
    def load_results_file(self):
        """Load and display a results file."""
        # Default directory
        default_dir = r"C:\Users\Admin\Documents"
        
        # Ensure directory exists, fallback to current directory if not
        if not os.path.exists(default_dir):
            default_dir = os.getcwd()
        
        # Create file dialog with files-first sorting
        dialog = QFileDialog(self, "Load Results File", default_dir)
        dialog.setFileMode(QFileDialog.ExistingFile)
        dialog.setNameFilter("Text files (*.txt);;All files (*.*)")
        dialog.setOption(QFileDialog.DontUseNativeDialog, True)
        
        # Apply dark theme styling to file dialog
        dialog.setStyleSheet("""
            QFileDialog {
                background-color: #2b2b2b;
                color: #ffffff;
            }
            QFileDialog QListView, QFileDialog QTreeView {
                background-color: #1e1e1e;
                color: #d4d4d4;
                border: 1px solid #3e3e42;
                border-radius: 4px;
                padding: 5px;
            }
            QFileDialog QListView::item, QFileDialog QTreeView::item {
                padding: 5px;
                border-radius: 3px;
                color: #d4d4d4;
            }
            QFileDialog QListView::item:hover, QFileDialog QTreeView::item:hover {
                background-color: #2a2d2e;
                color: #ffffff;
            }
            QFileDialog QListView::item:selected, QFileDialog QTreeView::item:selected {
                background-color: #094771;
                color: #ffffff;
            }
            QFileDialog QLineEdit {
                padding: 8px;
                border: 2px solid #3e3e42;
                border-radius: 4px;
                background-color: #1e1e1e;
                color: #d4d4d4;
                font-size: 11pt;
            }
            QFileDialog QLineEdit:focus {
                border-color: #007acc;
                background-color: #252526;
            }
            QFileDialog QPushButton {
                background-color: #007acc;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
                min-width: 80px;
            }
            QFileDialog QPushButton:hover {
                background-color: #005a9e;
            }
            QFileDialog QPushButton:pressed {
                background-color: #004578;
            }
            QFileDialog QComboBox {
                padding: 6px;
                border: 2px solid #3e3e42;
                border-radius: 4px;
                background-color: #1e1e1e;
                color: #d4d4d4;
            }
            QFileDialog QComboBox:focus {
                border-color: #007acc;
            }
            QFileDialog QComboBox::drop-down {
                border: none;
            }
            QFileDialog QComboBox QAbstractItemView {
                background-color: #1e1e1e;
                color: #d4d4d4;
                border: 1px solid #3e3e42;
                selection-background-color: #094771;
            }
            QFileDialog QLabel {
                color: #d4d4d4;
            }
            QFileDialog QToolButton {
                background-color: #3e3e42;
                color: #d4d4d4;
                border: 1px solid #3e3e42;
                border-radius: 3px;
            }
            QFileDialog QToolButton:hover {
                background-color: #505050;
            }
        """)
        
        if dialog.exec_() == QFileDialog.Accepted:
            filename = dialog.selectedFiles()[0] if dialog.selectedFiles() else ""
        else:
            filename = ""
        
        if not filename:
            return
            
        try:
            self.current_file_path = filename
            self.results_file_edit.setText(filename)
            
            # Read file
            with open(filename, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Display in text editor
            self.results_text.clear()
            self.results_text.setPlainText(content)
            
            # Parse and build navigation tree
            self.build_navigation_tree(content)
            
            # Move to top
            cursor = self.results_text.textCursor()
            cursor.movePosition(QTextCursor.Start)
            self.results_text.setTextCursor(cursor)
            
            # Switch to Results Panel tab
            self.tab_widget.setCurrentIndex(1)
            
            self._show_status(f"Loaded: {os.path.basename(filename)}", "success")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load file:\n{str(e)}")
            self._show_status("Error loading file", "error")
    
    def _extract_word_count_from_content(self, content):
        """Extract total word count from file content."""
        try:
            # First check if we have a stored word count from recent processing
            if hasattr(self, '_last_word_count') and self._last_word_count > 0:
                return self._last_word_count
            
            # Look for patterns like "Total words: 100" or "Summary: ... 100 vocabulary words"
            lines = content.split('\n')
            for line in lines:
                # Pattern 1: "Total words: X"
                match = re.search(r'Total words?:\s*(\d+)', line, re.IGNORECASE)
                if match:
                    return int(match.group(1))
                
                # Pattern 2: "X vocabulary words processed"
                match = re.search(r'(\d+)\s+vocabulary\s+words?\s+processed', line, re.IGNORECASE)
                if match:
                    return int(match.group(1))
                
                # Pattern 3: "X words" in summary section
                match = re.search(r'(\d+)\s+words?', line, re.IGNORECASE)
                if match and 'summary' in line.lower():
                    return int(match.group(1))
            
            # Count unique words from section headers as fallback
            word_count = 0
            for line in lines:
                section_match = re.match(r'SECTION \d+: VOCABULARY WORDS (\d+)-(\d+)', line)
                if section_match:
                    end_word = int(section_match.group(2))
                    word_count = max(word_count, end_word)
            
            return word_count if word_count > 0 else 0
        except Exception:
            return 0
    
    def build_navigation_tree(self, content):
        """Build navigation tree from file content."""
        self.nav_tree.clear()
        self.section_positions = {}
        
        lines = content.split('\n')
        root_item = self.nav_tree.invisibleRootItem()
        
        current_section = None
        current_test = None
        current_answer = None
        
        for line_num, line in enumerate(lines, 1):
            line_stripped = line.strip()
            
            # Main header
            if "VOCABULARY LEARNING MATERIALS" in line_stripped:
                # Try to extract word count from the file
                word_count = self._extract_word_count_from_content(content)
                if word_count > 0:
                    header_text = f"📚 Main Header ({word_count} words)"
                else:
                    header_text = "📚 Main Header"
                item = QTreeWidgetItem(root_item, [header_text])
                item.setData(0, Qt.UserRole, line_num)
                self.section_positions["Main Header"] = line_num
                root_item = item
                continue
            
            # Section headers (e.g., "SECTION 1: VOCABULARY WORDS 1-20")
            section_match = re.match(r'SECTION (\d+): VOCABULARY WORDS (\d+)-(\d+)', line_stripped)
            if section_match:
                section_num = section_match.group(1)
                word_range = f"{section_match.group(2)}-{section_match.group(3)}"
                section_name = f"Section {section_num} (Words {word_range})"
                
                current_section = QTreeWidgetItem(root_item, [f"📖 {section_name}"])
                current_section.setData(0, Qt.UserRole, line_num)
                self.section_positions[section_name] = line_num
                continue
            
            # Word entries (e.g., "1. word: innovation")
            word_match = re.match(r'(\d+)\. word: ([^\[]+)', line_stripped)
            if word_match and current_section:
                word_num = word_match.group(1)
                word_name = word_match.group(2).strip()
                word_item = QTreeWidgetItem(current_section, [f"  • Word {word_num}: {word_name}"])
                word_item.setData(0, Qt.UserRole, line_num)
                self.section_positions[f"{section_name} - Word {word_num}"] = line_num
                continue
            
            # Test headers (e.g., "TEST - Section 1, Batch 1")
            test_match = re.match(r'TEST - Section (\d+), Batch (\d+)', line_stripped)
            if test_match:
                section_num = test_match.group(1)
                batch_num = test_match.group(2)
                test_name = f"Test - Section {section_num}, Batch {batch_num}"
                
                current_test = QTreeWidgetItem(root_item, [f"📝 {test_name}"])
                current_test.setData(0, Qt.UserRole, line_num)
                self.section_positions[test_name] = line_num
                current_answer = None
                continue
            
            # Answer headers
            if "ANSWERS:" in line_stripped and current_test:
                answer_name = f"{test_name} - Answers"
                current_answer = QTreeWidgetItem(current_test, [f"  ✓ Answers"])
                current_answer.setData(0, Qt.UserRole, line_num)
                self.section_positions[answer_name] = line_num
                continue
        
        # Expand all items
        self.nav_tree.expandAll()
        
        # Update position label
        self.update_current_position()
    
    def navigate_to_section(self, item, column):
        """Navigate to the selected section."""
        line_num = item.data(0, Qt.UserRole)
        if line_num:
            # Clear any existing highlight
            self.clear_highlight()
            
            cursor = self.results_text.textCursor()
            block = self.results_text.document().findBlockByLineNumber(line_num - 1)
            if block.isValid():
                # Create extra selection for highlighting
                selection = QTextEdit.ExtraSelection()
                
                # Select the entire block (line) for highlighting
                cursor.setPosition(block.position())
                cursor.movePosition(QTextCursor.EndOfBlock, QTextCursor.KeepAnchor)
                
                # Set up bright, obvious highlight format
                highlight_format = QTextCharFormat()
                highlight_format.setBackground(QColor("#FFD700"))  # Bright gold/yellow background
                highlight_format.setForeground(QColor("#000000"))  # Black text for contrast
                highlight_format.setFontWeight(QFont.Bold)
                highlight_format.setUnderlineStyle(QTextCharFormat.SingleUnderline)
                highlight_format.setUnderlineColor(QColor("#FF6B00"))  # Orange underline
                
                selection.format = highlight_format
                selection.cursor = cursor
                
                # Apply the extra selection (this creates a temporary highlight)
                self.results_text.setExtraSelections([selection])
                
                # Move cursor to the start of the line for navigation
                cursor.setPosition(block.position())
                self.results_text.setTextCursor(cursor)
                
                # Center the cursor in the viewport by scrolling
                self.results_text.ensureCursorVisible()
                scrollbar = self.results_text.verticalScrollBar()
                viewport_height = self.results_text.viewport().height()
                cursor_rect = self.results_text.cursorRect(cursor)
                # Calculate scroll position to center the cursor
                target_scroll = scrollbar.value() + cursor_rect.top() - (viewport_height // 2)
                scrollbar.setValue(max(0, target_scroll))
                
                # Set focus and start timer to clear highlight after 6 seconds
                self.results_text.setFocus()
                self.highlight_timer.start(6000)  # Clear highlight after 6 seconds
    
    def clear_highlight(self):
        """Clear the current highlight by removing extra selections."""
        self.results_text.setExtraSelections([])
    
    def on_scroll(self):
        """Handle scroll event to update position indicator."""
        self.update_current_position()
    
    def update_current_position(self):
        """Update the current position indicator based on cursor position."""
        if not self.current_file_path:
            self.position_label.setText("Current Position: No file loaded")
            return
        
        cursor = self.results_text.textCursor()
        block = cursor.block()
        line_num = block.blockNumber() + 1
        
        # Find the closest section
        closest_section = None
        closest_distance = float('inf')
        
        for section_name, section_line in self.section_positions.items():
            distance = abs(line_num - section_line)
            if distance < closest_distance:
                closest_distance = distance
                closest_section = section_name
        
        if closest_section:
            # Highlight the current section in navigation tree
            self.highlight_current_section(closest_section)
            self.position_label.setText(f"Current Position: Line {line_num}\n{closest_section}")
        else:
            self.position_label.setText(f"Current Position: Line {line_num}")
    
    def highlight_current_section(self, section_name):
        """Highlight the current section in the navigation tree."""
        # Find and highlight the item
        def find_and_highlight(item, target_name):
            item_text = item.text(0)
            if target_name in item_text or item_text.replace("📚 ", "").replace("📖 ", "").replace("📝 ", "").replace("  • ", "").replace("  ✓ ", "") == target_name:
                self.nav_tree.setCurrentItem(item)
                item.setSelected(True)
                return True
            
            for i in range(item.childCount()):
                if find_and_highlight(item.child(i), target_name):
                    return True
            return False
        
        root = self.nav_tree.invisibleRootItem()
        for i in range(root.childCount()):
            find_and_highlight(root.child(i), section_name)
    
    def _lock_ui(self):
        """Lock all UI elements during processing."""
        # Disable all buttons
        self.browse_input_btn.setEnabled(False)
        self.browse_output_btn.setEnabled(False)
        self.browse_results_btn.setEnabled(False)
        self.process_btn.setEnabled(False)
        self.check_new_btn.setEnabled(False)
        self.clear_tracker_btn.setEnabled(False)
        
        # Disable all input fields
        self.input_file_edit.setEnabled(False)
        self.output_file_edit.setEnabled(False)
        self.results_file_edit.setEnabled(False)
        # tracker_label is read-only, no need to disable
        
        # Disable all spin boxes
        self.examples_spin.setEnabled(False)
        self.words_section_spin.setEnabled(False)
        self.test_batch_spin.setEnabled(False)
        
        # Checkbox removed - no longer needed
    
    def _unlock_ui(self):
        """Unlock all UI elements after processing."""
        # Enable all buttons
        self.browse_input_btn.setEnabled(True)
        self.browse_output_btn.setEnabled(True)
        self.browse_results_btn.setEnabled(True)
        self.process_btn.setEnabled(True)
        self.check_new_btn.setEnabled(True)
        self.clear_tracker_btn.setEnabled(True)
        
        # Enable all input fields
        self.input_file_edit.setEnabled(True)
        self.output_file_edit.setEnabled(True)
        self.results_file_edit.setEnabled(True)
        # tracker_label is read-only, no need to enable
        
        # Enable all spin boxes
        self.examples_spin.setEnabled(True)
        self.words_section_spin.setEnabled(True)
        self.test_batch_spin.setEnabled(True)
        
        # Checkbox removed - no longer needed
            
    def process_file(self):
        """Process one or more vocabulary files."""
        if self.processing_thread and self.processing_thread.isRunning():
            QMessageBox.warning(self, "Processing", "A process is already running. Please wait.")
            return
            
        input_text = self.input_file_edit.text().strip()
        if not input_text:
            QMessageBox.critical(self, "Error", "Please select input file(s).")
            return
        
        # Parse multiple files (separated by semicolon)
        input_paths = [path.strip() for path in input_text.split(';') if path.strip()]
        
        if not input_paths:
            QMessageBox.critical(self, "Error", "No valid input files specified.")
            return
        
        # Helper function to find file with extensions
        def find_file(file_path):
            if os.path.exists(file_path):
                return file_path
            extensions = ['', '.html', '.htm', '.txt', '.md']
            for ext in extensions:
                test_path = file_path + ext
                if os.path.exists(test_path):
                    return test_path
            return None
        
        # Validate all files exist
        valid_paths = []
        for input_path in input_paths:
            actual_path = find_file(input_path)
            if actual_path:
                valid_paths.append(actual_path)
            else:
                QMessageBox.critical(self, "Error", f"Input file not found: {input_path}")
                return
        
        if not valid_paths:
            QMessageBox.critical(self, "Error", "No valid input files found.")
            return
        
        # Update tracker file path based on first input file's directory
        if valid_paths and len(valid_paths) > 0:
            first_file_path = valid_paths[0]
            input_dir = os.path.dirname(first_file_path)
            tracker_file_path = os.path.join(input_dir, "processed_words.json")
            self.tracker_file_path = tracker_file_path
            self.tracker_label.setText(tracker_file_path)
            
            # Extract and display tokens before processing
            try:
                all_tokens = []
                for file_path in valid_paths:
                    tokens = self.extract_tokens_from_file(file_path)
                    all_tokens.extend(tokens)
                # Remove duplicates while preserving order
                seen = set()
                unique_tokens = []
                for token in all_tokens:
                    if token not in seen:
                        seen.add(token)
                        unique_tokens.append(token)
                if unique_tokens:
                    self.update_tokens_panel(unique_tokens)
            except Exception as e:
                print(f"Warning: Could not extract tokens: {e}")
        
        # Determine output file
        output_path = self.output_file_edit.text().strip()
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # Use Documents directory for output files
            output_path = os.path.join(r"C:\Users\Admin\Documents", f"vocabulary_learning_materials_{timestamp}.txt")
            self.output_file_edit.setText(output_path)
        
        # Clear results
        self.results_text.clear()
        self.nav_tree.clear()
        self.section_positions = {}
        self.current_file_path = None
        
        self.append_text("=" * 80 + "\n", 'header')
        self.append_text("VOCABULARY LEARNING MATERIALS\n", 'header')
        self.append_text("=" * 80 + "\n\n", 'header')
        
        # Lock all UI elements
        self._lock_ui()
        
        # Show progress bar
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate
        
        # Get AI provider from combo box
        provider_map = {
            0: "google",
            1: "openai",
            2: "deepseek",
            3: "anthropic"
        }
        ai_provider = provider_map.get(self.ai_provider_combo.currentIndex(), "google")
        
        # Get batch processing setting
        use_batch = self.use_batch_checkbox.isChecked()
        
        # Start processing thread
        self.processing_thread = ProcessingThread(
            valid_paths,  # Pass list of files
            output_path,
            self.examples_spin.value(),
            self.test_batch_spin.value(),
            self.words_section_spin.value(),
            True,  # Always track new words (checkbox removed - use "Check New Words" button for new words only)
            None,  # Pass None to use same directory as input file
            ai_provider=ai_provider,
            use_batch=use_batch
        )
        self.processing_thread.progress.connect(self.on_progress)
        self.processing_thread.finished.connect(self.on_finished)
        self.processing_thread.error.connect(self.on_error)
        self.processing_thread.start()
        
    def on_progress(self, message):
        """Handle progress updates."""
        self._show_status(message, "info")
        self.append_text(f"{message}\n")
        
    def on_finished(self, result):
        """Handle processing completion."""
        self.progress_bar.setVisible(False)
        self._show_status(f"Complete! Processed {result['total_words']} words", "success")
        
        # Load the output file into results viewer
        output_file = result.get('output_file') or result.get('output')  # Support both keys
        if output_file and os.path.exists(output_file):
            self.current_file_path = output_file
            self.results_file_edit.setText(output_file)
            
            # Read and display
            with open(output_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            self.results_text.clear()
            self.results_text.setPlainText(content)
            
            # Store word count for navigation tree
            self._last_word_count = result.get('total_words', 0)
            # Build navigation
            self.build_navigation_tree(content)
            
            # Switch to Results Panel tab
            self.tab_widget.setCurrentIndex(1)
        
        # Display results summary
        self._display_results_summary(result)
        
        # Unlock all UI elements
        self._unlock_ui()
        
    def on_error(self, error_msg):
        """Handle processing errors."""
        self.progress_bar.setVisible(False)
        self._show_status("Error occurred", "error")
        self.append_text(f"\nERROR:\n{error_msg}\n", 'important')
        
        # Unlock all UI elements
        self._unlock_ui()
        
        QMessageBox.critical(self, "Error", f"An error occurred:\n{error_msg.split(chr(10))[0]}")
        
    def append_text(self, text, tag=None):
        """Append text to results area."""
        cursor = self.results_text.textCursor()
        cursor.movePosition(QTextCursor.End)
        
        # Apply formatting based on tag
        if tag == 'header':
            format = QTextCharFormat()
            format.setForeground(QColor("#569cd6"))
            format.setFontWeight(QFont.Bold)
            cursor.setCharFormat(format)
        elif tag == 'important':
            format = QTextCharFormat()
            format.setForeground(QColor("#f48771"))
            format.setFontWeight(QFont.Bold)
            cursor.setCharFormat(format)
        elif tag == 'word':
            format = QTextCharFormat()
            format.setForeground(QColor("#4ec9b0"))
            cursor.setCharFormat(format)
        elif tag == 'explanation':
            format = QTextCharFormat()
            format.setForeground(QColor("#ce9178"))
            cursor.setCharFormat(format)
        elif tag == 'example':
            format = QTextCharFormat()
            format.setForeground(QColor("#dcdcaa"))
            cursor.setCharFormat(format)
        elif tag == 'test':
            format = QTextCharFormat()
            format.setForeground(QColor("#c586c0"))
            cursor.setCharFormat(format)
        elif tag == 'answer':
            format = QTextCharFormat()
            format.setForeground(QColor("#6a9955"))
            cursor.setCharFormat(format)
        
        cursor.insertText(text)
        self.results_text.setTextCursor(cursor)
        self.results_text.ensureCursorVisible()
        
    def _display_results_summary(self, result):
        """Display results summary in the GUI."""
        word_data = result['word_data']
        all_tests = result['all_tests']
        new_words = result['new_words']
        duplicate_words = result['duplicate_words']
        
        # Summary is already in the loaded file, so we don't need to display it again
        # The navigation tree will show the structure
        pass
            
    def check_new_words(self):
        """Check for new words and generate learning materials (supports multiple files)."""
        if self.processing_thread and self.processing_thread.isRunning():
            QMessageBox.warning(self, "Processing", "A process is already running. Please wait.")
            return
            
        input_text = self.input_file_edit.text().strip()
        if not input_text:
            QMessageBox.critical(self, "Error", "Please select input file(s).")
            return
        
        # Parse multiple files (separated by semicolon)
        input_paths = [path.strip() for path in input_text.split(';') if path.strip()]
        
        if not input_paths:
            QMessageBox.critical(self, "Error", "No valid input files specified.")
            return
        
        # Helper function to find file with extensions
        def find_file(file_path):
            if os.path.exists(file_path):
                return file_path
            extensions = ['', '.html', '.htm', '.txt', '.md']
            for ext in extensions:
                test_path = file_path + ext
                if os.path.exists(test_path):
                    return test_path
            return None
        
        # Validate all files exist
        valid_paths = []
        for input_path in input_paths:
            actual_path = find_file(input_path)
            if actual_path:
                valid_paths.append(actual_path)
            else:
                QMessageBox.critical(self, "Error", f"Input file not found: {input_path}")
                return
        
        if not valid_paths:
            QMessageBox.critical(self, "Error", "No valid input files found.")
            return
        
        # Extract and display ONLY NEW tokens for check-new
        try:
            # Get tracker file path based on first input file's directory
            if valid_paths and len(valid_paths) > 0:
                first_file_path = valid_paths[0]
                input_dir = os.path.dirname(first_file_path)
                tracker_file_path = os.path.join(input_dir, "processed_words.json")
                
                # Load all words from files
                all_tokens = []
                for file_path in valid_paths:
                    tokens = self.extract_tokens_from_file(file_path)
                    all_tokens.extend(tokens)
                # Remove duplicates while preserving order
                seen = set()
                unique_tokens = []
                for token in all_tokens:
                    if token not in seen:
                        seen.add(token)
                        unique_tokens.append(token)
                
                # Get only NEW words using WordTracker
                tracker = WordTracker(tracker_file_path)
                new_words, duplicate_words = tracker.get_new_words(unique_tokens)
                
                if new_words:
                    # Format for Google AI Studio with prompt template
                    examples_per_word = self.examples_spin.value()
                    self.update_tokens_panel(new_words, is_check_new=True, examples_per_word=examples_per_word)
                else:
                    self.tokens_text.setPlainText("No new tokens found. All words have already been processed.")
                    # Still switch to tokens panel to show this message
                    self.tab_widget.setCurrentIndex(2)
        except Exception as e:
            print(f"Warning: Could not extract new tokens: {e}")
        
        # Determine output file - generate full path in Documents directory
        output_path = self.output_file_edit.text().strip()
        if not output_path:
            # Generate default filename with timestamp in Documents directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(r"C:\Users\Admin\Documents", f"vocabulary_learning_materials_checknew_{timestamp}.txt")
        
        # Clear results
        self.results_text.clear()
        self.nav_tree.clear()
        self.section_positions = {}
        self.current_file_path = None
        
        self.append_text("=" * 80 + "\n", 'header')
        self.append_text("CHECKING FOR NEW WORDS AND GENERATING MATERIALS\n", 'header')
        self.append_text("=" * 80 + "\n\n", 'header')
        
        # Lock all UI elements
        self._lock_ui()
        
        # Show progress bar
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate
        
        # Get AI provider from combo box
        provider_map = {
            0: "google",
            1: "openai",
            2: "deepseek",
            3: "anthropic"
        }
        ai_provider = provider_map.get(self.ai_provider_combo.currentIndex(), "google")
        
        # Get batch processing setting
        use_batch = self.use_batch_checkbox.isChecked()
        
        # Start processing thread
        self.processing_thread = CheckNewThread(
            valid_paths,  # Pass list of files
            output_path,
            self.examples_spin.value(),
            self.test_batch_spin.value(),
            self.words_section_spin.value(),
            None,  # Pass None to use same directory as input file
            ai_provider=ai_provider,
            use_batch=use_batch
        )
        self.processing_thread.progress.connect(self.on_progress)
        self.processing_thread.finished.connect(self.on_check_new_finished)
        self.processing_thread.error.connect(self.on_error)
        self.processing_thread.start()
    
    def on_check_new_finished(self, result):
        """Handle check-new processing completion."""
        self.progress_bar.setVisible(False)
        
        if result['total_words'] == 0:
            self._show_status("No new words found - all words already processed", "warning")
        else:
            self._show_status(f"Complete! Processed {result['total_words']} new words", "success")
            
            # Load the output file into results viewer
            output_file = result['output']
            if os.path.exists(output_file):
                self.current_file_path = output_file
                self.results_file_edit.setText(output_file)
                
                # Read and display
                with open(output_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                self.results_text.clear()
                self.results_text.setPlainText(content)
                
                # Build navigation
                self.build_navigation_tree(content)
                
                # Switch to Results Panel tab
                self.tab_widget.setCurrentIndex(1)
        
        # Unlock all UI elements
        self._unlock_ui()
            
    def clear_tracker(self):
        """Clear the tracker file using shared function."""
        reply = QMessageBox.question(
            self,
            "Confirm Clear",
            "Are you sure you want to clear the tracker file?\n\nThis will delete all tracked words.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply != QMessageBox.Yes:
            return
            
        # Get tracker file path from current input file selection
        input_text = self.input_file_edit.text().strip()
        if not input_text:
            QMessageBox.critical(self, "Error", "Please select input file(s) first to determine tracker file location.")
            return
        
        # Parse first file path
        input_paths = [path.strip() for path in input_text.split(';') if path.strip()]
        if not input_paths:
            QMessageBox.critical(self, "Error", "No valid input files specified.")
            return
        
        # Helper function to find file with extensions
        def find_file(file_path):
            if os.path.exists(file_path):
                return file_path
            extensions = ['', '.html', '.htm', '.txt', '.md']
            for ext in extensions:
                test_path = file_path + ext
                if os.path.exists(test_path):
                    return test_path
            return None
        
        first_file_path = find_file(input_paths[0])
        if not first_file_path:
            QMessageBox.critical(self, "Error", f"Input file not found: {input_paths[0]}")
            return
        
        # Determine tracker file path based on input file directory
        input_dir = os.path.dirname(first_file_path)
        tracker_file = os.path.join(input_dir, "processed_words.json")
        
        try:
            # Use shared function
            result = clear_word_tracker(tracker_file, skip_confirmation=True)
            
            if result['success']:
                self.results_text.clear()
                self.append_text(f"Tracker cleared successfully!\n", 'header')
                self.append_text(f"Removed {result['words_removed']} tracked words\n", 'important')
                self.append_text(f"File deleted: {result['tracker_file']}\n", 'important')
                
                self._show_status("Tracker cleared", "success")
            else:
                self.append_text(f"{result['message']}\n", 'important')
                self._show_status("No tracker to clear", "warning")
            
        except Exception as e:
            self.append_text(f"Error: {str(e)}\n", 'important')
            self._show_status("Error occurred", "error")
            QMessageBox.critical(self, "Error", f"An error occurred:\n{str(e)}")


def main():
    """Main entry point for GUI application."""
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    # Set dark palette
    palette = app.palette()
    palette.setColor(palette.Window, QColor(30, 30, 30))
    palette.setColor(palette.WindowText, QColor(212, 212, 212))
    palette.setColor(palette.Base, QColor(37, 37, 38))
    palette.setColor(palette.AlternateBase, QColor(45, 45, 48))
    palette.setColor(palette.ToolTipBase, QColor(0, 0, 0))
    palette.setColor(palette.ToolTipText, QColor(212, 212, 212))
    palette.setColor(palette.Text, QColor(212, 212, 212))
    palette.setColor(palette.Button, QColor(45, 45, 48))
    palette.setColor(palette.ButtonText, QColor(212, 212, 212))
    palette.setColor(palette.BrightText, QColor(255, 0, 0))
    palette.setColor(palette.Link, QColor(42, 130, 218))
    palette.setColor(palette.Highlight, QColor(42, 130, 218))
    palette.setColor(palette.HighlightedText, QColor(0, 0, 0))
    app.setPalette(palette)
    
    window = EnglishLearnerGUI()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
