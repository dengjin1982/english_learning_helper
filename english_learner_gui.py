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
    QTabWidget, QDialog, QShortcut
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer, QSize
from PyQt5.QtGui import QFont, QTextCharFormat, QColor, QSyntaxHighlighter, QTextCursor, QKeySequence, QIcon, QPixmap, QPainter, QPen, QBrush
from PyQt5.QtWidgets import QTextEdit

# Import the core functionality
from english_learner import (
    VocabularyLoader, SentenceGenerator, FillInBlankGenerator, WordTracker
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


class ProcessingThread(QThread):
    """Thread for processing vocabulary file without blocking UI."""
    
    progress = pyqtSignal(str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    
    def __init__(self, input_files, output_file, examples_per_word, test_batch_size,
                 words_per_section, track_new_words, tracker_file):
        super().__init__()
        self.input_files = input_files if isinstance(input_files, list) else [input_files]
        self.output_file = output_file
        self.examples_per_word = examples_per_word
        self.test_batch_size = test_batch_size
        self.words_per_section = words_per_section
        self.track_new_words = track_new_words
        self.tracker_file = tracker_file
        
    def run(self):
        """Run the processing in background thread."""
        try:
            result = self._process_files()
            self.finished.emit(result)
        except Exception as e:
            import traceback
            self.error.emit(f"{str(e)}\n\n{traceback.format_exc()}")
    
    def _process_files(self):
        """Process one or more vocabulary files."""
        if len(self.input_files) > 1:
            self.progress.emit(f"Loading {len(self.input_files)} vocabulary files...")
        else:
            self.progress.emit("Loading vocabulary file...")
        
        loader = VocabularyLoader()
        all_words = []
        
        # Helper function to find file with extensions
        def find_file(file_path):
            import os
            if os.path.exists(file_path):
                return file_path
            extensions = ['', '.html', '.htm', '.txt', '.md']
            for ext in extensions:
                test_path = file_path + ext
                if os.path.exists(test_path):
                    return test_path
            return None
        
        # Load words from all files
        for input_file in self.input_files:
            actual_path = find_file(input_file)
            if not actual_path:
                raise Exception(f"File not found: {input_file}")
            
            words_from_file = loader.load_file(actual_path)
            all_words.extend(words_from_file)
            if len(self.input_files) > 1:
                self.progress.emit(f"Loaded {len(words_from_file)} words from {os.path.basename(actual_path)}")
        
        # Remove duplicates while preserving order
        seen = set()
        unique_words = []
        for word in all_words:
            if word.lower() not in seen:
                seen.add(word.lower())
                unique_words.append(word)
        
        all_words = unique_words
        
        if not all_words:
            raise Exception("No vocabulary words found in file(s).")
        
        if len(self.input_files) > 1:
            self.progress.emit(f"Total unique words from {len(self.input_files)} files: {len(all_words)}")
        
        # Track new words if enabled
        words_to_process = all_words
        duplicate_words = []
        new_words = []
        tracker = None
        
        if self.track_new_words:
            tracker = WordTracker(self.tracker_file)
            new_words, duplicate_words = tracker.get_new_words(all_words)
            words_to_process = new_words + duplicate_words
            
            if len(new_words) == 0 and len(duplicate_words) == 0:
                raise Exception("No new words found! All words have already been processed.")
        
        total_words = len(words_to_process)
        self.progress.emit(f"Generating explanations for {total_words} words...")
        
        # Generate explanations and examples
        generator = SentenceGenerator()
        word_data = {}
        duplicate_set = set(w.lower() for w in duplicate_words)
        
        for idx, word in enumerate(words_to_process, 1):
            try:
                is_duplicate = word.lower() in duplicate_set
                
                if is_duplicate and tracker:
                    occurrence_count = tracker.get_occurrence_count(word)
                    data = generator.generate_explanation_and_examples(word, self.examples_per_word, is_repeat=True)
                    data['is_important'] = True
                    data['occurrence_count'] = occurrence_count
                else:
                    data = generator.generate_explanation_and_examples(word, self.examples_per_word)
                    data['is_important'] = False
                
                word_data[word] = data
                
                if idx % 10 == 0 or idx == total_words:
                    self.progress.emit(f"Processing... {idx}/{total_words} words")
                    
            except Exception as e:
                self.progress.emit(f"WARNING: Failed to generate for '{word}': {e}")
                is_duplicate = word.lower() in duplicate_set
                data = generator._generate_mock_explanation_and_examples(word, self.examples_per_word, is_repeat=is_duplicate)
                data['is_important'] = is_duplicate
                if is_duplicate and tracker:
                    data['occurrence_count'] = tracker.get_occurrence_count(word)
                word_data[word] = data
        
        self.progress.emit("Creating fill-in-the-blank tests...")
        
        # Create tests
        test_generator = FillInBlankGenerator()
        word_sentences = {word: data['examples'] for word, data in word_data.items()}
        all_tests = test_generator.create_tests(word_sentences)
        test_batches = test_generator.split_tests_into_batches(all_tests, self.test_batch_size)
        
        # Save to file
        self.progress.emit(f"Saving results to: {self.output_file}")
        self._save_results_to_file(
            self.output_file, word_data, all_tests, test_batches,
            total_words, new_words, duplicate_words, words_to_process
        )
        
        # Update tracker
        if self.track_new_words:
            tracker.add_words(words_to_process)
            tracker.save()
        
        return {
            'word_data': word_data,
            'all_tests': all_tests,
            'test_batches': test_batches,
            'new_words': new_words,
            'duplicate_words': duplicate_words,
            'total_words': total_words,
            'output_file': self.output_file
        }
    
    def _save_results_to_file(self, output_path, word_data, all_tests, test_batches,
                              total_words, new_words, duplicate_words, words_to_process):
        """Save results to file."""
        # Preserve original file order - create word list in the same order as words_to_process
        word_list = []
        for word in words_to_process:
            if word in word_data:  # Only include words that were successfully processed
                word_list.append((word, word_data[word]))
        
        word_sections = []
        for i in range(0, len(word_list), self.words_per_section):
            word_sections.append(word_list[i:i + self.words_per_section])
        
        word_to_tests = {}
        for test_item in all_tests:
            word = test_item['answer'].lower()
            if word not in word_to_tests:
                word_to_tests[word] = []
            word_to_tests[word].append(test_item)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("VOCABULARY LEARNING MATERIALS\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Total Vocabulary Words: {total_words}\n")
            if self.track_new_words and len(duplicate_words) > 0:
                f.write(f"New Words: {len(new_words)}\n")
                f.write(f"Important/Duplicate Words: {len(duplicate_words)} (appeared in previous learning)\n")
            f.write(f"Examples per Word: {self.examples_per_word}\n")
            f.write(f"Total Test Questions: {len(all_tests)}\n")
            f.write(f"Words per Section: {self.words_per_section}\n")
            f.write(f"Test Batch Size: {self.test_batch_size}\n")
            f.write("=" * 80 + "\n\n\n")
            
            for section_num, word_section in enumerate(word_sections, 1):
                f.write(f"SECTION {section_num}: VOCABULARY WORDS {((section_num - 1) * self.words_per_section) + 1}-{min(section_num * self.words_per_section, total_words)}\n")
                f.write("=" * 80 + "\n\n")
                
                section_tests = []
                
                for idx, (word, data) in enumerate(word_section, 1):
                    global_idx = (section_num - 1) * self.words_per_section + idx
                    
                    # Get pronunciation and format it (strip existing slashes if present)
                    pronunciation = data.get('pronunciation', '')
                    if pronunciation:
                        # Remove leading/trailing slashes if present
                        pronunciation = pronunciation.strip('/')
                        pronunciation_text = f" /{pronunciation}/"
                    else:
                        pronunciation_text = ""
                    
                    if data.get('is_important', False):
                        occurrence_count = data.get('occurrence_count', 1)
                        f.write(f"{global_idx}. word: {word.lower()}{pronunciation_text} [IMPORTANT - appeared {occurrence_count} time(s) in previous learning]\n")
                        f.write("-" * 80 + "\n")
                        f.write("NOTE: This word appeared in previous learning sessions. Reviewing with different examples to reinforce understanding.\n\n")
                    else:
                        f.write(f"{global_idx}. word: {word.lower()}{pronunciation_text}\n")
                        f.write("-" * 80 + "\n")
                    
                    f.write(f"explanation:\n{data['explanation']}\n\n")
                    f.write("example:\n")
                    for example in data['examples']:
                        f.write(f"  • {example}\n")
                    f.write("\n")
                    
                    if word.lower() in word_to_tests:
                        section_tests.extend(word_to_tests[word.lower()])
                
                f.write("\n" + "=" * 80 + "\n\n")
                
                if section_tests:
                    # Shuffle tests to randomize order (makes it more challenging)
                    # Tests are shuffled so they don't appear in the same order as word explanations
                    shuffled_tests = section_tests.copy()
                    random.shuffle(shuffled_tests)
                    
                    section_test_batches = []
                    for i in range(0, len(shuffled_tests), self.test_batch_size):
                        section_test_batches.append(shuffled_tests[i:i + self.test_batch_size])
                    
                    for batch_num, batch in enumerate(section_test_batches, 1):
                        f.write(f"TEST - Section {section_num}, Batch {batch_num} ({len(batch)} questions)\n")
                        f.write("-" * 80 + "\n\n")
                        
                        for i, test_item in enumerate(batch, 1):
                            f.write(f"Question {i}:\n")
                            f.write(f"{test_item['sentence']}\n\n")
                        
                        f.write("\n" + "-" * 80 + "\n")
                        f.write("ANSWERS:\n")
                        f.write("-" * 80 + "\n\n")
                        
                        for i, test_item in enumerate(batch, 1):
                            f.write(f"Question {i}: {test_item['answer']}\n")
                            f.write(f"  Full sentence: {test_item['original']}\n\n")
                        
                        f.write("\n" + "=" * 80 + "\n\n\n")


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
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
        
        # Progress bar (hidden by default)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.status_bar.addPermanentWidget(self.progress_bar)
        
    def create_control_panel(self):
        """Create the control panel tab."""
        control_widget = QWidget()
        control_layout = QVBoxLayout(control_widget)
        control_layout.setSpacing(5)  # Reduced spacing
        control_layout.setContentsMargins(10, 10, 10, 10)
        
        # File selection group - renamed to Control Panel
        file_group = QGroupBox("Control Panel")
        file_layout = QVBoxLayout()
        file_layout.setSpacing(3)  # Tight spacing between rows
        file_layout.setContentsMargins(8, 8, 8, 8)
        
        # Input file
        input_layout = QHBoxLayout()
        input_layout.setSpacing(5)
        input_layout.addWidget(QLabel("Input File:"))
        self.input_file_edit = QLineEdit()
        self.input_file_edit.setPlaceholderText("Select vocabulary file(s)... (Hold Ctrl/Cmd to select multiple)")
        input_layout.addWidget(self.input_file_edit)
        self.browse_input_btn = QPushButton("Browse...")
        self.browse_input_btn.clicked.connect(self.browse_input_file)
        input_layout.addWidget(self.browse_input_btn)
        file_layout.addLayout(input_layout)
        
        # Output file
        output_layout = QHBoxLayout()
        output_layout.setSpacing(5)
        output_layout.addWidget(QLabel("Output File:"))
        self.output_file_edit = QLineEdit()
        self.output_file_edit.setPlaceholderText("Auto-generated if empty...")
        output_layout.addWidget(self.output_file_edit)
        self.browse_output_btn = QPushButton("Browse...")
        self.browse_output_btn.clicked.connect(self.browse_output_file)
        output_layout.addWidget(self.browse_output_btn)
        file_layout.addLayout(output_layout)
        
        # Load results file
        results_file_layout = QHBoxLayout()
        results_file_layout.setSpacing(5)
        results_file_layout.addWidget(QLabel("Results File:"))
        self.results_file_edit = QLineEdit()
        self.results_file_edit.setPlaceholderText("Select a results file to view...")
        results_file_layout.addWidget(self.results_file_edit)
        self.browse_results_btn = QPushButton("Load Results...")
        self.browse_results_btn.setStyleSheet("background-color: #9C27B0; color: white; padding: 6px;")
        self.browse_results_btn.clicked.connect(self.load_results_file)
        results_file_layout.addWidget(self.browse_results_btn)
        file_layout.addLayout(results_file_layout)
        
        file_group.setLayout(file_layout)
        control_layout.addWidget(file_group)
        
        # Options group
        options_group = QGroupBox("Options")
        options_layout = QGridLayout()
        options_layout.setSpacing(5)  # Reduced spacing
        options_layout.setContentsMargins(8, 8, 8, 8)
        
        # Examples per word
        options_layout.addWidget(QLabel("Examples per word:"), 0, 0)
        self.examples_spin = QSpinBox()
        self.examples_spin.setMinimum(1)
        self.examples_spin.setMaximum(10)
        self.examples_spin.setValue(1)
        options_layout.addWidget(self.examples_spin, 0, 1)
        
        # Words per section
        options_layout.addWidget(QLabel("Words per section:"), 0, 2)
        self.words_section_spin = QSpinBox()
        self.words_section_spin.setMinimum(1)
        self.words_section_spin.setMaximum(100)
        self.words_section_spin.setValue(20)
        options_layout.addWidget(self.words_section_spin, 0, 3)
        
        # Test batch size
        options_layout.addWidget(QLabel("Test batch size:"), 1, 0)
        self.test_batch_spin = QSpinBox()
        self.test_batch_spin.setMinimum(1)
        self.test_batch_spin.setMaximum(100)
        self.test_batch_spin.setValue(20)
        options_layout.addWidget(self.test_batch_spin, 1, 1)
        
        # Tracker file
        options_layout.addWidget(QLabel("Tracker file:"), 1, 2)
        self.tracker_edit = QLineEdit("processed_words.json")
        options_layout.addWidget(self.tracker_edit, 1, 3)
        
        # Track new words checkbox
        self.track_new_checkbox = QCheckBox("Track new words only")
        self.track_new_checkbox.setChecked(True)
        options_layout.addWidget(self.track_new_checkbox, 2, 0, 1, 2)
        
        options_group.setLayout(options_layout)
        control_layout.addWidget(options_group)
        
        # Buttons
        buttons_layout = QHBoxLayout()
        buttons_layout.setSpacing(5)
        
        self.process_btn = QPushButton("Process Vocabulary File")
        self.process_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 8px;")
        self.process_btn.clicked.connect(self.process_file)
        buttons_layout.addWidget(self.process_btn)
        
        self.check_new_btn = QPushButton("Check New Words")
        self.check_new_btn.setStyleSheet("background-color: #2196F3; color: white; padding: 8px;")
        self.check_new_btn.clicked.connect(self.check_new_words)
        buttons_layout.addWidget(self.check_new_btn)
        
        self.clear_tracker_btn = QPushButton("Clear Tracker")
        self.clear_tracker_btn.setStyleSheet("background-color: #f44336; color: white; padding: 8px;")
        self.clear_tracker_btn.clicked.connect(self.clear_tracker)
        buttons_layout.addWidget(self.clear_tracker_btn)
        
        buttons_layout.addStretch()
        control_layout.addLayout(buttons_layout)
        
        # Add stretch to push everything to top
        control_layout.addStretch()
        
        return control_widget
    
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
        filenames, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Vocabulary File(s) - Hold Ctrl/Cmd to select multiple",
            "",
            "All supported (*.html *.htm *.txt *.md);;HTML files (*.html *.htm);;Text files (*.txt);;Markdown files (*.md);;All files (*.*)"
        )
        if filenames:
            # Display multiple files separated by semicolon
            file_list = "; ".join(filenames)
            self.input_file_edit.setText(file_list)
            if len(filenames) > 1:
                self.statusBar().showMessage(f"Selected {len(filenames)} files", 3000)
            
    def browse_output_file(self):
        """Browse for output file."""
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save Output File",
            "",
            "Text files (*.txt);;All files (*.*)"
        )
        if filename:
            self.output_file_edit.setText(filename)
    
    def load_results_file(self):
        """Load and display a results file."""
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Load Results File",
            "",
            "Text files (*.txt);;All files (*.*)"
        )
        
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
            
            self.status_bar.showMessage(f"Loaded: {os.path.basename(filename)}")
            QMessageBox.information(self, "Success", f"Results file loaded successfully!\n\n{os.path.basename(filename)}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load file:\n{str(e)}")
            self.status_bar.showMessage("Error loading file")
    
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
                item = QTreeWidgetItem(root_item, ["📚 Main Header"])
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
        
        # Determine output file
        output_path = self.output_file_edit.text().strip()
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"vocabulary_learning_materials_{timestamp}.txt"
            self.output_file_edit.setText(output_path)
        
        # Clear results
        self.results_text.clear()
        self.nav_tree.clear()
        self.section_positions = {}
        self.current_file_path = None
        
        self.append_text("=" * 80 + "\n", 'header')
        self.append_text("VOCABULARY LEARNING MATERIALS\n", 'header')
        self.append_text("=" * 80 + "\n\n", 'header')
        
        # Disable buttons
        self.process_btn.setEnabled(False)
        self.check_new_btn.setEnabled(False)
        self.clear_tracker_btn.setEnabled(False)
        
        # Show progress bar
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate
        
        # Start processing thread
        self.processing_thread = ProcessingThread(
            valid_paths,  # Pass list of files
            output_path,
            self.examples_spin.value(),
            self.test_batch_spin.value(),
            self.words_section_spin.value(),
            self.track_new_checkbox.isChecked(),
            self.tracker_edit.text()
        )
        self.processing_thread.progress.connect(self.on_progress)
        self.processing_thread.finished.connect(self.on_finished)
        self.processing_thread.error.connect(self.on_error)
        self.processing_thread.start()
        
    def on_progress(self, message):
        """Handle progress updates."""
        self.status_bar.showMessage(message)
        self.append_text(f"{message}\n")
        
    def on_finished(self, result):
        """Handle processing completion."""
        self.progress_bar.setVisible(False)
        self.status_bar.showMessage(f"Complete! Processed {result['total_words']} words")
        
        # Load the output file into results viewer
        output_file = result['output_file']
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
        
        # Display results summary
        self._display_results_summary(result)
        
        # Re-enable buttons
        self.process_btn.setEnabled(True)
        self.check_new_btn.setEnabled(True)
        self.clear_tracker_btn.setEnabled(True)
        
        QMessageBox.information(
            self,
            "Success",
            f"Processing complete!\n\nProcessed: {result['total_words']} words\nOutput: {result['output_file']}"
        )
        
    def on_error(self, error_msg):
        """Handle processing errors."""
        self.progress_bar.setVisible(False)
        self.status_bar.showMessage("Error occurred")
        self.append_text(f"\nERROR:\n{error_msg}\n", 'important')
        
        # Re-enable buttons
        self.process_btn.setEnabled(True)
        self.check_new_btn.setEnabled(True)
        self.clear_tracker_btn.setEnabled(True)
        
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
        """Check for new words without processing (supports multiple files)."""
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
            
        try:
            self.results_text.clear()
            self.nav_tree.clear()
            if len(valid_paths) > 1:
                self.append_text(f"Checking for new words in {len(valid_paths)} files...\n\n", 'header')
            else:
                self.append_text("Checking for new words...\n\n", 'header')
            
            loader = VocabularyLoader()
            all_words = []
            
            # Load words from all files
            for input_path in valid_paths:
                words_from_file = loader.load_file(input_path)
                all_words.extend(words_from_file)
                if len(valid_paths) > 1:
                    self.append_text(f"Loaded {len(words_from_file)} words from {os.path.basename(input_path)}\n", 'header')
            
            # Remove duplicates while preserving order
            seen = set()
            unique_words = []
            for word in all_words:
                if word.lower() not in seen:
                    seen.add(word.lower())
                    unique_words.append(word)
            
            all_words = unique_words
            
            if len(valid_paths) > 1:
                self.append_text(f"\nTotal unique words from {len(valid_paths)} files: {len(all_words)}\n\n", 'header')
            
            if not all_words:
                self.append_text("ERROR: No vocabulary words found in file.\n", 'important')
                return
            
            tracker = WordTracker(self.tracker_edit.text())
            stats = tracker.get_stats()
            
            new_words, duplicate_words = tracker.get_new_words(all_words)
            
            self.append_text("Word Analysis:\n", 'header')
            self.append_text(f"  • Total words in file: {len(all_words)}\n")
            self.append_text(f"  • Already processed: {len(duplicate_words)}\n")
            self.append_text(f"  • New words: {len(new_words)}\n")
            self.append_text(f"  • Total tracked: {stats['total_processed']}\n\n")
            
            if duplicate_words:
                self.append_text("Important words (appeared before):\n", 'important')
                for i, word in enumerate(duplicate_words[:20], 1):
                    count = tracker.get_occurrence_count(word)
                    word_display = loader.format_word_with_pronunciation(word)
                    self.append_text(f"  {i}. {word_display} (appeared {count} time(s))\n", 'important')
                if len(duplicate_words) > 20:
                    self.append_text(f"  ... and {len(duplicate_words) - 20} more\n", 'important')
                self.append_text("\n")
            
            if new_words:
                self.append_text("New words found:\n", 'word')
                for i, word in enumerate(new_words[:50], 1):
                    word_display = loader.format_word_with_pronunciation(word)
                    self.append_text(f"  {i}. {word_display}\n", 'word')
                if len(new_words) > 50:
                    self.append_text(f"  ... and {len(new_words) - 50} more\n", 'word')
            elif not duplicate_words:
                self.append_text("No new words found! All words have already been processed.\n", 'important')
            
            # Switch to Results Panel tab
            self.tab_widget.setCurrentIndex(1)
                
            self.status_bar.showMessage("Check complete")
            
        except Exception as e:
            self.append_text(f"Error: {str(e)}\n", 'important')
            self.status_bar.showMessage("Error occurred")
            QMessageBox.critical(self, "Error", f"An error occurred:\n{str(e)}")
            
    def clear_tracker(self):
        """Clear the tracker file."""
        reply = QMessageBox.question(
            self,
            "Confirm Clear",
            "Are you sure you want to clear the tracker file?\n\nThis will delete all tracked words.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply != QMessageBox.Yes:
            return
            
        try:
            tracker_file = self.tracker_edit.text()
            tracker = WordTracker(tracker_file)
            stats_before = tracker.get_stats()
            tracker.clear()
            
            self.results_text.clear()
            self.append_text(f"Tracker cleared successfully!\n", 'header')
            self.append_text(f"Removed {stats_before['total_processed']} tracked words\n", 'important')
            self.append_text(f"File deleted: {tracker_file}\n", 'important')
            
            self.status_bar.showMessage("Tracker cleared")
            QMessageBox.information(self, "Success", "Tracker cleared successfully!")
            
        except Exception as e:
            self.append_text(f"Error: {str(e)}\n", 'important')
            self.status_bar.showMessage("Error occurred")
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
