import sys
import os

# CRITICAL: Fix Qt plugin conflict between PyQt5 and OpenCV
os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QScrollArea, 
                             QGridLayout, QDialog, QMessageBox, QProgressBar, 
                             QFrame, QInputDialog, QSizePolicy, QSpacerItem)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QPropertyAnimation, QEasingCurve, QSize
from PyQt5.QtGui import QPixmap, QFont, QCursor

# Import backend functions
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from imagemodel import (process_folder, get_all_persons, get_person_images, 
                           rename_person, delete_person, delete_image, export_to_csv)
except ModuleNotFoundError:
    print("Error: imagemodel.py not found!")
    print("Please ensure gui.py is in the same directory as imagemodel.py")
    sys.exit(1)


# Global Stylesheet
GLOBAL_STYLE = """
QMainWindow {
    background-color: #121212;
}

QWidget {
    font-family: 'Segoe UI', 'Inter', 'Arial', sans-serif;
    color: #E0E0E0;
}

QLabel {
    color: #E0E0E0;
}

QScrollBar:vertical {
    background-color: #1e1e1e;
    width: 10px;
    border-radius: 5px;
    margin: 0px;
}

QScrollBar::handle:vertical {
    background-color: #1E88E5;
    border-radius: 5px;
    min-height: 30px;
}

QScrollBar::handle:vertical:hover {
    background-color: #42A5F5;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}

QScrollBar:horizontal {
    background-color: #1e1e1e;
    height: 10px;
    border-radius: 5px;
}

QScrollBar::handle:horizontal {
    background-color: #1E88E5;
    border-radius: 5px;
}
"""


class ProcessingThread(QThread):
    """Thread for running face recognition processing"""
    progress = pyqtSignal(str)
    finished = pyqtSignal()
    
    def run(self):
        try:
            self.progress.emit("Starting face recognition...")
            process_folder()
            export_to_csv()
            self.progress.emit("Processing complete!")
            self.finished.emit()
        except Exception as e:
            self.progress.emit(f"Error: {str(e)}")
            self.finished.emit()


class ImageViewDialog(QDialog):
    """Dialog to view all images of a person"""
    def __init__(self, person_name, parent=None):
        super().__init__(parent)
        self.person_name = person_name
        self.images = get_person_images(person_name)
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle(f"Images - {self.person_name}")
        self.setGeometry(200, 200, 950, 650)
        self.setStyleSheet("""
            QDialog {
                background-color: #1e1e1e;
            }
        """)
        
        layout = QVBoxLayout()
        layout.setSpacing(20)
        layout.setContentsMargins(25, 25, 25, 25)
        
        # Header
        header = QLabel(f"📸 {self.person_name}")
        header.setFont(QFont("Segoe UI", 20, QFont.Bold))
        header.setStyleSheet("""
            color: #1E88E5;
            padding: 15px;
            background-color: #2a2a2a;
            border-radius: 8px;
        """)
        layout.addWidget(header)
        
        # Count label
        count_label = QLabel(f"{len(self.images)} photos in collection")
        count_label.setFont(QFont("Segoe UI", 11))
        count_label.setStyleSheet("color: #999999; padding: 5px 15px;")
        layout.addWidget(count_label)
        
        # Scroll area for images
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background-color: #1e1e1e; }")
        
        container = QWidget()
        grid = QGridLayout(container)
        grid.setSpacing(20)
        
        # Display images in grid
        row, col = 0, 0
        for img_path in self.images:
            img_widget = self.create_image_widget(img_path)
            grid.addWidget(img_widget, row, col)
            
            col += 1
            if col >= 3:
                col = 0
                row += 1
        
        scroll.setWidget(container)
        layout.addWidget(scroll)
        
        # Close button
        close_btn = QPushButton("Close")
        close_btn.setCursor(QCursor(Qt.PointingHandCursor))
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: #333333;
                color: white;
                border: 1px solid #444444;
                padding: 12px 30px;
                border-radius: 8px;
                font-size: 13px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #444444;
                border: 1px solid #555555;
            }
        """)
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn, alignment=Qt.AlignCenter)
        
        self.setLayout(layout)
    
    def create_image_widget(self, img_path):
        """Create widget for single image with delete option"""
        frame = QFrame()
        frame.setStyleSheet("""
            QFrame {
                background-color: #2a2a2a;
                border: 1px solid #333333;
                border-radius: 8px;
                padding: 12px;
            }
            QFrame:hover {
                border: 1px solid #1E88E5;
            }
        """)
        
        layout = QVBoxLayout(frame)
        layout.setSpacing(10)
        
        # Image
        img_label = QLabel()
        pixmap = QPixmap(img_path)
        if not pixmap.isNull():
            scaled = pixmap.scaled(260, 260, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            img_label.setPixmap(scaled)
        img_label.setAlignment(Qt.AlignCenter)
        img_label.setStyleSheet("border: none;")
        layout.addWidget(img_label)
        
        # Image name
        name_label = QLabel(os.path.basename(img_path))
        name_label.setAlignment(Qt.AlignCenter)
        name_label.setFont(QFont("Segoe UI", 9))
        name_label.setStyleSheet("color: #999999; padding: 5px; border: none;")
        name_label.setWordWrap(True)
        layout.addWidget(name_label)
        
        # Delete button
        del_btn = QPushButton("🗑️ Delete")
        del_btn.setCursor(QCursor(Qt.PointingHandCursor))
        del_btn.setStyleSheet("""
            QPushButton {
                background-color: #cf6679;
                color: white;
                border: none;
                padding: 8px;
                border-radius: 6px;
                font-size: 11px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #e07889;
            }
        """)
        del_btn.clicked.connect(lambda: self.delete_image(img_path))
        layout.addWidget(del_btn)
        
        return frame
    
    def delete_image(self, img_path):
        """Delete an image"""
        reply = QMessageBox.question(self, 'Delete Image', 
                                     f"Delete {os.path.basename(img_path)}?",
                                     QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            if delete_image(self.person_name, os.path.basename(img_path)):
                QMessageBox.information(self, "Success", "Image deleted!")
                self.close()
                self.parent().load_persons()
            else:
                QMessageBox.warning(self, "Error", "Could not delete image")


class PersonCard(QFrame):
    """Material Design Person Card"""
    def __init__(self, person_data, parent=None):
        super().__init__(parent)
        self.person_data = person_data
        self.parent_window = parent
        self.init_ui()
        self.setCursor(QCursor(Qt.PointingHandCursor))
        
    def init_ui(self):
        self.setStyleSheet("""
            QFrame {
                background-color: #2a2a2a;
                border: 1px solid #333333;
                border-radius: 12px;
                padding: 15px;
            }
            QFrame:hover {
                background-color: #323232;
                border: 1px solid #1E88E5;
            }
        """)
        
        self.setFixedWidth(220)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        
        # Thumbnail
        thumb_label = QLabel()
        thumb_label.setFixedSize(180, 180)
        if self.person_data['thumbnail'] and os.path.exists(self.person_data['thumbnail']):
            pixmap = QPixmap(self.person_data['thumbnail'])
            scaled = pixmap.scaled(180, 180, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            thumb_label.setPixmap(scaled)
        else:
            thumb_label.setText("👤")
            thumb_label.setStyleSheet("font-size: 70px; background-color: #1e1e1e; border-radius: 8px;")
        thumb_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(thumb_label)
        
        # Name
        name_label = QLabel(self.person_data['name'])
        name_label.setFont(QFont("Segoe UI", 13, QFont.Bold))
        name_label.setAlignment(Qt.AlignCenter)
        name_label.setStyleSheet("color: #E0E0E0; padding: 5px;")
        name_label.setWordWrap(True)
        layout.addWidget(name_label)
        
        # Photo count badge
        count_label = QLabel(f"📷 {self.person_data['photo_count']}")
        count_label.setAlignment(Qt.AlignCenter)
        count_label.setFont(QFont("Segoe UI", 11))
        count_label.setStyleSheet("""
            color: #1E88E5;
            background-color: #1e1e1e;
            padding: 6px 12px;
            border-radius: 12px;
        """)
        layout.addWidget(count_label)
        
        # Buttons row
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(8)
        
        view_btn = QPushButton("View")
        view_btn.setCursor(QCursor(Qt.PointingHandCursor))
        view_btn.setStyleSheet(self.get_button_style("#1E88E5", "#42A5F5"))
        view_btn.clicked.connect(self.view_images)
        btn_layout.addWidget(view_btn)
        
        rename_btn = QPushButton("✏️")
        rename_btn.setCursor(QCursor(Qt.PointingHandCursor))
        rename_btn.setFixedWidth(40)
        rename_btn.setStyleSheet(self.get_button_style("#4CAF50", "#66BB6A"))
        rename_btn.clicked.connect(self.rename)
        btn_layout.addWidget(rename_btn)
        
        delete_btn = QPushButton("🗑️")
        delete_btn.setCursor(QCursor(Qt.PointingHandCursor))
        delete_btn.setFixedWidth(40)
        delete_btn.setStyleSheet(self.get_button_style("#cf6679", "#e07889"))
        delete_btn.clicked.connect(self.delete)
        btn_layout.addWidget(delete_btn)
        
        layout.addLayout(btn_layout)
    
    def get_button_style(self, bg_color, hover_color):
        return f"""
            QPushButton {{
                background-color: {bg_color};
                color: white;
                border: none;
                padding: 8px;
                border-radius: 6px;
                font-weight: 600;
                font-size: 12px;
            }}
            QPushButton:hover {{
                background-color: {hover_color};
            }}
            QPushButton:pressed {{
                padding: 9px 7px 7px 9px;
            }}
        """
    
    def view_images(self):
        dialog = ImageViewDialog(self.person_data['name'], self.parent_window)
        dialog.exec_()
    
    def rename(self):
        new_name, ok = QInputDialog.getText(self, 'Rename Person', 
                                           'Enter new name:',
                                           text=self.person_data['name'])
        if ok and new_name and new_name != self.person_data['name']:
            rename_person(self.person_data['name'], new_name)
            QMessageBox.information(self, "Success", f"Renamed to {new_name}")
            self.parent_window.load_persons()
    
    def delete(self):
        reply = QMessageBox.question(self, 'Delete Person', 
                                    f"Delete {self.person_data['name']} and all photos?",
                                    QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            delete_person(self.person_data['name'])
            QMessageBox.information(self, "Deleted", f"{self.person_data['name']} deleted")
            self.parent_window.load_persons()


class Sidebar(QWidget):
    """Modern Sidebar with vertical navigation"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.init_ui()
        
    def init_ui(self):
        self.setFixedWidth(250)
        self.setStyleSheet("""
            QWidget {
                background-color: #121212;
            }
        """)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 30, 20, 20)
        
        # Logo/Title
        title = QLabel("🎭 Face\nRecognition")
        title.setFont(QFont("Segoe UI", 18, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("""
            color: #1E88E5;
            padding: 20px;
            background-color: #1e1e1e;
            border-radius: 12px;
        """)
        layout.addWidget(title)
        
        layout.addSpacing(20)
        
        # Navigation buttons
        self.run_btn = self.create_nav_button("▶️  Run Recognizer", "#1E88E5", "#42A5F5")
        self.run_btn.clicked.connect(self.parent_window.run_recognition)
        layout.addWidget(self.run_btn)
        
        self.refresh_btn = self.create_nav_button("🔄  Refresh Gallery", "#4CAF50", "#66BB6A")
        self.refresh_btn.clicked.connect(self.parent_window.load_persons)
        layout.addWidget(self.refresh_btn)
        
        self.export_btn = self.create_nav_button("📊  Export Report", "#03A9F4", "#29B6F6")
        self.export_btn.clicked.connect(self.parent_window.export_report)
        layout.addWidget(self.export_btn)
        
        self.no_face_btn = self.create_nav_button("🚫  No Face Photos", "#FF9800", "#FFB74D")
        self.no_face_btn.clicked.connect(self.parent_window.show_no_face_photos)
        layout.addWidget(self.no_face_btn)
        
        layout.addSpacing(20)
        
        # Stats section
        stats_frame = QFrame()
        stats_frame.setStyleSheet("""
            QFrame {
                background-color: #1e1e1e;
                border-radius: 12px;
                padding: 15px;
            }
        """)
        stats_layout = QVBoxLayout(stats_frame)
        
        self.stats_label = QLabel("📈 Statistics")
        self.stats_label.setFont(QFont("Segoe UI", 12, QFont.Bold))
        self.stats_label.setStyleSheet("color: #1E88E5;")
        stats_layout.addWidget(self.stats_label)
        
        self.person_count_label = QLabel("Total Persons: 0")
        self.person_count_label.setFont(QFont("Segoe UI", 10))
        self.person_count_label.setStyleSheet("color: #999999; padding-top: 8px;")
        stats_layout.addWidget(self.person_count_label)
        
        layout.addWidget(stats_frame)
        
        # Spacer
        layout.addItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))
        
        # Status indicator at bottom
        self.status_label = QLabel("● Ready")
        self.status_label.setFont(QFont("Segoe UI", 10))
        self.status_label.setStyleSheet("color: #4CAF50; padding: 10px;")
        layout.addWidget(self.status_label)
        
    def create_nav_button(self, text, bg_color, hover_color):
        btn = QPushButton(text)
        btn.setCursor(QCursor(Qt.PointingHandCursor))
        btn.setFont(QFont("Segoe UI", 11, QFont.Bold))
        btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {bg_color};
                color: white;
                border: none;
                padding: 15px;
                border-radius: 8px;
                text-align: left;
            }}
            QPushButton:hover {{
                background-color: {hover_color};
            }}
            QPushButton:pressed {{
                padding: 16px 14px 14px 16px;
            }}
            QPushButton:disabled {{
                background-color: #2a2a2a;
                color: #666666;
            }}
        """)
        return btn
    
    def update_stats(self, count):
        self.person_count_label.setText(f"Total Persons: {count}")
    
    def update_status(self, message, color="#4CAF50"):
        self.status_label.setText(f"● {message}")
        self.status_label.setStyleSheet(f"color: {color}; padding: 10px;")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("🎭 Face Recognition Dashboard")
        self.setGeometry(100, 50, 1400, 850)
        self.setStyleSheet(GLOBAL_STYLE)
        
        # Central widget with horizontal layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Sidebar
        self.sidebar = Sidebar(self)
        main_layout.addWidget(self.sidebar)
        
        # Main content area
        content_area = QWidget()
        content_area.setStyleSheet("background-color: #1e1e1e;")
        content_layout = QVBoxLayout(content_area)
        content_layout.setSpacing(20)
        content_layout.setContentsMargins(30, 30, 30, 30)
        
        # Header
        header_layout = QHBoxLayout()
        
        page_title = QLabel("Gallery")
        page_title.setFont(QFont("Segoe UI", 24, QFont.Bold))
        page_title.setStyleSheet("color: #E0E0E0;")
        header_layout.addWidget(page_title)
        
        header_layout.addStretch()
        
        # Search box placeholder
        search_label = QLabel("🔍 View all recognized faces")
        search_label.setFont(QFont("Segoe UI", 11))
        search_label.setStyleSheet("color: #999999;")
        header_layout.addWidget(search_label)
        
        content_layout.addLayout(header_layout)
        
        # Progress bar
        self.progress_container = QWidget()
        progress_layout = QVBoxLayout(self.progress_container)
        progress_layout.setContentsMargins(0, 0, 0, 0)
        progress_layout.setSpacing(8)
        
        self.progress_label = QLabel("Ready to process")
        self.progress_label.setFont(QFont("Segoe UI", 11))
        self.progress_label.setStyleSheet("color: #999999;")
        self.progress_label.setAlignment(Qt.AlignCenter)
        progress_layout.addWidget(self.progress_label)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #2a2a2a;
                border-radius: 8px;
                background-color: #2a2a2a;
                height: 8px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #1E88E5;
                border-radius: 6px;
            }
        """)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setRange(0, 0)
        self.progress_bar.hide()
        progress_layout.addWidget(self.progress_bar)
        
        content_layout.addWidget(self.progress_container)
        
        # Scroll area for person cards
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background-color: #1e1e1e; }")
        
        self.persons_container = QWidget()
        self.persons_layout = QGridLayout(self.persons_container)
        self.persons_layout.setSpacing(25)
        self.persons_layout.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        
        scroll.setWidget(self.persons_container)
        content_layout.addWidget(scroll)
        
        main_layout.addWidget(content_area, stretch=1)
        
        # Load persons on startup
        self.load_persons()
    
    def load_persons(self):
        """Load and display all persons"""
        # Clear existing
        for i in reversed(range(self.persons_layout.count())): 
            widget = self.persons_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)
        
        # Get persons
        persons = get_all_persons()
        
        # Update stats
        self.sidebar.update_stats(len(persons))
        
        if not persons:
            no_data = QLabel("No persons found yet\n\nAdd photos to 'new_photos' folder\nand click 'Run Recognizer'")
            no_data.setAlignment(Qt.AlignCenter)
            no_data.setFont(QFont("Segoe UI", 14))
            no_data.setStyleSheet("color: #666666; padding: 100px;")
            self.persons_layout.addWidget(no_data, 0, 0)
            return
        
        # Display in grid (4 per row)
        row, col = 0, 0
        for person_data in persons:
            card = PersonCard(person_data, self)
            self.persons_layout.addWidget(card, row, col)
            
            col += 1
            if col >= 4:
                col = 0
                row += 1
        
        self.sidebar.update_status("Gallery loaded", "#4CAF50")
    
    def run_recognition(self):
        """Start recognition process"""
        self.sidebar.run_btn.setEnabled(False)
        self.sidebar.update_status("Processing...", "#FF9800")
        self.progress_bar.show()
        self.progress_label.setText("Processing images...")
        
        # Start processing thread
        self.thread = ProcessingThread()
        self.thread.progress.connect(self.update_progress)
        self.thread.finished.connect(self.processing_finished)
        self.thread.start()
    
    def update_progress(self, message):
        self.progress_label.setText(message)
    
    def processing_finished(self):
        self.progress_bar.hide()
        self.sidebar.run_btn.setEnabled(True)
        self.sidebar.update_status("Complete!", "#4CAF50")
        self.progress_label.setText("✅ Processing complete!")
        QTimer.singleShot(2000, lambda: self.progress_label.setText("Ready to process"))
        self.load_persons()
    
    def export_report(self):
        """Export CSV report"""
        try:
            export_to_csv()
            self.sidebar.update_status("Report exported", "#4CAF50")
            QMessageBox.information(self, "Success", "CSV report exported successfully!")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to export: {str(e)}")
    
    def show_no_face_photos(self):
        """Show dialog with photos that had no face detected"""
        from imagemodel import NO_FACE_DIR
        
        if not os.path.exists(NO_FACE_DIR):
            QMessageBox.information(self, "No Face Photos", "No photos found without faces.")
            return
        
        image_files = [f for f in os.listdir(NO_FACE_DIR) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))]
        
        if not image_files:
            QMessageBox.information(self, "No Face Photos", "No photos found without faces.")
            return
        
        # Create dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("Photos Without Faces")
        dialog.setGeometry(200, 200, 950, 650)
        dialog.setStyleSheet("QDialog { background-color: #1e1e1e; }")
        
        layout = QVBoxLayout()
        layout.setSpacing(20)
        layout.setContentsMargins(25, 25, 25, 25)
        
        # Header
        header = QLabel(f"🚫 No Face Detected")
        header.setFont(QFont("Segoe UI", 20, QFont.Bold))
        header.setStyleSheet("""
            color: #FF9800;
            padding: 15px;
            background-color: #2a2a2a;
            border-radius: 8px;
        """)
        layout.addWidget(header)
        
        # Count label
        count_label = QLabel(f"{len(image_files)} photos with no face detected")
        count_label.setFont(QFont("Segoe UI", 11))
        count_label.setStyleSheet("color: #999999; padding: 5px 15px;")
        layout.addWidget(count_label)
        
        # Scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background-color: #1e1e1e; }")
        
        container = QWidget()
        grid = QGridLayout(container)
        grid.setSpacing(20)
        
        # Display images
        row, col = 0, 0
        for img_name in image_files:
            img_path = os.path.join(NO_FACE_DIR, img_name)
            img_widget = self.create_no_face_widget(img_path)
            grid.addWidget(img_widget, row, col)
            
            col += 1
            if col >= 3:
                col = 0
                row += 1
        
        scroll.setWidget(container)
        layout.addWidget(scroll)
        
        # Close button
        close_btn = QPushButton("Close")
        close_btn.setCursor(QCursor(Qt.PointingHandCursor))
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: #333333;
                color: white;
                border: 1px solid #444444;
                padding: 12px 30px;
                border-radius: 8px;
                font-size: 13px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #444444;
                border: 1px solid #555555;
            }
        """)
        close_btn.clicked.connect(dialog.close)
        layout.addWidget(close_btn, alignment=Qt.AlignCenter)
        
        dialog.setLayout(layout)
        dialog.exec_()
    
    def create_no_face_widget(self, img_path):
        """Create widget for no-face photo"""
        frame = QFrame()
        frame.setStyleSheet("""
            QFrame {
                background-color: #2a2a2a;
                border: 1px solid #333333;
                border-radius: 8px;
                padding: 12px;
            }
            QFrame:hover {
                border: 1px solid #FF9800;
            }
        """)
        
        layout = QVBoxLayout(frame)
        layout.setSpacing(10)
        
        # Image
        img_label = QLabel()
        pixmap = QPixmap(img_path)
        if not pixmap.isNull():
            scaled = pixmap.scaled(260, 260, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            img_label.setPixmap(scaled)
        img_label.setAlignment(Qt.AlignCenter)
        img_label.setStyleSheet("border: none;")
        layout.addWidget(img_label)
        
        # Image name
        name_label = QLabel(os.path.basename(img_path))
        name_label.setAlignment(Qt.AlignCenter)
        name_label.setFont(QFont("Segoe UI", 9))
        name_label.setStyleSheet("color: #999999; padding: 5px; border: none;")
        name_label.setWordWrap(True)
        layout.addWidget(name_label)
        
        # Delete button
        del_btn = QPushButton("🗑️ Delete")
        del_btn.setCursor(QCursor(Qt.PointingHandCursor))
        del_btn.setStyleSheet("""
            QPushButton {
                background-color: #cf6679;
                color: white;
                border: none;
                padding: 8px;
                border-radius: 6px;
                font-size: 11px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #e07889;
            }
        """)
        del_btn.clicked.connect(lambda: self.delete_no_face_image(img_path))
        layout.addWidget(del_btn)
        
        return frame
    
    def delete_no_face_image(self, img_path):
        """Delete a no-face image"""
        reply = QMessageBox.question(self, 'Delete Image', 
                                     f"Delete {os.path.basename(img_path)}?",
                                     QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            try:
                os.remove(img_path)
                QMessageBox.information(self, "Success", "Image deleted!")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Could not delete image: {str(e)}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())