import sys
import json
import os
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QPushButton, QMenu, QDialog, QLineEdit,
    QLabel, QFormLayout, QColorDialog, QSlider, QDialogButtonBox,
    QSpinBox, QDoubleSpinBox
)
from PyQt6.QtCore import (
    Qt, QTimer, QPoint, QSize, QRect, QThread, pyqtSignal
)
from PyQt6.QtGui import QPainter, QColor, QTextCursor
from openai import OpenAI
import pyperclip

# Define a Worker Thread for handling OpenAI API calls
class OpenAIWorker(QThread):
    token_received = pyqtSignal(str)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, endpoint, api_key, system_prompt, messages, model_name, generation_params, custom_params, parent=None):
        super().__init__(parent)
        self.endpoint = endpoint
        self.api_key = api_key
        self.system_prompt = system_prompt
        self.messages = messages
        self.model_name = model_name
        self.generation_params = generation_params
        self.custom_params = custom_params

    def run(self):
        try:
            client = OpenAI(
                base_url=self.endpoint,
                api_key=self.api_key  # Use the provided API key
            )
            # Prepare messages with system prompt
            all_messages = [{"role": "system", "content": self.system_prompt}] + self.messages

            response_stream = client.chat.completions.create(
                model=self.model_name,
                messages=all_messages,
                stream=True,
                **self.generation_params,
                extra_body=self.custom_params
            )

            accumulated_response = ""
            for chunk in response_stream:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    accumulated_response += content
                    self.token_received.emit(content)
                #if chunk.choices[0].finish_reason != 'null':
                #    break
            self.finished.emit(accumulated_response)
        except Exception as e:
            self.error.emit(str(e))


class ModelDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent, Qt.WindowType.Window | Qt.WindowType.FramelessWindowHint)
        
            # Fetch the current text color and background color from the parent
        text_color = parent.text_color if parent else "#FFFFFF"
        bg_color = parent.bg_color if parent else "#1E1E1E"

        self.setStyleSheet(f"""
            QDialog {{
                background-color: {bg_color};
                border-radius: 10px;
                color: {text_color};
            }}
            QLineEdit, QTextEdit {{
                background-color: rgba(50, 50, 50, 0.78); /* Adjusted alpha */
                border: 1px solid rgba(100, 100, 100, 0.59); /* Adjusted alpha */
                color: {text_color};
                border-radius: 5px;
            }}
            QPushButton {{
                background-color: rgba(70, 70, 70, 0.78); /* Adjusted alpha */
                color: {text_color};
                border-radius: 5px;
                padding: 5px;
            }}
            QPushButton:hover {{
                background-color: rgba(90, 90, 90, 0.90); /* Adjusted alpha */
            }}
            QSpinBox, QDoubleSpinBox {{
                background-color: rgba(50, 50, 50, 0.78); /* Adjusted alpha */
                border: 1px solid rgba(100, 100, 100, 0.59); /* Adjusted alpha */
                color: {text_color};
                border-radius: 5px;
            }}
        """)
        self.setModal(True)

        layout = QFormLayout()

        # API Settings
        self.api_address_input = QLineEdit()
        self.api_address_input.setText(parent.endpoint)

        self.api_key_input = QLineEdit()
        self.api_key_input.setText(parent.api_key)
        self.api_key_input.setPlaceholderText("http://127.0.0.1:5001/v1")
        layout.addRow(QLabel("API Address:"), self.api_address_input)
        layout.addRow(QLabel("API Key:"), self.api_key_input)

        # Model Settings
        self.model_name_input = QLineEdit()
        self.model_name_input.setText(parent.model_name)

        self.system_prompt_input = QTextEdit()
        self.system_prompt_input.setText(parent.system_prompt)

        layout.addRow(QLabel("Model Name:"), self.model_name_input)
        layout.addRow(QLabel("System Prompt:"), self.system_prompt_input)

        # Generation Parameters
        layout.addRow(QLabel("<b>Generation Parameters:</b>"))

        self.temperature_input = QDoubleSpinBox()
        self.temperature_input.setRange(0.0, 10.0)
        self.temperature_input.setSingleStep(0.1)
        self.temperature_input.setValue(parent.generation_params.get("temperature", 1.0))

        self.max_tokens_input = QSpinBox()
        self.max_tokens_input.setRange(1, 4096)
        self.max_tokens_input.setValue(parent.generation_params.get("max_tokens", 300))

        self.presence_penalty_input = QDoubleSpinBox()
        self.presence_penalty_input.setRange(0.0, 2.0)
        self.presence_penalty_input.setSingleStep(0.1)
        self.presence_penalty_input.setValue(parent.generation_params.get("presence_penalty", 0.0))

        self.frequency_penalty_input = QDoubleSpinBox()
        self.frequency_penalty_input.setRange(0.0, 2.0)
        self.frequency_penalty_input.setSingleStep(0.1)
        self.frequency_penalty_input.setValue(parent.generation_params.get("frequency_penalty", 0.0))

        self.top_p_input = QDoubleSpinBox()
        self.top_p_input.setRange(0.0, 1.0)
        self.top_p_input.setSingleStep(0.01)
        self.top_p_input.setValue(parent.generation_params.get("top_p", 1.0))

        layout.addRow("Temperature:", self.temperature_input)
        layout.addRow("Max Tokens:", self.max_tokens_input)
        layout.addRow("Presence Penalty:", self.presence_penalty_input)
        layout.addRow("Frequency Penalty:", self.frequency_penalty_input)
        layout.addRow("Top P:", self.top_p_input)

        # Additional Custom Parameters
        self.custom_params_input = QTextEdit()
        self.custom_params_input.setPlaceholderText("Enter custom parameters, one per line.\nExample:\n- top_k: 20\n- min_p: 0.1")
        custom_params = parent.custom_params if parent else {}
        custom_params_text = "\n".join([f"- {k}: {v}" for k, v in custom_params.items()])
        self.custom_params_input.setText(custom_params_text)

        layout.addRow(QLabel("<b>Additional Custom Parameters:</b>"))
        layout.addRow(self.custom_params_input)

        # Save Button
        save_button = QPushButton("Save Model Settings")
        save_button.clicked.connect(self.save_settings)
        layout.addRow(save_button)

        self.setLayout(layout)

    def save_settings(self):
        parent = self.parent()
        parent.endpoint = self.api_address_input.text()
        parent.api_key = self.api_key_input.text()
        parent.model_name = self.model_name_input.text()
        parent.system_prompt = self.system_prompt_input.toPlainText()

        # Update generation parameters
        parent.generation_params = {
            "temperature": self.temperature_input.value(),
            "max_tokens": self.max_tokens_input.value(),
            "presence_penalty": self.presence_penalty_input.value(),
            "frequency_penalty": self.frequency_penalty_input.value(),
            "top_p": self.top_p_input.value()
        }

        # Parse custom parameters
        custom_params_text = self.custom_params_input.toPlainText()
        custom_params = {}
        for line in custom_params_text.splitlines():
            line = line.strip()
            if line.startswith("-"):
                try:
                    key, value = line[1:].split(":", 1)
                    key = key.strip()
                    value = json.loads(value.strip())  # Support for different data types
                    custom_params[key] = value
                except Exception:
                    continue  # Ignore malformed lines
        parent.custom_params = custom_params

        parent.init_openai()
        parent.save_settings()  # Save settings to shinku-config.json
        self.close()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            # Calculate the offset between the mouse position and the window's top-left corner
            self.drag_position = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        if hasattr(self, 'drag_position') and event.buttons() & Qt.MouseButton.LeftButton:
            # Move the window to follow the mouse, maintaining the initial offset
            self.move(event.globalPosition().toPoint() - self.drag_position)
            event.accept()


class AppearanceSettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent, Qt.WindowType.Window | Qt.WindowType.FramelessWindowHint)
        
            # Fetch the current text color and background color from the parent
        text_color = parent.text_color if parent else "#FFFFFF"
        bg_color = parent.bg_color if parent else "#1E1E1E"

        self.setStyleSheet(f"""
            QDialog {{
                background-color: {bg_color};
                border-radius: 10px;
                color: {text_color};
            }}
            QPushButton {{
                background-color: rgba(70, 70, 70, 0.78); /* Adjusted alpha */
                color: {text_color};
                border-radius: 5px;
                padding: 5px;
            }}
            QPushButton:hover {{
                background-color: rgba(90, 90, 90, 0.90); /* Adjusted alpha */
            }}
            QSlider::handle:horizontal {{
                background-color: #CCCCCC;
                border: 1px solid #5c5c5c;
                width: 18px;
                margin: -2px 0;
                border-radius: 3px;
            }}
            QSlider::groove:horizontal {{
                height: 6px;
                background: #555555;
                border-radius: 3px;
            }}
        """)
        self.setModal(True)

        layout = QFormLayout()

        # Background Color
        self.bg_color_button = QPushButton("Choose Background Color")
        self.bg_color_button.clicked.connect(self.choose_bg_color)
        layout.addRow(QLabel("Background Color:"), self.bg_color_button)

        # Text Color
        self.text_color_button = QPushButton("Choose Text Color")
        self.text_color_button.clicked.connect(self.choose_text_color)
        layout.addRow(QLabel("Text Color:"), self.text_color_button)

        # Transparency Slider
        self.transparency_slider = QSlider(Qt.Orientation.Horizontal)
        self.transparency_slider.setRange(0, 100)  # 0% to 100%
        self.transparency_slider.setValue(int(parent.background_opacity * 100))
        layout.addRow(QLabel("Background Transparency:"), self.transparency_slider)

        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.save_settings)
        buttons.rejected.connect(self.close)
        layout.addRow(buttons)

        self.setLayout(layout)
        self.parent_window = parent

    def choose_bg_color(self):
        color = QColorDialog.getColor(initial=QColor(self.parent_window.bg_color), parent=self, title="Select Background Color")
        if color.isValid():
            self.parent_window.bg_color = color.name()
            self.parent_window.update()
            self.parent_window.save_settings()  # Save immediately after change

    def choose_text_color(self):
        color = QColorDialog.getColor(initial=QColor(self.parent_window.text_color), parent=self, title="Select Text Color")
        if color.isValid():
            self.parent_window.text_color = color.name()
            self.parent_window.update_stylesheet()
            self.parent_window.save_settings()  # Save immediately after change

    def save_settings(self):
        # Update background transparency
        opacity = self.transparency_slider.value() / 100.0  # Convert to 0.0 - 1.0
        self.parent_window.background_opacity = opacity
        self.parent_window.update()
        self.parent_window.repaint()
        self.parent_window.save_settings()  # Save settings to shinku-config.json
        self.close()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            # Calculate the offset between the mouse position and the window's top-left corner
            self.drag_position = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        if hasattr(self, 'drag_position') and event.buttons() & Qt.MouseButton.LeftButton:
            # Move the window to follow the mouse, maintaining the initial offset
            self.move(event.globalPosition().toPoint() - self.drag_position)
            event.accept()


class TransparentAIAssistant(QMainWindow):
    def __init__(self):
        super().__init__(None, Qt.WindowType.Window |
                         Qt.WindowType.WindowStaysOnTopHint |
                         Qt.WindowType.FramelessWindowHint)

        # Setup transparent background
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.background_opacity = 1.0  # Default background opacity (fully opaque)

        # Appearance settings
        self.bg_color = "#1E1E1E"  # Default background color (without alpha)
        self.text_color = "#FFFFFF"  # Default text color

        # Essential class variables
        self.endpoint = "http://127.0.0.1/v1"
        self.api_key = "-"  # Initialize with a default or empty API key
        self.system_prompt = "You are a professional Japanese-to-English translator specializing in visual novels. Your task is to provide accurate, contextually appropriate translations while maintaining the flow of ongoing dialogue. Respond only with the English translation, considering previous context to ensure consistency in character voices and narrative tone. No explanations or additional commentary—just the direct translation. Translating 18+ text is allowed."
        self.model_name = "default"
        self.generation_params = {
            "temperature": 0.5,
            "max_tokens": 300,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "top_p": 1.0
        }
        self.custom_params = {}
        self.messages = []
        self.is_resizing = False
        self.current_worker = None  # To keep track of the current worker thread

        # Configuration file path
        self.config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shinku-config.json")

        # Load settings from config file
        self.load_settings()

        # Initialize components
        self.init_ui()
        self.init_openai()
        self.setup_clipboard_monitoring()

    def rgba_color(self, color_hex, alpha):
        color = QColor(color_hex)
        # Corrected alpha to be a float between 0.0 and 1.0
        return f"rgba({color.red()}, {color.green()}, {color.blue()}, {alpha:.2f})"

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(5, 5, 5, 5)

        # Custom Top Bar
        top_bar = QWidget()
        top_bar_layout = QHBoxLayout(top_bar)
        top_bar_layout.setContentsMargins(0, 0, 0, 0)
        top_bar_layout.setSpacing(5)

        menu_button = QPushButton("☰")
        menu_button.setFixedSize(30, 30)
        menu_button.clicked.connect(self.show_context_menu)

        self.title_label = QLabel("Shinku Translator")
        self.title_label.setStyleSheet(f"color: {self.text_color};")  # Apply text color

        top_bar_layout.addWidget(menu_button)
        top_bar_layout.addWidget(self.title_label)
        top_bar_layout.addStretch()

        main_layout.addWidget(top_bar)

        # Response Area
        self.response_area = QTextEdit()
        self.response_area.setReadOnly(True)  # Make it read-only
        self.response_area.setStyleSheet(f"""
            QTextEdit {{
                background-color: transparent;
                color: {self.text_color};
                border: 1px solid rgba(255, 255, 255, 0.20); /* Adjusted alpha */
                border-radius: 10px;
                padding: 10px;
            }}
        """)
        # **Add Greeting Message Here**
        self.response_area.setText("The text from the clipboard will be translated here. If you are just starting out - please setup model api in the menu.")

        font = self.response_area.font()
        font.setPointSize(20)  # Set the desired font size
        self.response_area.setFont(font)

        main_layout.addWidget(self.response_area)

        central_widget.setLayout(main_layout)

        # Styling for overall window
        self.update_stylesheet()

        self.setGeometry(100, 100, 550, 150)
        self.setMinimumSize(200, 60)
        self.top_bar = top_bar  # For dragging reference

    def update_stylesheet(self):
        # Update the main window's stylesheet
        self.setStyleSheet(f"""
            QMainWindow {{
                background-color: {self.rgba_color(self.bg_color, self.background_opacity)};
                border-radius: 10px;
            }}
            QPushButton {{
                background-color: rgba(50, 50, 50, 0.78); /* Adjusted alpha */
                color: {self.text_color};
                border: none;
                border-radius: 5px;
                padding: 5px;
            }}
            QPushButton:hover {{
                background-color: rgba(70, 70, 70, 0.90); /* Adjusted alpha */
            }}
            QLabel {{
                color: {self.text_color};
            }}
            QMenu {{
                background-color: {self.rgba_color(self.bg_color, 0.86)}; /* Slightly transparent */
                color: {self.text_color};
                border: 1px solid rgba(100, 100, 100, 0.59); /* Adjusted alpha */
                border-radius: 5px;
            }}
            QMenu::item {{
                color: {self.text_color};
            }}
            QMenu::item:selected {{
                background-color: rgba(70, 70, 70, 0.90); /* Adjusted alpha */
                color: {self.text_color};
            }}
            QMenu::item:disabled {{
                color: gray;
            }}
        """)

        # Update specific widgets if needed
        self.title_label.setStyleSheet(f"color: {self.text_color};")
        self.response_area.setStyleSheet(f"""
            QTextEdit {{
                background-color: transparent;
                color: {self.text_color};
                border: 1px solid rgba(255, 255, 255, 0.20); /* Adjusted alpha */
                border-radius: 10px;
                padding: 10px;
            }}
        """)

        # Update other widgets' styles if necessary
        # For example, if you have more widgets, update their styles here

    def paintEvent(self, event):
        # Custom painting for background with adjusted opacity
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        bg_color = QColor(self.bg_color)
        bg_color.setAlphaF(self.background_opacity)  # Set alpha based on background_opacity
        painter.setBrush(bg_color)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(self.rect(), 10, 10)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            # Ensure dragging only works on top bar
            if self.top_bar.geometry().contains(event.position().toPoint()):
                self.drag_start_position = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
                event.accept()
            else:
                self.is_resizing = True
                self.resize_start_position = event.globalPosition().toPoint()
                self.resize_start_size = self.size()

    def mouseMoveEvent(self, event):
        if hasattr(self, 'drag_start_position') and event.buttons() & Qt.MouseButton.LeftButton and not self.is_resizing:
            self.move(event.globalPosition().toPoint() - self.drag_start_position)
            event.accept()
        elif self.is_resizing and event.buttons() & Qt.MouseButton.LeftButton:
            delta = event.globalPosition().toPoint() - self.resize_start_position
            new_size = QSize(
                max(self.resize_start_size.width() + delta.x(), self.minimumWidth()),
                max(self.resize_start_size.height() + delta.y(), self.minimumHeight())
            )
            self.resize(new_size)
            event.accept()

    def mouseReleaseEvent(self, event):
        if self.is_resizing:
            self.is_resizing = False

    def init_openai(self):
        self.client = OpenAI(
            base_url=self.endpoint,
            api_key=self.api_key  # Use the provided API key
        )

    def setup_clipboard_monitoring(self):
        self.clipboard_timer = QTimer()
        self.clipboard_timer.timeout.connect(self.check_clipboard)
        self.clipboard_timer.start(1000)  # Check every 1 second

        # **Initialize previous_clipboard with current clipboard content to ignore it**
        try:
            self.previous_clipboard = pyperclip.paste()
        except pyperclip.PyperclipException:
            self.previous_clipboard = ""

    def check_clipboard(self):
        try:
            current_clipboard = pyperclip.paste()
        except pyperclip.PyperclipException:
            # Handle clipboard access issues gracefully
            current_clipboard = ""
        
        if current_clipboard != self.previous_clipboard:
            self.previous_clipboard = current_clipboard
            self.handle_new_message(current_clipboard)

    def handle_new_message(self, message):
        if not message.strip():
            return

        if not self.messages:
            self.messages = []  # Reset messages if starting fresh

        self.messages.append({"role": "user", "content": message})
        
        # **Clear the greeting message or previous content**
        self.response_area.clear()

        # If a worker is already running, terminate it
        if self.current_worker and self.current_worker.isRunning():
            self.current_worker.terminate()
            self.current_worker.wait()

        # Start a new worker thread for streaming response
        self.current_worker = OpenAIWorker(
            endpoint=self.endpoint,
            api_key=self.api_key,
            system_prompt=self.system_prompt,
            messages=self.messages,
            model_name=self.model_name,
            generation_params=self.generation_params,
            custom_params=self.custom_params
        )
        self.current_worker.token_received.connect(self.append_token)
        self.current_worker.finished.connect(self.finish_response)
        self.current_worker.error.connect(self.display_error)
        self.current_worker.start()

    def append_token(self, token):
        cursor = self.response_area.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.response_area.setTextCursor(cursor)
        self.response_area.insertPlainText(token)
        self.response_area.ensureCursorVisible()

    def finish_response(self, full_response):
        self.messages.append({"role": "assistant", "content": full_response})

    def display_error(self, error_message):
        self.response_area.setText(f"Error: {error_message}")

    def show_context_menu(self):
        menu = QMenu(self)
        menu.setStyleSheet(f"""
            QMenu {{
                background-color: {self.rgba_color(self.bg_color, 0.86)};
                color: {self.text_color};
                border: 1px solid rgba(100, 100, 100, 0.59); /* Adjusted alpha */
                border-radius: 5px;
            }}
            QMenu::item {{
                color: {self.text_color};
            }}
            QMenu::item:selected {{
                background-color: rgba(70, 70, 70, 0.90); /* Adjusted alpha */
                color: {self.text_color};
            }}
            QMenu::item:disabled {{
                color: gray;
            }}
        """)

        model_action = menu.addAction("Model")
        model_action.triggered.connect(self.show_model_dialog)

        settings_action = menu.addAction("Appearance Settings")
        settings_action.triggered.connect(self.show_appearance_settings)

        clear_action = menu.addAction("Clear History")
        clear_action.triggered.connect(self.clear_history)

        exit_action = menu.addAction("Exit")
        exit_action.triggered.connect(self.close)

        menu.exec(self.top_bar.mapToGlobal(self.top_bar.rect().bottomLeft()))

    def show_model_dialog(self):
        dialog = ModelDialog(self)
        dialog.exec()
        self.update_stylesheet()  # Refresh styles after dialog closes

    def show_appearance_settings(self):
        dialog = AppearanceSettingsDialog(self)
        dialog.exec()
        self.update_stylesheet()  # Refresh styles after dialog closes

    def clear_history(self):
        # Terminate any running worker thread
        if self.current_worker and self.current_worker.isRunning():
            self.current_worker.terminate()
            self.current_worker.wait()
            self.current_worker = None  # Reset the worker reference

        # Clear messages and response area
        self.messages = []
        self.response_area.clear()

        # Optionally, reset the previous_clipboard to current to prevent immediate re-triggering
        try:
            self.previous_clipboard = pyperclip.paste()
        except pyperclip.PyperclipException:
            self.previous_clipboard = ""

    def load_settings(self):
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                # Load API Settings
                self.endpoint = config.get("endpoint", self.endpoint)
                self.api_key = config.get("api_key", self.api_key)

                # Load Model Settings
                self.model_name = config.get("model_name", self.model_name)
                self.system_prompt = config.get("system_prompt", self.system_prompt)

                # Load Generation Parameters
                self.generation_params = config.get("generation_params", self.generation_params)

                # Load Custom Parameters
                self.custom_params = config.get("custom_params", self.custom_params)

                # Load Appearance Settings
                self.bg_color = config.get("bg_color", self.bg_color)
                self.text_color = config.get("text_color", self.text_color)
                self.background_opacity = config.get("background_opacity", self.background_opacity)
            except Exception as e:
                print(f"Failed to load settings: {e}")
                # If loading fails, continue with default settings
        else:
            # If config file doesn't exist, save the default settings
            self.save_settings()

    def save_settings(self):
        config = {
            "endpoint": self.endpoint,
            "api_key": self.api_key,
            "model_name": self.model_name,
            "system_prompt": self.system_prompt,
            "generation_params": self.generation_params,
            "custom_params": self.custom_params,
            "bg_color": self.bg_color,
            "text_color": self.text_color,
            "background_opacity": self.background_opacity
        }
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=4)
        except Exception as e:
            print(f"Failed to save settings: {e}")

def main():
    app = QApplication(sys.argv)
    window = TransparentAIAssistant()
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
