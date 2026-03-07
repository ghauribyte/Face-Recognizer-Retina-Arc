import customtkinter as ctk
from PIL import Image, ImageDraw, ImageFont
import os
import threading
from tkinter import messagebox
import imagemodel as backend

# Set appearance mode and color theme
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class FaceReconApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        # Window configuration
        self.title("Face Recognition Gallery")
        self.geometry("1600x1000")
        self.minsize(1400, 800)
        
        # Modern color scheme - Sharp and vibrant
        self.bg_color = "#0f0f0f"
        self.sidebar_color = "#1a1a1a"
        self.card_color = "#252525"
        self.card_hover = "#2d2d2d"
        self.accent_color = "#00d4ff"
        self.accent_hover = "#00b8e6"
        self.accent_dark = "#0099cc"
        self.text_primary = "#ffffff"
        self.text_secondary = "#b0b0b0"
        self.text_muted = "#707070"
        self.success_color = "#00ff88"
        self.danger_color = "#ff4444"
        self.warning_color = "#ffaa00"
        
        # Current view state
        self.current_person = None
        self.processing = False
        self.total_files = 0
        self.progress_monitoring = False
        self.image_cache = {}  # Cache for loaded images
        
        # Setup UI
        self.setup_ui()
        
        # Load initial gallery
        self.after(100, self.refresh_gallery)
    
    def setup_ui(self):
        """Setup the main UI layout"""
        # Configure grid weights
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        # Configure main window background
        self.configure(fg_color=self.bg_color)
        
        # Sidebar
        self.create_sidebar()
        
        # Main content area
        self.create_main_content()
    
    def create_sidebar(self):
        """Create the modern sidebar navigation"""
        self.sidebar = ctk.CTkFrame(
            self, 
            width=280, 
            corner_radius=0, 
            fg_color=self.sidebar_color,
            border_width=0
        )
        self.sidebar.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)
        self.sidebar.grid_propagate(False)
        self.sidebar.grid_rowconfigure(6, weight=1)
        self.sidebar.grid_columnconfigure(0, weight=1)
        
        # Modern header with gradient effect
        header_frame = ctk.CTkFrame(
            self.sidebar,
            fg_color="transparent",
            height=120
        )
        header_frame.grid(row=0, column=0, sticky="ew", padx=0, pady=0)
        header_frame.grid_propagate(False)
        
        # Icon and title
        title_label = ctk.CTkLabel(
            header_frame,
            text="👤",
            font=ctk.CTkFont(size=40),
            text_color=self.accent_color
        )
        title_label.grid(row=0, column=0, padx=25, pady=(25, 5), sticky="w")
        
        title_text = ctk.CTkLabel(
            header_frame,
            text="Face Recognition",
            font=ctk.CTkFont(size=26, weight="bold"),
            text_color=self.text_primary
        )
        title_text.grid(row=1, column=0, padx=25, pady=(0, 5), sticky="w")
        
        subtitle_label = ctk.CTkLabel(
            header_frame,
            text="AI-Powered Gallery",
            font=ctk.CTkFont(size=13),
            text_color=self.text_secondary
        )
        subtitle_label.grid(row=2, column=0, padx=25, pady=(0, 20), sticky="w")
        
        # Navigation buttons with modern design
        button_frame = ctk.CTkFrame(
            self.sidebar,
            fg_color="transparent"
        )
        button_frame.grid(row=1, column=0, sticky="ew", padx=15, pady=10)
        button_frame.grid_columnconfigure(0, weight=1)
        
        # Run Recognition Button - Primary action
        self.run_btn = ctk.CTkButton(
            button_frame,
            text="🚀 Run Recognition",
            command=self.run_recognition,
            height=56,
            font=ctk.CTkFont(size=16, weight="bold"),
            corner_radius=14,
            fg_color=self.accent_color,
            hover_color=self.accent_hover,
            text_color="#000000",
            border_width=0
        )
        self.run_btn.grid(row=0, column=0, padx=0, pady=(0, 12), sticky="ew")
        
        # View Gallery Button
        self.gallery_btn = ctk.CTkButton(
            button_frame,
            text="📁 View Gallery",
            command=self.show_gallery,
            height=52,
            font=ctk.CTkFont(size=15),
            corner_radius=12,
            fg_color=self.card_color,
            hover_color=self.card_hover,
            text_color=self.text_primary,
            border_width=0
        )
        self.gallery_btn.grid(row=1, column=0, padx=0, pady=(0, 10), sticky="ew")
        
        # Rename Person Button
        self.rename_btn = ctk.CTkButton(
            button_frame,
            text="✏️ Rename Person",
            command=self.rename_person_dialog,
            height=52,
            font=ctk.CTkFont(size=15),
            corner_radius=12,
            fg_color=self.card_color,
            hover_color=self.card_hover,
            text_color=self.text_primary,
            border_width=0
        )
        self.rename_btn.grid(row=2, column=0, padx=0, pady=(0, 10), sticky="ew")
        
        # Delete Person Button
        self.delete_btn = ctk.CTkButton(
            button_frame,
            text="🗑️ Delete Person",
            command=self.delete_person_dialog,
            height=52,
            font=ctk.CTkFont(size=15),
            corner_radius=12,
            fg_color=self.danger_color,
            hover_color="#ff3333",
            text_color=self.text_primary,
            border_width=0
        )
        self.delete_btn.grid(row=3, column=0, padx=0, pady=(0, 0), sticky="ew")
        
        # Status section
        status_frame = ctk.CTkFrame(
            self.sidebar,
            fg_color=self.card_color,
            corner_radius=12
        )
        status_frame.grid(row=7, column=0, sticky="ew", padx=15, pady=(20, 20))
        status_frame.grid_columnconfigure(0, weight=1)
        
        status_title = ctk.CTkLabel(
            status_frame,
            text="Status",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=self.text_secondary
        )
        status_title.grid(row=0, column=0, padx=15, pady=(12, 5), sticky="w")
        
        self.status_label = ctk.CTkLabel(
            status_frame,
            text="● Ready",
            font=ctk.CTkFont(size=13),
            text_color=self.success_color
        )
        self.status_label.grid(row=1, column=0, padx=15, pady=(0, 12), sticky="w")
    
    def create_main_content(self):
        """Create the main content area with modern design"""
        self.main_frame = ctk.CTkFrame(
            self, 
            corner_radius=0, 
            fg_color=self.bg_color,
            border_width=0
        )
        self.main_frame.grid(row=0, column=1, sticky="nsew", padx=0, pady=0)
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(1, weight=1)  # Content area should expand, not header
        
        # Header bar
        header_bar = ctk.CTkFrame(
            self.main_frame,
            height=70,
            corner_radius=0,
            fg_color=self.sidebar_color,
            border_width=0
        )
        header_bar.grid(row=0, column=0, sticky="ew", padx=0, pady=0)
        header_bar.grid_propagate(False)
        header_bar.grid_columnconfigure(1, weight=1)
        
        # Title in header
        self.header_title = ctk.CTkLabel(
            header_bar,
            text="Gallery",
            font=ctk.CTkFont(size=26, weight="bold"),
            text_color=self.text_primary
        )
        self.header_title.grid(row=0, column=0, padx=30, pady=0, sticky="w")
        
        # Content area
        content_frame = ctk.CTkFrame(
            self.main_frame,
            fg_color="transparent"
        )
        content_frame.grid(row=1, column=0, sticky="nsew", padx=20, pady=(0, 15))
        content_frame.grid_columnconfigure(0, weight=1)
        content_frame.grid_rowconfigure(0, weight=1)
        
        # Create scrollable frame for gallery
        self.gallery_scroll = ctk.CTkScrollableFrame(
            content_frame,
            corner_radius=0,
            fg_color="transparent",
            border_width=0
        )
        self.gallery_scroll.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)
        self.gallery_scroll.grid_columnconfigure(0, weight=1)
        
        # Progress bar (hidden initially)
        self.progress_bar = ctk.CTkProgressBar(
            self.main_frame, 
            height=4, 
            corner_radius=2,
            fg_color=self.card_color,
            progress_color=self.accent_color
        )
        self.progress_bar.grid(row=2, column=0, padx=25, pady=(0, 10), sticky="ew")
        self.progress_bar.set(0)
        self.progress_bar.grid_remove()
        
        # Processing label
        self.processing_label = ctk.CTkLabel(
            self.main_frame,
            text="",
            font=ctk.CTkFont(size=14),
            text_color=self.accent_color
        )
        self.processing_label.grid(row=3, column=0, padx=25, pady=(0, 20))
        self.processing_label.grid_remove()
    
    def refresh_gallery(self):
        """Refresh the gallery view with modern cards"""
        try:
            # Clear existing widgets
            for widget in self.gallery_scroll.winfo_children():
                widget.destroy()
            
            # Get all persons
            persons = backend.get_all_persons()
            
            if not persons:
                # Show beautiful empty state
                empty_frame = ctk.CTkFrame(
                    self.gallery_scroll,
                    fg_color="transparent"
                )
                empty_frame.grid(row=0, column=0, pady=150, sticky="n")
                
                empty_icon = ctk.CTkLabel(
                    empty_frame,
                    text="📷",
                    font=ctk.CTkFont(size=80),
                    text_color=self.text_muted
                )
                empty_icon.grid(row=0, column=0, pady=(0, 20))
                
                empty_label = ctk.CTkLabel(
                    empty_frame,
                    text="No persons found",
                    font=ctk.CTkFont(size=22, weight="bold"),
                    text_color=self.text_secondary
                )
                empty_label.grid(row=1, column=0, pady=(0, 10))
                
                empty_subtitle = ctk.CTkLabel(
                    empty_frame,
                    text="Click 'Run Recognition' to process photos",
                    font=ctk.CTkFont(size=14),
                    text_color=self.text_muted
                )
                empty_subtitle.grid(row=2, column=0)
                return
            
            # Update header
            self.header_title.configure(text=f"Gallery ({len(persons)} persons)")
            
            # Create grid layout
            cols = 5
            row = 0
            col = 0
            
            # Build cards in batches for better performance
            batch_size = 10
            for i, person in enumerate(persons):
                self.create_person_card(person, row, col)
                col += 1
                if col >= cols:
                    col = 0
                    row += 1
                # Update UI every batch_size items to keep it responsive
                if (i + 1) % batch_size == 0:
                    self.update_idletasks()
            
            # Configure grid columns
            for i in range(cols):
                self.gallery_scroll.grid_columnconfigure(i, weight=1, uniform="cols")
                
        except Exception as e:
            # Show error state
            error_label = ctk.CTkLabel(
                self.gallery_scroll,
                text=f"Error loading gallery:\n{str(e)}",
                font=ctk.CTkFont(size=14),
                text_color=self.danger_color
            )
            error_label.grid(row=0, column=0, pady=100)
    
    def create_person_card(self, person, row, col):
        """Create a modern person card"""
        card = ctk.CTkFrame(
            self.gallery_scroll,
            corner_radius=16,
            fg_color=self.card_color,
            border_width=1,
            border_color="#333333"
        )
        card.grid(row=row, column=col, padx=10, pady=10, sticky="nsew")
        card.grid_columnconfigure(0, weight=1)
        card.grid_rowconfigure(0, weight=0)
        card.grid_rowconfigure(1, weight=0)
        card.grid_rowconfigure(2, weight=0)
        
        # Thumbnail container
        thumbnail_frame = ctk.CTkFrame(
            card, 
            corner_radius=12, 
            fg_color="#1a1a1a"
        )
        thumbnail_frame.grid(row=0, column=0, padx=12, pady=(12, 8), sticky="ew")
        thumbnail_frame.grid_columnconfigure(0, weight=1)
        
        if person['thumbnail'] and os.path.exists(person['thumbnail']):
            try:
                # Use cache if available
                cache_key = person['thumbnail']
                if cache_key not in self.image_cache:
                    img = Image.open(person['thumbnail'])
                    img.thumbnail((180, 180), Image.Resampling.LANCZOS)  # Smaller for faster loading
                    self.image_cache[cache_key] = img
                else:
                    img = self.image_cache[cache_key]
                
                # Create rounded corners
                mask = Image.new('L', img.size, 0)
                draw = ImageDraw.Draw(mask)
                draw.rounded_rectangle([(0, 0), img.size], radius=12, fill=255)
                
                photo = ctk.CTkImage(light_image=img, dark_image=img, size=img.size)
                img_label = ctk.CTkLabel(thumbnail_frame, image=photo, text="")
                img_label.grid(row=0, column=0, padx=8, pady=8)
            except Exception:
                placeholder = ctk.CTkLabel(
                    thumbnail_frame,
                    text="📷",
                    font=ctk.CTkFont(size=40),
                    text_color=self.text_muted
                )
                placeholder.grid(row=0, column=0, padx=10, pady=10)
        else:
            placeholder = ctk.CTkLabel(
                thumbnail_frame,
                text="📷",
                font=ctk.CTkFont(size=40),
                text_color=self.text_muted
            )
            placeholder.grid(row=0, column=0, padx=10, pady=10)
        
        # Person name
        name_label = ctk.CTkLabel(
            card,
            text=person['name'],
            font=ctk.CTkFont(size=16, weight="bold"),
            anchor="w",
            text_color=self.text_primary
        )
        name_label.grid(row=1, column=0, padx=12, pady=(0, 4), sticky="w")
        
        # Photo count
        count_label = ctk.CTkLabel(
            card,
            text=f"{person['photo_count']} photo{'s' if person['photo_count'] != 1 else ''}",
            font=ctk.CTkFont(size=12),
            text_color=self.text_secondary,
            anchor="w"
        )
        count_label.grid(row=2, column=0, padx=12, pady=(0, 12), sticky="w")
        
        # Make card clickable with hover effect
        def on_card_click(event):
            self.show_person_detail(person['name'])
        
        def on_enter(e):
            card.configure(fg_color=self.card_hover)
        
        def on_leave(e):
            card.configure(fg_color=self.card_color)
        
        for widget in [card, thumbnail_frame, name_label, count_label]:
            widget.bind("<Button-1>", on_card_click)
            widget.bind("<Enter>", on_enter)
            widget.bind("<Leave>", on_leave)
            if hasattr(widget, 'configure'):
                widget.configure(cursor="hand2")
    
    def show_person_detail(self, person_name):
        """Show all photos for a specific person"""
        try:
            self.current_person = person_name
            
            # Clear gallery
            for widget in self.gallery_scroll.winfo_children():
                widget.destroy()
            
            # Update header
            self.header_title.configure(text=person_name)
            
            # Action buttons frame
            action_frame = ctk.CTkFrame(
                self.gallery_scroll,
                fg_color="transparent"
            )
            action_frame.grid(row=0, column=0, columnspan=5, padx=0, pady=(0, 20), sticky="ew")
            action_frame.grid_columnconfigure(0, weight=1)
            
            # Left side - Back button
            back_btn = ctk.CTkButton(
                action_frame,
                text="← Back to Gallery",
                command=self.show_gallery,
                height=42,
                font=ctk.CTkFont(size=14),
                corner_radius=10,
                fg_color=self.card_color,
                hover_color=self.card_hover,
                text_color=self.text_primary,
                border_width=0
            )
            back_btn.grid(row=0, column=0, sticky="w")
            
            # Right side - Action buttons
            btn_frame = ctk.CTkFrame(action_frame, fg_color="transparent")
            btn_frame.grid(row=0, column=1, sticky="e")
            
            rename_btn = ctk.CTkButton(
                btn_frame,
                text="✏️ Rename",
                command=lambda: self.rename_person_dialog(selected_person=person_name),
                height=42,
                width=120,
                font=ctk.CTkFont(size=13),
                corner_radius=10,
                fg_color=self.accent_color,
                hover_color=self.accent_hover,
                text_color="#000000",
                border_width=0
            )
            rename_btn.grid(row=0, column=0, padx=(0, 10))
            
            delete_btn = ctk.CTkButton(
                btn_frame,
                text="🗑️ Delete",
                command=lambda: self.delete_person_dialog(selected_person=person_name),
                height=42,
                width=120,
                font=ctk.CTkFont(size=13),
                corner_radius=10,
                fg_color=self.danger_color,
                hover_color="#ff3333",
                text_color=self.text_primary,
                border_width=0
            )
            delete_btn.grid(row=0, column=1)
            
            # Get all images for this person
            images = backend.get_person_images(person_name)
            
            if not images:
                empty_label = ctk.CTkLabel(
                    self.gallery_scroll,
                    text="No photos found for this person.",
                    font=ctk.CTkFont(size=16),
                    text_color=self.text_secondary
                )
                empty_label.grid(row=1, column=0, columnspan=5, pady=50, sticky="n")
                return
            
            # Display images in grid
            cols = 5
            row = 1
            col = 0
            
            # Build image cards in batches for better performance
            batch_size = 10
            for i, img_path in enumerate(images):
                self.create_image_card(img_path, person_name, row, col)
                col += 1
                if col >= cols:
                    col = 0
                    row += 1
                # Update UI every batch_size items to keep it responsive
                if (i + 1) % batch_size == 0:
                    self.update_idletasks()
            
            # Configure grid columns
            for i in range(cols):
                self.gallery_scroll.grid_columnconfigure(i, weight=1, uniform="cols")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load person details:\n{str(e)}")
    
    def create_image_card(self, img_path, person_name, row, col):
        """Create an image card with delete button"""
        card = ctk.CTkFrame(
            self.gallery_scroll,
            corner_radius=12,
            fg_color=self.card_color,
            border_width=1,
            border_color="#333333"
        )
        card.grid(row=row, column=col, padx=10, pady=10, sticky="ew")
        card.grid_columnconfigure(0, weight=1)
        
        # Image container
        img_container = ctk.CTkFrame(card, corner_radius=8, fg_color="#1a1a1a")
        img_container.grid(row=0, column=0, padx=8, pady=(8, 5), sticky="nsew")
        img_container.grid_columnconfigure(0, weight=1)
        
        try:
            # Use cache if available
            cache_key = img_path
            if cache_key not in self.image_cache:
                img = Image.open(img_path)
                img.thumbnail((180, 180), Image.Resampling.LANCZOS)  # Smaller for faster loading
                self.image_cache[cache_key] = img
            else:
                img = self.image_cache[cache_key]
            
            photo = ctk.CTkImage(light_image=img, dark_image=img, size=img.size)
            img_label = ctk.CTkLabel(img_container, image=photo, text="")
            img_label.grid(row=0, column=0, padx=5, pady=5)
        except Exception:
            error_label = ctk.CTkLabel(
                img_container,
                text="Error",
                font=ctk.CTkFont(size=12),
                text_color=self.danger_color
            )
            error_label.grid(row=0, column=0, padx=5, pady=5)
        
        # Delete button overlay
        delete_btn = ctk.CTkButton(
            card,
            text="🗑️",
            width=36,
            height=36,
            font=ctk.CTkFont(size=16),
            corner_radius=8,
            fg_color=self.danger_color,
            hover_color="#ff3333",
            command=lambda: self.delete_image(img_path, person_name),
            border_width=0
        )
        delete_btn.grid(row=0, column=0, padx=(8, 8), pady=(8, 8), sticky="ne")
    
    def delete_image(self, img_path, person_name):
        """Delete an individual image"""
        image_name = os.path.basename(img_path)
        result = messagebox.askyesno(
            "Delete Image",
            f"Are you sure you want to delete '{image_name}'?",
            icon="warning"
        )
        
        if result:
            try:
                if backend.delete_image(person_name, image_name):
                    # Clear cache for deleted image
                    if img_path in self.image_cache:
                        del self.image_cache[img_path]
                    messagebox.showinfo("Success", f"Image '{image_name}' deleted successfully.")
                    self.show_person_detail(person_name)
                else:
                    messagebox.showerror("Error", "Failed to delete image.")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to delete image:\n{str(e)}")
    
    def show_gallery(self):
        """Show the main gallery view"""
        try:
            self.current_person = None
            self.header_title.configure(text="Gallery")
            # Clear cache when switching views
            self.image_cache.clear()
            self.refresh_gallery()
            self.status_label.configure(text="● Gallery refreshed", text_color=self.success_color)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to show gallery:\n{str(e)}")
    
    def run_recognition(self):
        """Run face recognition in a separate thread"""
        if self.processing:
            messagebox.showinfo("Processing", "Recognition is already running. Please wait.")
            return
            
        if not os.path.exists(backend.NEW_PHOTOS):
            os.makedirs(backend.NEW_PHOTOS, exist_ok=True)
        
        # Check if folder is empty
        if backend.is_folder_empty(backend.NEW_PHOTOS):
            messagebox.showwarning(
                "No Photos",
                f"No images found in '{backend.NEW_PHOTOS}' folder.\nPlease add photos to process."
            )
            return
        
        # Count total files for progress tracking
        files = [
            f for f in os.listdir(backend.NEW_PHOTOS)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
        ]
        self.total_files = len(files)
        
        # Disable button and show progress
        self.processing = True
        self.progress_monitoring = True
        self.run_btn.configure(state="disabled", text="⏳ Processing...")
        self.progress_bar.grid()
        self.progress_bar.set(0)
        self.processing_label.grid()
        self.processing_label.configure(text="🔄 Processing photos...")
        self.status_label.configure(text="● Processing...", text_color=self.warning_color)
        
        # Start progress monitoring thread
        progress_thread = threading.Thread(target=self.monitor_progress, daemon=True)
        progress_thread.start()
        
        # Run in separate thread
        thread = threading.Thread(target=self.process_recognition_thread, daemon=True)
        thread.start()
    
    def monitor_progress(self):
        """Monitor progress by checking remaining files"""
        import time
        while self.progress_monitoring and self.processing:
            try:
                if os.path.exists(backend.NEW_PHOTOS):
                    files = [
                        f for f in os.listdir(backend.NEW_PHOTOS)
                        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
                    ]
                    remaining_files = len(files)
                    
                    if self.total_files > 0:
                        processed = self.total_files - remaining_files
                        progress = min(processed / self.total_files, 0.95)  # Cap at 95% until done
                        self.after(0, lambda p=progress: self.progress_bar.set(p))
                        self.after(0, lambda p=processed, total=self.total_files: self.processing_label.configure(
                            text=f"🔄 Processing photos... ({p}/{total})"
                        ))
                
                time.sleep(0.5)  # Check every 0.5 seconds
            except:
                pass
    
    def process_recognition_thread(self):
        """Process recognition in background thread"""
        try:
            # Run the recognition process
            backend.process_folder()
            
            # Export to CSV
            backend.export_to_csv()
            
            # Stop progress monitoring
            self.progress_monitoring = False
            
            # Update UI in main thread
            self.after(0, self.recognition_complete)
        except Exception as e:
            self.progress_monitoring = False
            self.after(0, lambda: self.recognition_error(str(e)))
    
    def recognition_complete(self):
        """Called when recognition is complete"""
        self.processing = False
        self.progress_monitoring = False
        self.progress_bar.set(1.0)  # Complete the progress bar
        self.run_btn.configure(state="normal", text="🚀 Run Recognition")
        self.after(100, lambda: self.progress_bar.grid_remove())
        self.after(100, lambda: self.processing_label.grid_remove())
        self.status_label.configure(text="● Ready", text_color=self.success_color)
        
        messagebox.showinfo("Success", "Face recognition completed successfully!")
        
        # Clear cache and refresh gallery
        self.image_cache.clear()
        self.refresh_gallery()
    
    def recognition_error(self, error_msg):
        """Called when recognition encounters an error"""
        self.processing = False
        self.progress_monitoring = False
        self.run_btn.configure(state="normal", text="🚀 Run Recognition")
        self.progress_bar.grid_remove()
        self.processing_label.grid_remove()
        self.status_label.configure(text="● Error", text_color=self.danger_color)
        
        messagebox.showerror("Error", f"An error occurred:\n{error_msg}")
    
    def rename_person_dialog(self, selected_person=None):
        """Show modern dialog to rename a person"""
        try:
            persons = backend.get_all_persons()
            
            if not persons:
                messagebox.showinfo("No Persons", "No persons found to rename.")
                return
            
            # Create modern dialog window
            dialog = ctk.CTkToplevel(self)
            dialog.title("Rename Person")
            dialog.geometry("550x380")
            dialog.transient(self)
            dialog.grab_set()
            dialog.focus()
            dialog.configure(fg_color=self.sidebar_color)
            
            # Center the dialog
            dialog.update_idletasks()
            x = (dialog.winfo_screenwidth() // 2) - (550 // 2)
            y = (dialog.winfo_screenheight() // 2) - (380 // 2)
            dialog.geometry(f"550x380+{x}+{y}")
            
            # Header
            header = ctk.CTkLabel(
                dialog,
                text="✏️ Rename Person",
                font=ctk.CTkFont(size=24, weight="bold"),
                text_color=self.text_primary
            )
            header.grid(row=0, column=0, padx=30, pady=(30, 20))
            
            # Person selection
            ctk.CTkLabel(
                dialog,
                text="Select Person:",
                font=ctk.CTkFont(size=14, weight="bold"),
                text_color=self.text_secondary
            ).grid(row=1, column=0, padx=30, pady=(10, 8), sticky="w")
            
            person_names = [p['name'] for p in persons]
            # Use selected person if provided, otherwise use first person
            initial_person = selected_person if selected_person and selected_person in person_names else (person_names[0] if person_names else "")
            person_var = ctk.StringVar(value=initial_person)
            person_menu = ctk.CTkOptionMenu(
                dialog,
                values=person_names,
                variable=person_var,
                width=490,
                height=45,
                corner_radius=10,
                fg_color=self.card_color,
                button_color=self.accent_color,
                button_hover_color=self.accent_hover,
                text_color=self.text_primary
            )
            person_menu.grid(row=2, column=0, padx=30, pady=(0, 20), sticky="ew")
            
            # New name input
            ctk.CTkLabel(
                dialog,
                text="New Name:",
                font=ctk.CTkFont(size=14, weight="bold"),
                text_color=self.text_secondary
            ).grid(row=3, column=0, padx=30, pady=(10, 8), sticky="w")
            
            new_name_entry = ctk.CTkEntry(
                dialog, 
                width=490, 
                height=45, 
                font=ctk.CTkFont(size=14),
                corner_radius=10,
                fg_color=self.card_color,
                border_color="#444444",
                text_color=self.text_primary
            )
            new_name_entry.grid(row=4, column=0, padx=30, pady=(0, 30), sticky="ew")
            new_name_entry.insert(0, initial_person)
            new_name_entry.select_range(0, "end")
            new_name_entry.focus()
            
            def rename_action():
                old_name = person_var.get()
                new_name = new_name_entry.get().strip()
                
                if not new_name:
                    messagebox.showwarning("Invalid Name", "Please enter a new name.")
                    return
                
                if new_name == old_name:
                    messagebox.showinfo("Same Name", "New name is the same as the old name.")
                    return
                
                if new_name in person_names:
                    messagebox.showwarning("Name Exists", "A person with this name already exists.")
                    return
                
                try:
                    backend.rename_person(old_name, new_name)
                    # Clear cache after rename
                    self.image_cache.clear()
                    messagebox.showinfo("Success", f"Renamed '{old_name}' to '{new_name}' successfully.")
                    dialog.destroy()
                    self.refresh_gallery()
                    if self.current_person == old_name:
                        self.current_person = new_name
                        self.show_person_detail(new_name)
                    self.status_label.configure(text="● Renamed successfully", text_color=self.success_color)
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to rename person:\n{str(e)}")
            
            # Buttons
            button_frame = ctk.CTkFrame(dialog, fg_color="transparent")
            button_frame.grid(row=5, column=0, padx=30, pady=(0, 30), sticky="ew")
            button_frame.grid_columnconfigure(0, weight=1)
            button_frame.grid_columnconfigure(1, weight=1)
            
            ctk.CTkButton(
                button_frame,
                text="Cancel",
                command=dialog.destroy,
                height=45,
                font=ctk.CTkFont(size=15),
                corner_radius=10,
                fg_color=self.card_color,
                hover_color=self.card_hover,
                text_color=self.text_primary,
                border_width=0
            ).grid(row=0, column=0, padx=(0, 10), sticky="ew")
            
            ctk.CTkButton(
                button_frame,
                text="Rename",
                command=rename_action,
                height=45,
                font=ctk.CTkFont(size=15, weight="bold"),
                corner_radius=10,
                fg_color=self.accent_color,
                hover_color=self.accent_hover,
                text_color="#000000",
                border_width=0
            ).grid(row=0, column=1, padx=(10, 0), sticky="ew")
            
            dialog.grid_columnconfigure(0, weight=1)
            
            # Make Enter key trigger rename
            new_name_entry.bind("<Return>", lambda e: rename_action())
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open rename dialog:\n{str(e)}")
    
    def delete_person_dialog(self, selected_person=None):
        """Show modern dialog to delete a person"""
        try:
            persons = backend.get_all_persons()
            
            if not persons:
                messagebox.showinfo("No Persons", "No persons found to delete.")
                return
            
            # Create modern dialog window
            dialog = ctk.CTkToplevel(self)
            dialog.title("Delete Person")
            dialog.geometry("550x320")
            dialog.transient(self)
            dialog.grab_set()
            dialog.focus()
            dialog.configure(fg_color=self.sidebar_color)
            
            # Center the dialog
            dialog.update_idletasks()
            x = (dialog.winfo_screenwidth() // 2) - (550 // 2)
            y = (dialog.winfo_screenheight() // 2) - (320 // 2)
            dialog.geometry(f"550x320+{x}+{y}")
            
            # Warning header
            header = ctk.CTkLabel(
                dialog,
                text="⚠️ Delete Person",
                font=ctk.CTkFont(size=24, weight="bold"),
                text_color=self.danger_color
            )
            header.grid(row=0, column=0, padx=30, pady=(30, 10))
            
            # Warning label
            warning_label = ctk.CTkLabel(
                dialog,
                text="This will permanently delete the person and all their photos!",
                font=ctk.CTkFont(size=14),
                text_color=self.text_secondary,
                wraplength=490
            )
            warning_label.grid(row=1, column=0, padx=30, pady=(0, 20))
            
            # Person selection
            ctk.CTkLabel(
                dialog,
                text="Select Person to Delete:",
                font=ctk.CTkFont(size=14, weight="bold"),
                text_color=self.text_secondary
            ).grid(row=2, column=0, padx=30, pady=(10, 8), sticky="w")
            
            person_names = [p['name'] for p in persons]
            # Use selected person if provided, otherwise use first person
            initial_person = selected_person if selected_person and selected_person in person_names else (person_names[0] if person_names else "")
            person_var = ctk.StringVar(value=initial_person)
            person_menu = ctk.CTkOptionMenu(
                dialog,
                values=person_names,
                variable=person_var,
                width=490,
                height=45,
                corner_radius=10,
                fg_color=self.card_color,
                button_color=self.danger_color,
                button_hover_color="#ff3333",
                text_color=self.text_primary
            )
            person_menu.grid(row=3, column=0, padx=30, pady=(0, 30), sticky="ew")
            
            def delete_action():
                person_name = person_var.get()
                
                result = messagebox.askyesno(
                    "Confirm Delete",
                    f"Are you sure you want to delete '{person_name}' and all their photos?\n\nThis action cannot be undone!",
                    icon="warning"
                )
                
                if result:
                    try:
                        backend.delete_person(person_name)
                        # Clear cache after delete
                        self.image_cache.clear()
                        messagebox.showinfo("Success", f"Deleted '{person_name}' successfully.")
                        dialog.destroy()
                        if self.current_person == person_name:
                            self.current_person = None
                            self.show_gallery()
                        else:
                            self.refresh_gallery()
                        self.status_label.configure(text="● Deleted successfully", text_color=self.success_color)
                    except Exception as e:
                        messagebox.showerror("Error", f"Failed to delete person:\n{str(e)}")
            
            # Buttons
            button_frame = ctk.CTkFrame(dialog, fg_color="transparent")
            button_frame.grid(row=4, column=0, padx=30, pady=(0, 30), sticky="ew")
            button_frame.grid_columnconfigure(0, weight=1)
            button_frame.grid_columnconfigure(1, weight=1)
            
            ctk.CTkButton(
                button_frame,
                text="Cancel",
                command=dialog.destroy,
                height=45,
                font=ctk.CTkFont(size=15),
                corner_radius=10,
                fg_color=self.card_color,
                hover_color=self.card_hover,
                text_color=self.text_primary,
                border_width=0
            ).grid(row=0, column=0, padx=(0, 10), sticky="ew")
            
            ctk.CTkButton(
                button_frame,
                text="Delete",
                command=delete_action,
                height=45,
                font=ctk.CTkFont(size=15, weight="bold"),
                corner_radius=10,
                fg_color=self.danger_color,
                hover_color="#ff3333",
                text_color=self.text_primary,
                border_width=0
            ).grid(row=0, column=1, padx=(10, 0), sticky="ew")
            
            dialog.grid_columnconfigure(0, weight=1)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open delete dialog:\n{str(e)}")


if __name__ == "__main__":
    app = FaceReconApp()
    app.mainloop()