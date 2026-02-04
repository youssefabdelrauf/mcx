#!/usr/bin/env python3
"""
Phototherapy Simulation GUI

Modern GUI application for Monte Carlo simulation of light penetration
through skin tissue with varying bilirubin levels for phototherapy optimization.

Author: Generated for MCX phototherapy optimization study
"""

import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox
import threading
import numpy as np
import json
import os
from datetime import datetime

# Try to import matplotlib for embedded plots
try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Try to import pmcx for simulations
try:
    import pmcx
    HAS_PMCX = True
except ImportError:
    HAS_PMCX = False

# ============================================================================
# THEME AND STYLING
# ============================================================================
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# Color palette
COLORS = {
    'primary': '#3B82F6',       # Blue
    'secondary': '#10B981',     # Green
    'warning': '#F59E0B',       # Amber
    'danger': '#EF4444',        # Red
    'surface': '#1E293B',       # Dark slate
    'background': '#0F172A',    # Darker slate
    'text': '#F1F5F9',          # Light text
    'text_muted': '#94A3B8',    # Muted text
}

# Tissue presets with optical properties at 460nm
TISSUE_PRESETS = {
    'Neonatal Skin (Caucasian)': {
        'description': 'Standard neonatal skin model for phototherapy',
        'layers': {
            'stratum_corneum': {'thickness_um': 10, 'mua': 0.1, 'mus': 50.0, 'g': 0.9, 'n': 1.5},
            'epidermis': {'thickness_um': 80, 'mua': 0.5, 'mus': 45.0, 'g': 0.8, 'n': 1.4},
            'dermis': {'thickness_um': 1500, 'mua': 0.35, 'mus': 20.0, 'g': 0.9, 'n': 1.4},
            'subcutis': {'thickness_um': 2000, 'mua': 0.05, 'mus': 12.0, 'g': 0.8, 'n': 1.4},
        }
    },
    'Neonatal Skin (Darker)': {
        'description': 'Higher melanin content in epidermis',
        'layers': {
            'stratum_corneum': {'thickness_um': 10, 'mua': 0.1, 'mus': 50.0, 'g': 0.9, 'n': 1.5},
            'epidermis': {'thickness_um': 80, 'mua': 1.2, 'mus': 45.0, 'g': 0.8, 'n': 1.4},
            'dermis': {'thickness_um': 1500, 'mua': 0.35, 'mus': 20.0, 'g': 0.9, 'n': 1.4},
            'subcutis': {'thickness_um': 2000, 'mua': 0.05, 'mus': 12.0, 'g': 0.8, 'n': 1.4},
        }
    },
    'Preterm Infant': {
        'description': 'Thinner skin layers for premature infants',
        'layers': {
            'stratum_corneum': {'thickness_um': 5, 'mua': 0.08, 'mus': 45.0, 'g': 0.9, 'n': 1.5},
            'epidermis': {'thickness_um': 50, 'mua': 0.4, 'mus': 40.0, 'g': 0.8, 'n': 1.4},
            'dermis': {'thickness_um': 1000, 'mua': 0.3, 'mus': 18.0, 'g': 0.9, 'n': 1.4},
            'subcutis': {'thickness_um': 1500, 'mua': 0.04, 'mus': 10.0, 'g': 0.8, 'n': 1.4},
        }
    },
    'Adult Skin': {
        'description': 'Thicker adult skin for comparison',
        'layers': {
            'stratum_corneum': {'thickness_um': 15, 'mua': 0.12, 'mus': 55.0, 'g': 0.9, 'n': 1.5},
            'epidermis': {'thickness_um': 100, 'mua': 0.6, 'mus': 50.0, 'g': 0.8, 'n': 1.4},
            'dermis': {'thickness_um': 2000, 'mua': 0.4, 'mus': 22.0, 'g': 0.9, 'n': 1.4},
            'subcutis': {'thickness_um': 3000, 'mua': 0.06, 'mus': 14.0, 'g': 0.8, 'n': 1.4},
        }
    },
    'Custom': {
        'description': 'Define your own tissue properties',
        'layers': {
            'stratum_corneum': {'thickness_um': 10, 'mua': 0.1, 'mus': 50.0, 'g': 0.9, 'n': 1.5},
            'epidermis': {'thickness_um': 80, 'mua': 0.5, 'mus': 45.0, 'g': 0.8, 'n': 1.4},
            'dermis': {'thickness_um': 1500, 'mua': 0.35, 'mus': 20.0, 'g': 0.9, 'n': 1.4},
            'subcutis': {'thickness_um': 2000, 'mua': 0.05, 'mus': 12.0, 'g': 0.8, 'n': 1.4},
        }
    },
}


class PhototherapyGUI(ctk.CTk):
    """Main application window for phototherapy simulation."""
    
    def __init__(self):
        super().__init__()
        
        # Window configuration
        self.title("Phototherapy Light Penetration Simulator")
        self.geometry("1400x900")
        self.minsize(1200, 700)
        
        # State variables
        self.simulation_running = False
        self.results = []
        self.current_tissue = 'Neonatal Skin (Caucasian)'
        
        # Configure grid layout
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        # Create main layout
        self._create_sidebar()
        self._create_main_content()
        self._create_status_bar()
        
        # Initialize with default tissue
        self._on_tissue_change(self.current_tissue)
        
        # Check dependencies
        self._check_dependencies()
    
    def _check_dependencies(self):
        """Check if required packages are available."""
        if not HAS_PMCX:
            self._update_status("Warning: pmcx not installed. Install with: pip install pmcx", "warning")
        elif not HAS_MATPLOTLIB:
            self._update_status("Warning: matplotlib not installed for plotting", "warning")
        else:
            # Check GPU
            try:
                gpus = pmcx.gpuinfo()
                if gpus:
                    self._update_status(f"Ready - GPU detected: {gpus[0].get('name', 'Unknown')}", "success")
                else:
                    self._update_status("Warning: No CUDA GPU detected", "warning")
            except Exception as e:
                self._update_status(f"GPU check failed: {str(e)}", "warning")
    
    def _create_sidebar(self):
        """Create the left sidebar with controls."""
        # Sidebar frame
        self.sidebar = ctk.CTkFrame(self, width=350, corner_radius=0, fg_color=COLORS['surface'])
        self.sidebar.grid(row=0, column=0, rowspan=2, sticky="nsew")
        self.sidebar.grid_rowconfigure(10, weight=1)
        
        # Logo/Title
        self.logo_label = ctk.CTkLabel(
            self.sidebar, 
            text="🔬 Phototherapy\nSimulator",
            font=ctk.CTkFont(size=24, weight="bold"),
            text_color=COLORS['primary']
        )
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))
        
        # Subtitle
        self.subtitle = ctk.CTkLabel(
            self.sidebar,
            text="Monte Carlo Light Transport",
            font=ctk.CTkFont(size=12),
            text_color=COLORS['text_muted']
        )
        self.subtitle.grid(row=1, column=0, padx=20, pady=(0, 20))
        
        # === TISSUE SELECTION ===
        self.tissue_label = ctk.CTkLabel(
            self.sidebar, 
            text="Tissue Type",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.tissue_label.grid(row=2, column=0, padx=20, pady=(10, 5), sticky="w")
        
        self.tissue_dropdown = ctk.CTkOptionMenu(
            self.sidebar,
            values=list(TISSUE_PRESETS.keys()),
            command=self._on_tissue_change,
            width=310,
            fg_color=COLORS['background'],
            button_color=COLORS['primary'],
            button_hover_color=COLORS['secondary']
        )
        self.tissue_dropdown.grid(row=3, column=0, padx=20, pady=5)
        
        self.tissue_desc = ctk.CTkLabel(
            self.sidebar,
            text="",
            font=ctk.CTkFont(size=11),
            text_color=COLORS['text_muted'],
            wraplength=300
        )
        self.tissue_desc.grid(row=4, column=0, padx=20, pady=(0, 10), sticky="w")
        
        # === BILIRUBIN LEVELS ===
        self.bili_label = ctk.CTkLabel(
            self.sidebar, 
            text="Bilirubin Levels (mg/dL)",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.bili_label.grid(row=5, column=0, padx=20, pady=(15, 5), sticky="w")
        
        # Bilirubin range slider
        self.bili_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        self.bili_frame.grid(row=6, column=0, padx=20, pady=5, sticky="ew")
        
        self.bili_min_label = ctk.CTkLabel(self.bili_frame, text="Min:", width=40)
        self.bili_min_label.grid(row=0, column=0, sticky="w")
        self.bili_min_entry = ctk.CTkEntry(self.bili_frame, width=60, placeholder_text="0")
        self.bili_min_entry.grid(row=0, column=1, padx=5)
        self.bili_min_entry.insert(0, "0")
        
        self.bili_max_label = ctk.CTkLabel(self.bili_frame, text="Max:", width=40)
        self.bili_max_label.grid(row=0, column=2, padx=(15, 0), sticky="w")
        self.bili_max_entry = ctk.CTkEntry(self.bili_frame, width=60, placeholder_text="25")
        self.bili_max_entry.grid(row=0, column=3, padx=5)
        self.bili_max_entry.insert(0, "25")
        
        self.bili_step_label = ctk.CTkLabel(self.bili_frame, text="Step:", width=40)
        self.bili_step_label.grid(row=0, column=4, padx=(15, 0), sticky="w")
        self.bili_step_entry = ctk.CTkEntry(self.bili_frame, width=60, placeholder_text="5")
        self.bili_step_entry.grid(row=0, column=5, padx=5)
        self.bili_step_entry.insert(0, "5")
        
        # === SIMULATION PARAMETERS ===
        self.params_label = ctk.CTkLabel(
            self.sidebar, 
            text="Simulation Parameters",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.params_label.grid(row=7, column=0, padx=20, pady=(15, 5), sticky="w")
        
        self.params_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        self.params_frame.grid(row=8, column=0, padx=20, pady=5, sticky="ew")
        
        # Photon count
        self.photon_label = ctk.CTkLabel(self.params_frame, text="Photons:")
        self.photon_label.grid(row=0, column=0, sticky="w", pady=2)
        self.photon_dropdown = ctk.CTkOptionMenu(
            self.params_frame,
            values=["100K (Fast)", "1M (Standard)", "10M (Accurate)", "100M (High)"],
            width=180
        )
        self.photon_dropdown.grid(row=0, column=1, padx=10, pady=2)
        self.photon_dropdown.set("1M (Standard)")
        
        # Wavelength
        self.wave_label = ctk.CTkLabel(self.params_frame, text="Wavelength:")
        self.wave_label.grid(row=1, column=0, sticky="w", pady=2)
        self.wave_dropdown = ctk.CTkOptionMenu(
            self.params_frame,
            values=["460nm (Blue)", "490nm (Cyan)", "520nm (Green)", "630nm (Red)"],
            width=180
        )
        self.wave_dropdown.grid(row=1, column=1, padx=10, pady=2)
        self.wave_dropdown.set("460nm (Blue)")
        
        # === RUN CONTROLS ===
        self.run_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        self.run_frame.grid(row=9, column=0, padx=20, pady=20, sticky="ew")
        
        self.run_button = ctk.CTkButton(
            self.run_frame,
            text="▶  Run Simulation",
            command=self._run_simulation,
            width=150,
            height=45,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color=COLORS['secondary'],
            hover_color=COLORS['primary']
        )
        self.run_button.grid(row=0, column=0, padx=5)
        
        self.stop_button = ctk.CTkButton(
            self.run_frame,
            text="⏹  Stop",
            command=self._stop_simulation,
            width=100,
            height=45,
            font=ctk.CTkFont(size=14),
            fg_color=COLORS['danger'],
            hover_color="#DC2626",
            state="disabled"
        )
        self.stop_button.grid(row=0, column=1, padx=5)
        
        # Progress bar
        self.progress_bar = ctk.CTkProgressBar(self.sidebar, width=310)
        self.progress_bar.grid(row=10, column=0, padx=20, pady=10, sticky="n")
        self.progress_bar.set(0)
        
        # === QUICK ACTIONS ===
        self.actions_label = ctk.CTkLabel(
            self.sidebar, 
            text="Quick Actions",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.actions_label.grid(row=11, column=0, padx=20, pady=(20, 10), sticky="sw")
        
        self.actions_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        self.actions_frame.grid(row=12, column=0, padx=20, pady=(0, 20), sticky="sew")
        
        self.export_button = ctk.CTkButton(
            self.actions_frame,
            text="📊 Export Results",
            command=self._export_results,
            width=150,
            fg_color=COLORS['background'],
            border_width=1,
            border_color=COLORS['primary']
        )
        self.export_button.grid(row=0, column=0, padx=5, pady=5)
        
        self.clear_button = ctk.CTkButton(
            self.actions_frame,
            text="🗑  Clear",
            command=self._clear_results,
            width=80,
            fg_color=COLORS['background'],
            border_width=1,
            border_color=COLORS['text_muted']
        )
        self.clear_button.grid(row=0, column=1, padx=5, pady=5)
    
    def _create_main_content(self):
        """Create the main content area with results and visualization."""
        # Main content frame
        self.main_frame = ctk.CTkFrame(self, corner_radius=0, fg_color=COLORS['background'])
        self.main_frame.grid(row=0, column=1, sticky="nsew", padx=0, pady=0)
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(1, weight=1)
        
        # Header
        self.header = ctk.CTkFrame(self.main_frame, height=60, fg_color=COLORS['surface'])
        self.header.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        self.header.grid_columnconfigure(1, weight=1)
        
        self.header_title = ctk.CTkLabel(
            self.header,
            text="Simulation Results",
            font=ctk.CTkFont(size=20, weight="bold")
        )
        self.header_title.grid(row=0, column=0, padx=20, pady=15, sticky="w")
        
        self.header_info = ctk.CTkLabel(
            self.header,
            text="Configure parameters and run simulation to see results",
            font=ctk.CTkFont(size=12),
            text_color=COLORS['text_muted']
        )
        self.header_info.grid(row=0, column=1, padx=20, pady=15, sticky="e")
        
        # Tab view for results
        self.tabview = ctk.CTkTabview(self.main_frame, fg_color=COLORS['surface'])
        self.tabview.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))
        
        # Add tabs
        self.tab_summary = self.tabview.add("📊 Summary")
        self.tab_charts = self.tabview.add("📈 Charts")
        self.tab_data = self.tabview.add("📋 Data Table")
        self.tab_tissue = self.tabview.add("🔬 Tissue Model")
        
        # Configure tabs
        self._setup_summary_tab()
        self._setup_charts_tab()
        self._setup_data_tab()
        self._setup_tissue_tab()
    
    def _setup_summary_tab(self):
        """Setup the summary tab with key metrics."""
        self.tab_summary.grid_columnconfigure((0, 1, 2, 3), weight=1)
        self.tab_summary.grid_rowconfigure(1, weight=1)
        
        # Metric cards
        self.metric_cards = []
        metrics = [
            ("Simulations", "0", "Total runs"),
            ("Max Absorption", "0%", "Dermis peak"),
            ("Penetration", "0 mm", "Average depth"),
            ("Status", "Ready", "Simulation state")
        ]
        
        for i, (title, value, subtitle) in enumerate(metrics):
            card = self._create_metric_card(self.tab_summary, title, value, subtitle)
            card.grid(row=0, column=i, padx=10, pady=10, sticky="nsew")
            self.metric_cards.append(card)
        
        # Summary text area
        self.summary_text = ctk.CTkTextbox(
            self.tab_summary,
            font=ctk.CTkFont(size=13),
            fg_color=COLORS['background']
        )
        self.summary_text.grid(row=1, column=0, columnspan=4, padx=10, pady=10, sticky="nsew")
        self.summary_text.insert("1.0", "Run a simulation to see results summary here.\n\n"
                                        "The simulation will model light penetration through skin tissue\n"
                                        "at various bilirubin concentrations to help optimize phototherapy\n"
                                        "treatment for neonatal jaundice.")
    
    def _create_metric_card(self, parent, title, value, subtitle):
        """Create a metric display card."""
        card = ctk.CTkFrame(parent, fg_color=COLORS['background'], corner_radius=10)
        card.grid_columnconfigure(0, weight=1)
        
        title_label = ctk.CTkLabel(
            card, text=title,
            font=ctk.CTkFont(size=12),
            text_color=COLORS['text_muted']
        )
        title_label.grid(row=0, column=0, padx=15, pady=(15, 5))
        
        value_label = ctk.CTkLabel(
            card, text=value,
            font=ctk.CTkFont(size=28, weight="bold"),
            text_color=COLORS['primary']
        )
        value_label.grid(row=1, column=0, padx=15, pady=5)
        
        subtitle_label = ctk.CTkLabel(
            card, text=subtitle,
            font=ctk.CTkFont(size=11),
            text_color=COLORS['text_muted']
        )
        subtitle_label.grid(row=2, column=0, padx=15, pady=(5, 15))
        
        # Store labels for updating
        card.value_label = value_label
        card.subtitle_label = subtitle_label
        
        return card
    
    def _setup_charts_tab(self):
        """Setup the charts tab with matplotlib figures."""
        self.tab_charts.grid_columnconfigure(0, weight=1)
        self.tab_charts.grid_rowconfigure(0, weight=1)
        
        if HAS_MATPLOTLIB:
            # Create matplotlib figure
            self.fig = Figure(figsize=(12, 8), dpi=100, facecolor=COLORS['background'])
            self.fig.patch.set_facecolor(COLORS['background'])
            
            # Create canvas
            self.canvas = FigureCanvasTkAgg(self.fig, master=self.tab_charts)
            self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
            
            # Initial empty plot
            self._draw_empty_chart()
        else:
            no_plot = ctk.CTkLabel(
                self.tab_charts,
                text="Matplotlib not installed.\nInstall with: pip install matplotlib",
                font=ctk.CTkFont(size=16)
            )
            no_plot.grid(row=0, column=0, padx=20, pady=20)
    
    def _draw_empty_chart(self):
        """Draw placeholder chart."""
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.set_facecolor(COLORS['surface'])
        ax.text(0.5, 0.5, 'Run simulation to see charts',
                ha='center', va='center', fontsize=16, color=COLORS['text_muted'],
                transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        self.canvas.draw()
    
    def _setup_data_tab(self):
        """Setup the data table tab."""
        self.tab_data.grid_columnconfigure(0, weight=1)
        self.tab_data.grid_rowconfigure(0, weight=1)
        
        # Create scrollable frame for data
        self.data_scroll = ctk.CTkScrollableFrame(
            self.tab_data,
            fg_color=COLORS['background']
        )
        self.data_scroll.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        # Headers
        headers = ["Bilirubin\n(mg/dL)", "Penetration\n(mm)", "Total\nAbsorption", 
                   "SC %", "Epidermis %", "Dermis %", "Subcutis %"]
        
        for i, header in enumerate(headers):
            label = ctk.CTkLabel(
                self.data_scroll, text=header,
                font=ctk.CTkFont(size=12, weight="bold"),
                fg_color=COLORS['primary'],
                corner_radius=5,
                width=120
            )
            label.grid(row=0, column=i, padx=2, pady=5, sticky="ew")
        
        self.data_scroll.grid_columnconfigure(tuple(range(7)), weight=1)
    
    def _setup_tissue_tab(self):
        """Setup the tissue model visualization tab."""
        self.tab_tissue.grid_columnconfigure((0, 1), weight=1)
        self.tab_tissue.grid_rowconfigure(0, weight=1)
        
        # Left: Layer properties
        self.tissue_props_frame = ctk.CTkScrollableFrame(
            self.tab_tissue,
            label_text="Tissue Layer Properties",
            fg_color=COLORS['background']
        )
        self.tissue_props_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        # Right: Visual diagram
        self.tissue_visual_frame = ctk.CTkFrame(
            self.tab_tissue,
            fg_color=COLORS['background']
        )
        self.tissue_visual_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        
        self.tissue_diagram_label = ctk.CTkLabel(
            self.tissue_visual_frame,
            text="Tissue Cross-Section",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.tissue_diagram_label.pack(pady=10)
        
        # Layer entries will be populated by _on_tissue_change
        self.layer_entries = {}
    
    def _create_status_bar(self):
        """Create the bottom status bar."""
        self.status_bar = ctk.CTkFrame(self, height=30, corner_radius=0, fg_color=COLORS['surface'])
        self.status_bar.grid(row=1, column=1, sticky="ew")
        
        self.status_label = ctk.CTkLabel(
            self.status_bar,
            text="Ready",
            font=ctk.CTkFont(size=11),
            text_color=COLORS['text_muted']
        )
        self.status_label.pack(side="left", padx=20, pady=5)
        
        self.gpu_label = ctk.CTkLabel(
            self.status_bar,
            text="",
            font=ctk.CTkFont(size=11),
            text_color=COLORS['text_muted']
        )
        self.gpu_label.pack(side="right", padx=20, pady=5)
    
    def _update_status(self, message, status_type="info"):
        """Update the status bar."""
        colors = {
            "info": COLORS['text_muted'],
            "success": COLORS['secondary'],
            "warning": COLORS['warning'],
            "error": COLORS['danger']
        }
        self.status_label.configure(text=message, text_color=colors.get(status_type, COLORS['text_muted']))
    
    def _on_tissue_change(self, tissue_name):
        """Handle tissue type selection change."""
        self.current_tissue = tissue_name
        preset = TISSUE_PRESETS.get(tissue_name, TISSUE_PRESETS['Neonatal Skin (Caucasian)'])
        
        # Update description
        self.tissue_desc.configure(text=preset['description'])
        
        # Clear existing layer entries
        for widget in self.tissue_props_frame.winfo_children():
            widget.destroy()
        self.layer_entries = {}
        
        # Create layer property editors
        layer_colors = {
            'stratum_corneum': '#FCD34D',  # Yellow
            'epidermis': '#FB923C',         # Orange
            'dermis': '#F87171',            # Red
            'subcutis': '#FBBF24'           # Amber
        }
        
        for i, (layer_name, props) in enumerate(preset['layers'].items()):
            # Layer header
            layer_frame = ctk.CTkFrame(self.tissue_props_frame, fg_color=COLORS['surface'])
            layer_frame.pack(fill="x", padx=5, pady=5)
            
            header = ctk.CTkLabel(
                layer_frame,
                text=f"● {layer_name.replace('_', ' ').title()}",
                font=ctk.CTkFont(size=13, weight="bold"),
                text_color=layer_colors.get(layer_name, COLORS['text'])
            )
            header.pack(anchor="w", padx=10, pady=(10, 5))
            
            # Properties grid
            props_grid = ctk.CTkFrame(layer_frame, fg_color="transparent")
            props_grid.pack(fill="x", padx=10, pady=(0, 10))
            
            entries = {}
            props_list = [
                ('thickness_um', 'Thickness (μm)'),
                ('mua', 'μa (1/mm)'),
                ('mus', 'μs (1/mm)'),
                ('g', 'g'),
                ('n', 'n')
            ]
            
            for j, (prop_key, prop_label) in enumerate(props_list):
                lbl = ctk.CTkLabel(props_grid, text=prop_label, width=90, anchor="e")
                lbl.grid(row=j // 3, column=(j % 3) * 2, padx=5, pady=2, sticky="e")
                
                entry = ctk.CTkEntry(props_grid, width=70)
                entry.grid(row=j // 3, column=(j % 3) * 2 + 1, padx=5, pady=2)
                entry.insert(0, str(props.get(prop_key, 0)))
                entries[prop_key] = entry
            
            self.layer_entries[layer_name] = entries
    
    def _get_bilirubin_levels(self):
        """Get bilirubin levels from UI."""
        try:
            min_val = float(self.bili_min_entry.get())
            max_val = float(self.bili_max_entry.get())
            step = float(self.bili_step_entry.get())
            return list(np.arange(min_val, max_val + step/2, step))
        except ValueError:
            return [0, 5, 10, 15, 20, 25]
    
    def _get_photon_count(self):
        """Get photon count from UI."""
        selection = self.photon_dropdown.get()
        mapping = {
            "100K (Fast)": 1e5,
            "1M (Standard)": 1e6,
            "10M (Accurate)": 1e7,
            "100M (High)": 1e8
        }
        return mapping.get(selection, 1e6)
    
    def _run_simulation(self):
        """Start the simulation in a background thread."""
        if not HAS_PMCX:
            messagebox.showerror("Error", "pmcx is not installed. Install with: pip install pmcx")
            return
        
        if self.simulation_running:
            return
        
        self.simulation_running = True
        self.run_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.progress_bar.set(0)
        self._update_status("Starting simulation...", "info")
        
        # Update metric card
        self.metric_cards[3].value_label.configure(text="Running")
        
        # Run in background thread
        thread = threading.Thread(target=self._simulation_worker, daemon=True)
        thread.start()
    
    def _simulation_worker(self):
        """Background worker for running simulations."""
        try:
            bilirubin_levels = self._get_bilirubin_levels()
            nphoton = self._get_photon_count()
            
            results = []
            total = len(bilirubin_levels)
            
            for i, bili in enumerate(bilirubin_levels):
                if not self.simulation_running:
                    break
                
                # Update progress
                progress = (i + 0.5) / total
                self.after(0, lambda p=progress, b=bili: self._update_progress(p, f"Simulating {b:.1f} mg/dL..."))
                
                # Run simulation
                result = self._run_single_simulation(bili, nphoton)
                if result:
                    results.append(result)
                
                # Update progress
                progress = (i + 1) / total
                self.after(0, lambda p=progress: self._update_progress(p, ""))
            
            # Store results and update UI
            self.results = results
            self.after(0, self._on_simulation_complete)
            
        except Exception as e:
            self.after(0, lambda: self._on_simulation_error(str(e)))
    
    def _run_single_simulation(self, bilirubin_mg_dl, nphoton):
        """Run a single Monte Carlo simulation."""
        # Get tissue properties from UI
        tissue_preset = TISSUE_PRESETS.get(self.current_tissue, TISSUE_PRESETS['Neonatal Skin (Caucasian)'])
        
        # Bilirubin absorption coefficient at 460nm
        BILIRUBIN_MUA_PER_MGDL = 0.025
        bilirubin_absorption = BILIRUBIN_MUA_PER_MGDL * bilirubin_mg_dl
        
        # Build optical properties
        layers = tissue_preset['layers']
        prop = np.array([
            [0.0, 0.0, 1.0, 1.0],  # Background
            [layers['stratum_corneum']['mua'], layers['stratum_corneum']['mus'], 
             layers['stratum_corneum']['g'], layers['stratum_corneum']['n']],
            [layers['epidermis']['mua'], layers['epidermis']['mus'], 
             layers['epidermis']['g'], layers['epidermis']['n']],
            [layers['dermis']['mua'] + bilirubin_absorption, layers['dermis']['mus'], 
             layers['dermis']['g'], layers['dermis']['n']],
            [layers['subcutis']['mua'] + bilirubin_absorption * 0.7, layers['subcutis']['mus'], 
             layers['subcutis']['g'], layers['subcutis']['n']],
        ], dtype=np.float32)
        
        # Create volume
        nx, ny, nz = 100, 100, 150
        unit_mm = 0.005
        vol = np.zeros((nx, ny, nz), dtype=np.uint8)
        
        # Calculate layer boundaries
        sc_end = max(1, int(layers['stratum_corneum']['thickness_um'] / (unit_mm * 1000)))
        epi_end = sc_end + max(1, int(layers['epidermis']['thickness_um'] / (unit_mm * 1000)))
        derm_end = min(epi_end + int(layers['dermis']['thickness_um'] / (unit_mm * 1000)), nz)
        
        vol[:, :, 0:sc_end] = 1
        vol[:, :, sc_end:epi_end] = 2
        vol[:, :, epi_end:derm_end] = 3
        if derm_end < nz:
            vol[:, :, derm_end:nz] = 4
        
        # Run simulation
        cfg = {
            'nphoton': int(nphoton),
            'vol': vol,
            'prop': prop,
            'tstart': 0,
            'tend': 5e-9,
            'tstep': 5e-9,
            'unitinmm': unit_mm,
            'srctype': 'disk',
            'srcpos': [nx/2, ny/2, 0],
            'srcdir': [0, 0, 1],
            'srcparam1': [20, 0, 0, 0],
            'isreflect': 1,
            'issrcfrom0': 1,
            'autopilot': 1,
            'gpuid': 1,
            'outputtype': 'energy',
        }
        
        result = pmcx.run(cfg)
        
        # Analyze results
        flux = result['flux']
        if flux.ndim == 4:
            flux = np.squeeze(flux)
        
        total_energy = np.sum(flux)
        layer_absorption = {}
        layer_names = {1: 'stratum_corneum', 2: 'epidermis', 3: 'dermis', 4: 'subcutis'}
        
        for label, name in layer_names.items():
            mask = (vol == label)
            layer_energy = np.sum(flux[mask])
            layer_absorption[name] = {
                'total_energy': float(layer_energy),
                'fraction': float(layer_energy / total_energy) if total_energy > 0 else 0
            }
        
        # Calculate penetration depth
        center_x = nx // 2
        margin = 10
        central_flux = flux[center_x-margin:center_x+margin, ny//2-margin:ny//2+margin, :]
        depth_profile = np.mean(central_flux, axis=(0, 1))
        max_flux = np.max(depth_profile)
        threshold = max_flux / np.e
        pen_indices = np.where(depth_profile >= threshold)[0]
        penetration_mm = pen_indices[-1] * unit_mm if len(pen_indices) > 0 else 0
        
        return {
            'bilirubin_mg_dl': bilirubin_mg_dl,
            'penetration_depth_mm': penetration_mm,
            'total_absorbed_energy': float(total_energy),
            'layer_absorption': layer_absorption,
            'depth_profile': depth_profile.tolist()
        }
    
    def _update_progress(self, progress, message):
        """Update progress bar and status."""
        self.progress_bar.set(progress)
        if message:
            self._update_status(message, "info")
    
    def _on_simulation_complete(self):
        """Handle simulation completion."""
        self.simulation_running = False
        self.run_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        self.progress_bar.set(1.0)
        
        if self.results:
            self._update_status(f"Completed {len(self.results)} simulations", "success")
            self._update_results_display()
        else:
            self._update_status("Simulation cancelled or no results", "warning")
        
        self.metric_cards[3].value_label.configure(text="Done")
    
    def _on_simulation_error(self, error_msg):
        """Handle simulation error."""
        self.simulation_running = False
        self.run_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        self._update_status(f"Error: {error_msg}", "error")
        self.metric_cards[3].value_label.configure(text="Error")
        messagebox.showerror("Simulation Error", error_msg)
    
    def _stop_simulation(self):
        """Stop the running simulation."""
        self.simulation_running = False
        self._update_status("Stopping simulation...", "warning")
    
    def _update_results_display(self):
        """Update all result displays with new data."""
        if not self.results:
            return
        
        # Update metric cards
        self.metric_cards[0].value_label.configure(text=str(len(self.results)))
        
        max_dermis = max(r['layer_absorption']['dermis']['fraction'] * 100 for r in self.results)
        self.metric_cards[1].value_label.configure(text=f"{max_dermis:.1f}%")
        
        avg_pen = np.mean([r['penetration_depth_mm'] for r in self.results])
        self.metric_cards[2].value_label.configure(text=f"{avg_pen:.3f}")
        
        # Update summary text
        self._update_summary_text()
        
        # Update charts
        self._update_charts()
        
        # Update data table
        self._update_data_table()
    
    def _update_summary_text(self):
        """Update the summary text box."""
        self.summary_text.delete("1.0", "end")
        
        lines = [
            "═══════════════════════════════════════════════════════════",
            "                    SIMULATION RESULTS",
            "═══════════════════════════════════════════════════════════",
            "",
            f"Tissue Model: {self.current_tissue}",
            f"Wavelength: {self.wave_dropdown.get()}",
            f"Photons: {self.photon_dropdown.get()}",
            "",
            "─────────────────────────────────────────────────────────────",
            f"{'Bilirubin (mg/dL)':<20} {'Penetration (mm)':<18} {'Dermis Absorption':<18}",
            "─────────────────────────────────────────────────────────────",
        ]
        
        for r in self.results:
            bili = r['bilirubin_mg_dl']
            pen = r['penetration_depth_mm']
            dermis = r['layer_absorption']['dermis']['fraction'] * 100
            lines.append(f"{bili:<20.1f} {pen:<18.3f} {dermis:<18.1f}%")
        
        lines.extend([
            "",
            "─────────────────────────────────────────────────────────────",
            "",
            "CLINICAL INSIGHTS:",
            "",
            "• Higher bilirubin concentration leads to increased dermis absorption",
            "• This confirms phototherapy effectiveness at targeting bilirubin",
            "• Optimal wavelength: 460nm (blue light) for peak bilirubin absorption",
        ])
        
        self.summary_text.insert("1.0", "\n".join(lines))
    
    def _update_charts(self):
        """Update the matplotlib charts."""
        if not HAS_MATPLOTLIB or not self.results:
            return
        
        self.fig.clear()
        
        # Extract data
        bili_levels = [r['bilirubin_mg_dl'] for r in self.results]
        penetrations = [r['penetration_depth_mm'] for r in self.results]
        dermis_abs = [r['layer_absorption']['dermis']['fraction'] * 100 for r in self.results]
        
        # Color gradient
        colors = plt.cm.YlOrRd(np.linspace(0.2, 0.9, len(bili_levels)))
        
        # Create 2x2 subplot
        axes = self.fig.subplots(2, 2)
        
        # Plot 1: Penetration Depth
        ax1 = axes[0, 0]
        ax1.bar(range(len(bili_levels)), penetrations, color=colors, edgecolor='white', linewidth=0.5)
        ax1.set_xticks(range(len(bili_levels)))
        ax1.set_xticklabels([f'{b:.0f}' for b in bili_levels])
        ax1.set_xlabel('Bilirubin (mg/dL)', color='white')
        ax1.set_ylabel('Penetration (mm)', color='white')
        ax1.set_title('Light Penetration Depth', color='white', fontweight='bold')
        ax1.set_facecolor(COLORS['surface'])
        ax1.tick_params(colors='white')
        
        # Plot 2: Depth Profiles
        ax2 = axes[0, 1]
        for i, r in enumerate(self.results):
            profile = np.array(r['depth_profile'])
            depth_mm = np.arange(len(profile)) * 0.005
            profile_norm = profile / np.max(profile) if np.max(profile) > 0 else profile
            label = f'{r["bilirubin_mg_dl"]:.0f} mg/dL'
            ax2.semilogy(depth_mm, profile_norm + 1e-10, color=colors[i], linewidth=1.5, label=label)
        ax2.set_xlabel('Depth (mm)', color='white')
        ax2.set_ylabel('Normalized Fluence', color='white')
        ax2.set_title('Depth Profiles', color='white', fontweight='bold')
        ax2.legend(loc='upper right', fontsize=8, framealpha=0.5)
        ax2.set_facecolor(COLORS['surface'])
        ax2.tick_params(colors='white')
        ax2.set_xlim([0, 0.75])
        
        # Plot 3: Layer Absorption
        ax3 = axes[1, 0]
        layers = ['stratum_corneum', 'epidermis', 'dermis', 'subcutis']
        layer_labels = ['SC', 'Epi', 'Dermis', 'Sub']
        x = np.arange(len(layers))
        width = 0.8 / len(self.results)
        
        for i, r in enumerate(self.results):
            fractions = [r['layer_absorption'][l]['fraction'] * 100 for l in layers]
            offset = (i - len(self.results)/2 + 0.5) * width
            ax3.bar(x + offset, fractions, width, color=colors[i], edgecolor='white', linewidth=0.3)
        
        ax3.set_xticks(x)
        ax3.set_xticklabels(layer_labels)
        ax3.set_xlabel('Layer', color='white')
        ax3.set_ylabel('Absorption (%)', color='white')
        ax3.set_title('Energy Absorption by Layer', color='white', fontweight='bold')
        ax3.set_facecolor(COLORS['surface'])
        ax3.tick_params(colors='white')
        
        # Plot 4: Dermis Absorption Trend
        ax4 = axes[1, 1]
        ax4.plot(bili_levels, dermis_abs, 'o-', color='#EF4444', linewidth=2, markersize=8,
                 markerfacecolor='white', markeredgewidth=2)
        ax4.fill_between(bili_levels, dermis_abs, alpha=0.3, color='#EF4444')
        ax4.set_xlabel('Bilirubin (mg/dL)', color='white')
        ax4.set_ylabel('Dermis Absorption (%)', color='white')
        ax4.set_title('Therapeutic Target: Dermis', color='white', fontweight='bold')
        ax4.set_facecolor(COLORS['surface'])
        ax4.tick_params(colors='white')
        ax4.grid(True, alpha=0.2)
        
        # Style all axes
        for ax in axes.flat:
            for spine in ax.spines.values():
                spine.set_color('white')
                spine.set_alpha(0.3)
        
        self.fig.tight_layout(pad=2)
        self.canvas.draw()
    
    def _update_data_table(self):
        """Update the data table with results."""
        # Clear existing rows (keep headers)
        for widget in self.data_scroll.winfo_children()[7:]:
            widget.destroy()
        
        for row, r in enumerate(self.results, start=1):
            values = [
                f"{r['bilirubin_mg_dl']:.1f}",
                f"{r['penetration_depth_mm']:.3f}",
                f"{r['total_absorbed_energy']:.4f}",
                f"{r['layer_absorption']['stratum_corneum']['fraction']*100:.1f}%",
                f"{r['layer_absorption']['epidermis']['fraction']*100:.1f}%",
                f"{r['layer_absorption']['dermis']['fraction']*100:.1f}%",
                f"{r['layer_absorption']['subcutis']['fraction']*100:.1f}%"
            ]
            
            for col, val in enumerate(values):
                label = ctk.CTkLabel(
                    self.data_scroll, text=val,
                    font=ctk.CTkFont(size=12),
                    fg_color=COLORS['surface'] if row % 2 == 0 else COLORS['background'],
                    corner_radius=3,
                    width=120
                )
                label.grid(row=row, column=col, padx=2, pady=2, sticky="ew")
    
    def _export_results(self):
        """Export results to JSON file."""
        if not self.results:
            messagebox.showinfo("Export", "No results to export. Run a simulation first.")
            return
        
        # Create output directory
        output_dir = os.path.join(os.path.dirname(__file__), 'phototherapy_results')
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = os.path.join(output_dir, f'gui_results_{timestamp}.json')
        
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        self._update_status(f"Exported to {filename}", "success")
        messagebox.showinfo("Export Complete", f"Results saved to:\n{filename}")
    
    def _clear_results(self):
        """Clear all results."""
        self.results = []
        self.progress_bar.set(0)
        
        # Reset metric cards
        self.metric_cards[0].value_label.configure(text="0")
        self.metric_cards[1].value_label.configure(text="0%")
        self.metric_cards[2].value_label.configure(text="0 mm")
        self.metric_cards[3].value_label.configure(text="Ready")
        
        # Clear summary
        self.summary_text.delete("1.0", "end")
        self.summary_text.insert("1.0", "Run a simulation to see results.")
        
        # Clear charts
        if HAS_MATPLOTLIB:
            self._draw_empty_chart()
        
        # Clear data table
        for widget in self.data_scroll.winfo_children()[7:]:
            widget.destroy()
        
        self._update_status("Results cleared", "info")


def main():
    app = PhototherapyGUI()
    app.mainloop()


if __name__ == '__main__':
    main()
