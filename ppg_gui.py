#!/usr/bin/env python3
"""
PPG Analysis GUI
================

Graphical User Interface for PPG Sensor Optimization and Melanin Bias Analysis.
Allows users to:
1. Configure simulation parameters (wavelengths, skin types, photon count)
2. Run Monte Carlo simulations
3. Visualize results (Intensity profiles, Signal strength, Bias heatmaps)

Author: Generated for PPG optimization study
"""

import sys
import os
import json
import numpy as np
from datetime import datetime

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QComboBox, QSpinBox, 
                             QDoubleSpinBox, QPushButton, QCheckBox, QGroupBox, 
                             QTabWidget, QTextEdit, QProgressBar, QSplitter,
                             QScrollArea, QFileDialog, QMessageBox, QTableWidget,
                             QTableWidgetItem, QHeaderView)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QIcon

# Matplotlib integration
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

# Import simulation logic
# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ppg_simulation import run_full_analysis, save_results, get_wavelength_range, FITZPATRICK_TYPES
from ppg_analysis import run_analysis, load_results

# Import optimization functions
try:
    from ppg_optimization import (
        find_optimal_wavelength, find_optimal_spo2_pair,
        optimize_sdd, co_optimize_wavelength_sdd, run_full_optimization
    )
    from ppg_analysis import generate_optimization_summary_figures
    HAS_OPTIMIZATION = True
except ImportError:
    HAS_OPTIMIZATION = False
    print("Warning: Optimization module not available")

class SimulationWorker(QThread):
    """Worker thread for running simulations without freezing GUI"""
    progress_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(object)
    error_signal = pyqtSignal(str)
    
    def __init__(self, mode, nphoton, test_mode, custom_wavelengths=None):
        super().__init__()
        self.mode = mode
        self.nphoton = nphoton
        self.test_mode = test_mode
        self.custom_wavelengths = custom_wavelengths
        self.is_running = True
        
    def run(self):
        try:
            self.progress_signal.emit(f"Initializing simulation... (Mode: {self.mode})")
            
            # Determine wavelengths
            if self.mode == 'custom' and self.custom_wavelengths:
                wavelengths = self.custom_wavelengths
            else:
                wavelengths = get_wavelength_range(self.mode)
            
            self.progress_signal.emit(f"Wavelengths: {wavelengths}")
            self.progress_signal.emit(f"Skin Types: I-VI")
            self.progress_signal.emit(f"Photons: {self.nphoton:.0e}")
            
            # Redirect stdout to capture print statements
            # Note: This is a simple capture; for real-time log, we'd need more complex handling
            # relying on progress updates for now
            
            results = run_full_analysis(
                wavelengths=wavelengths,
                nphoton=self.nphoton,
                test_mode=self.test_mode
            )
            
            if self.is_running:
                self.progress_signal.emit("Simulation complete. Saving results...")
                # Save results automatically
                output_dir = 'ppg_results'
                results_file, _ = save_results(results, output_dir)
                self.progress_signal.emit(f"Results saved to {results_file}")
                
                self.finished_signal.emit(results)
                
        except Exception as e:
            import traceback
            error_msg = f"Simulation failed: {str(e)}\n{traceback.format_exc()}"
            self.error_signal.emit(error_msg)

    def stop(self):
        self.is_running = False


class OptimizationWorker(QThread):
    """Worker thread for running optimization without freezing GUI"""
    progress_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(object)
    error_signal = pyqtSignal(str)
    
    def __init__(self, opt_type='full', test_mode=True):
        super().__init__()
        self.opt_type = opt_type  # 'wavelength', 'spo2', 'sdd', 'co_opt', 'full'
        self.test_mode = test_mode
        self.is_running = True
        
    def run(self):
        try:
            self.progress_signal.emit(f"Starting optimization: {self.opt_type}")
            
            results = {}
            
            if self.opt_type == 'wavelength':
                self.progress_signal.emit("Finding optimal wavelength...")
                results = find_optimal_wavelength(test_mode=self.test_mode)
                
            elif self.opt_type == 'spo2':
                self.progress_signal.emit("Finding optimal SpO2 pair...")
                results = find_optimal_spo2_pair(test_mode=self.test_mode)
                
            elif self.opt_type == 'sdd':
                self.progress_signal.emit("Optimizing Source-Detector Distance...")
                results = optimize_sdd(wavelength=880, test_mode=self.test_mode)
                
            elif self.opt_type == 'co_opt':
                self.progress_signal.emit("Running joint wavelength-SDD optimization...")
                results = co_optimize_wavelength_sdd(test_mode=self.test_mode)
                
            elif self.opt_type == 'full':
                self.progress_signal.emit("Running full optimization pipeline...")
                results = run_full_optimization(test_mode=self.test_mode)
            
            if self.is_running:
                self.progress_signal.emit("Optimization complete!")
                self.finished_signal.emit(results)
                
        except Exception as e:
            import traceback
            error_msg = f"Optimization failed: {str(e)}\n{traceback.format_exc()}"
            self.error_signal.emit(error_msg)

    def stop(self):
        self.is_running = False


class PlotCanvas(FigureCanvas):
    """Matplotlib canvas for PyQt5"""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self,
                                   os.sys.modules['PyQt5.QtWidgets'].QSizePolicy.Expanding,
                                   os.sys.modules['PyQt5.QtWidgets'].QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.worker = None
        self.results = None
        
        # Load existing results if available
        self.check_existing_results()

    def initUI(self):
        self.setWindowTitle('PPG Sensor Optimization & Melanin Bias Analyzer')
        self.setGeometry(100, 100, 1400, 900)
        
        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        
        # Splitter for resizable panels
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # === LEFT PANEL: CONTROLS ===
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(10, 10, 10, 10)
        
        # Configuration Group
        config_group = QGroupBox("Simulation Parameters")
        config_layout = QVBoxLayout()
        
        # Wavelength Selection
        config_layout.addWidget(QLabel("Wavelength Mode:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(['Standard (660, 880, 940 nm)', 'Full Sweep (500-1000 nm)', 'Extended', 'Custom'])
        self.mode_combo.currentIndexChanged.connect(self.toggle_custom_wavelengths)
        config_layout.addWidget(self.mode_combo)
        
        self.custom_wl_input = QTextEdit()
        self.custom_wl_input.setPlaceholderText("Enter wavelengths (comma separated, e.g., 660, 940)")
        self.custom_wl_input.setMaximumHeight(50)
        self.custom_wl_input.setVisible(False)
        config_layout.addWidget(self.custom_wl_input)
        
        # Photon Count
        config_layout.addWidget(QLabel("Photon Count (per sim):"))
        self.photon_spin = QDoubleSpinBox()
        self.photon_spin.setRange(1e4, 1e9)
        self.photon_spin.setValue(1e6)
        self.photon_spin.setSingleStep(1e5)
        self.photon_spin.setDecimals(0)
        config_layout.addWidget(self.photon_spin)
        
        # Options
        self.test_mode_check = QCheckBox("Test Mode (Fast, Low Quality)")
        self.test_mode_check.setChecked(True)
        config_layout.addWidget(self.test_mode_check)
        
        config_group.setLayout(config_layout)
        left_layout.addWidget(config_group)
        
        # Run Button
        self.run_btn = QPushButton("RUN SIMULATION")
        self.run_btn.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold; padding: 10px;")
        self.run_btn.clicked.connect(self.run_simulation)
        left_layout.addWidget(self.run_btn)
        
        # Progress
        self.progress_bar = QProgressBar()
        left_layout.addWidget(self.progress_bar)
        
        # Log
        left_layout.addWidget(QLabel("Simulation Log:"))
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        left_layout.addWidget(self.log_text)
        
        # Add left panel to splitter
        left_panel_scroll = QScrollArea()
        left_panel_scroll.setWidget(left_panel)
        left_panel_scroll.setWidgetResizable(True)
        left_panel_scroll.setMinimumWidth(300)
        left_panel_scroll.setMaximumWidth(400)
        splitter.addWidget(left_panel_scroll)
        
        # === RIGHT PANEL: VISUALIZATION ===
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Tabs
        self.tabs = QTabWidget()
        
        # Tab 1: Intensity Profiles
        self.tab1 = QWidget()
        self.tab1_layout = QVBoxLayout(self.tab1)
        self.canvas1 = PlotCanvas(self.tab1, width=5, height=4, dpi=100)
        self.toolbar1 = NavigationToolbar(self.canvas1, self.tab1)
        self.tab1_layout.addWidget(self.toolbar1)
        self.tab1_layout.addWidget(self.canvas1)
        self.tabs.addTab(self.tab1, "1. Intensity vs Depth")
        
        # Tab 2: Signal Strength
        self.tab2 = QWidget()
        self.tab2_layout = QVBoxLayout(self.tab2)
        self.canvas2 = PlotCanvas(self.tab2, width=5, height=4, dpi=100)
        self.toolbar2 = NavigationToolbar(self.canvas2, self.tab2)
        self.tab2_layout.addWidget(self.toolbar2)
        self.tab2_layout.addWidget(self.canvas2)
        self.tabs.addTab(self.tab2, "2. Signal Strength")
        
        # Tab 3: Bias Heatmap
        self.tab3 = QWidget()
        self.tab3_layout = QVBoxLayout(self.tab3)
        self.canvas3 = PlotCanvas(self.tab3, width=5, height=4, dpi=100)
        self.toolbar3 = NavigationToolbar(self.canvas3, self.tab3)
        self.tab3_layout.addWidget(self.toolbar3)
        self.tab3_layout.addWidget(self.canvas3)
        self.tabs.addTab(self.tab3, "3. Bias Heatmap")
        
        # Tab 4: Data Tables
        self.tab4 = QWidget()
        self.tab4_layout = QVBoxLayout(self.tab4)
        self.table_widget = QTableWidget()
        self.tab4_layout.addWidget(self.table_widget)
        self.tabs.addTab(self.tab4, "Data Tables")
        
        # Tab 5: Optimization Analysis (Enhanced)
        self.tab5 = QWidget()
        self.tab5_layout = QVBoxLayout(self.tab5)
        
        # Optimization Controls Group
        opt_controls = QGroupBox("Run Advanced Optimization")
        opt_controls_layout = QHBoxLayout()
        
        self.opt_wl_btn = QPushButton("Find Optimal λ")
        self.opt_wl_btn.setToolTip("Find wavelength that minimizes PI variance across skin types")
        self.opt_wl_btn.clicked.connect(lambda: self.run_optimization('wavelength'))
        opt_controls_layout.addWidget(self.opt_wl_btn)
        
        self.opt_spo2_btn = QPushButton("Find SpO2 Pair")
        self.opt_spo2_btn.setToolTip("Find wavelength pair with minimum RoR deviation")
        self.opt_spo2_btn.clicked.connect(lambda: self.run_optimization('spo2'))
        opt_controls_layout.addWidget(self.opt_spo2_btn)
        
        self.opt_sdd_btn = QPushButton("Optimize SDD")
        self.opt_sdd_btn.setToolTip("Find optimal Source-Detector Distance")
        self.opt_sdd_btn.clicked.connect(lambda: self.run_optimization('sdd'))
        opt_controls_layout.addWidget(self.opt_sdd_btn)
        
        self.opt_full_btn = QPushButton("🚀 Run Full Optimization")
        self.opt_full_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        self.opt_full_btn.setToolTip("Run complete optimization pipeline (all 4 phases)")
        self.opt_full_btn.clicked.connect(lambda: self.run_optimization('full'))
        opt_controls_layout.addWidget(self.opt_full_btn)
        
        opt_controls.setLayout(opt_controls_layout)
        self.tab5_layout.addWidget(opt_controls)
        
        # Results Section
        results_group = QGroupBox("Optimization Results")
        results_layout = QVBoxLayout()
        
        # Optimal Wavelength Result
        self.opt_wl_result = QLabel("Optimal Wavelength: Not yet computed")
        self.opt_wl_result.setStyleSheet("font-size: 14px; padding: 5px;")
        results_layout.addWidget(self.opt_wl_result)
        
        # Optimal SpO2 Pair Result
        self.opt_spo2_result = QLabel("Optimal SpO2 Pair: Not yet computed")
        self.opt_spo2_result.setStyleSheet("font-size: 14px; padding: 5px;")
        results_layout.addWidget(self.opt_spo2_result)
        
        # Optimal SDD Result
        self.opt_sdd_result = QLabel("Optimal SDD: Not yet computed")
        self.opt_sdd_result.setStyleSheet("font-size: 14px; padding: 5px;")
        results_layout.addWidget(self.opt_sdd_result)
        
        # Co-optimization Result
        self.opt_coopt_result = QLabel("Co-Optimized Config: Not yet computed")
        self.opt_coopt_result.setStyleSheet("font-size: 14px; padding: 5px; font-weight: bold; color: #2196F3;")
        results_layout.addWidget(self.opt_coopt_result)
        
        results_group.setLayout(results_layout)
        self.tab5_layout.addWidget(results_group)
        
        # Table for detailed results
        self.tab5_layout.addWidget(QLabel("Detailed Analysis:"))
        self.opt_table = QTableWidget()
        self.tab5_layout.addWidget(self.opt_table)
        
        # Canvas for optimization visualization
        self.opt_canvas = PlotCanvas(self.tab5, width=5, height=3, dpi=100)
        self.tab5_layout.addWidget(self.opt_canvas)
        
        self.tabs.addTab(self.tab5, "🎯 Optimization")
        
        right_layout.addWidget(self.tabs)
        splitter.addWidget(right_panel)
        
        # Set initial sizes
        splitter.setSizes([350, 950])

    def log(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    # ... (rest of methods) ...

    def update_plots(self):
        # ... (previous plot updates) ...
        # After updating tables, update optimization tab
        self.update_optimization_tab()

    def update_optimization_tab(self):
        if not self.results:
            return
            
        wavelengths = sorted(self.results.keys())
        skin_types = sorted(self.results[wavelengths[0]].keys())
        
        # 1. Best Single Wavelength
        self.opt_table.clear()
        self.opt_table.setRowCount(len(skin_types) + 1)
        self.opt_table.setColumnCount(2)
        self.opt_table.setHorizontalHeaderLabels(["Target", "Best Wavelength"])
        
        # Per skin type
        for i, st in enumerate(skin_types):
            best_wl = None
            max_pi = -1
            
            for wl in wavelengths:
                pi = self.results[wl][st].get('perfusion_index', 0)
                if pi > max_pi:
                    max_pi = pi
                    best_wl = wl
            
            self.opt_table.setItem(i, 0, QTableWidgetItem(f"Best for Type {st}"))
            self.opt_table.setItem(i, 1, QTableWidgetItem(f"{best_wl} nm (PI={max_pi:.4f})"))
            
            # Highlight Type VI
            if st == 'VI':
                self.opt_table.item(i, 0).setBackground(Qt.yellow)
                self.opt_table.item(i, 1).setBackground(Qt.yellow)
        
        # Universal Best (Min Variance)
        best_universal_wl = None
        min_cv = float('inf')
        
        for wl in wavelengths:
            pis = [self.results[wl][st].get('perfusion_index', 0) for st in skin_types]
            if len(pis) > 0 and np.mean(pis) > 0:
                cv = np.std(pis) / np.mean(pis) # Coefficient of Variation
                if cv < min_cv:
                    min_cv = cv
                    best_universal_wl = wl
        
        row = len(skin_types)
        self.opt_table.setItem(row, 0, QTableWidgetItem("UNIVERSAL BEST (Most Equitable)"))
        self.opt_table.setItem(row, 1, QTableWidgetItem(f"{best_universal_wl} nm (CV={min_cv:.4f})"))
        self.opt_table.item(row, 0).setBackground(Qt.green)
        self.opt_table.item(row, 1).setBackground(Qt.green)
        
        self.opt_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        
        # 2. Optimal SpO2 Pair
        # Simple search for pairs with stable Ratio-of-Ratios (RoR)
        best_pair = None
        min_ror_var = float('inf')
        
        pairs = []
        for i in range(len(wavelengths)):
            for j in range(i+1, len(wavelengths)):
                wl1 = wavelengths[i]
                wl2 = wavelengths[j]
                
                # Calculate RoR for each skin type
                rors = []
                for st in skin_types:
                    pi1 = self.results[wl1][st].get('perfusion_index', 1e-9)
                    pi2 = self.results[wl2][st].get('perfusion_index', 1e-9)
                    if pi2 > 0:
                        rors.append(pi1/pi2)
                
                if len(rors) == len(skin_types):
                    # Variance of RoR across skin types
                    ror_var = np.var(rors)
                    if ror_var < min_ror_var:
                        min_ror_var = ror_var
                        best_pair = (wl1, wl2)
        
        if best_pair:
            self.opt_spo2_result.setText(
                f"Quick Analysis - Use Pair: {best_pair[0]} nm / {best_pair[1]} nm\n"
                f"RoR Variance: {min_ror_var:.6f} (Run full optimization for detailed results)"
            )
        # Note: Full optimization via buttons gives more accurate results

    def toggle_custom_wavelengths(self, index):
        # Index 3 is 'Custom'
        self.custom_wl_input.setVisible(index == 3)

    def get_selected_mode(self):
        modes = ['standard', 'full', 'extended', 'custom']
        return modes[self.mode_combo.currentIndex()]

    def run_simulation(self):
        # Disable button
        self.run_btn.setEnabled(False)
        self.run_btn.setText("Simulating...")
        self.progress_bar.setRange(0, 0) # Indeterminate
        
        mode = self.get_selected_mode()
        nphoton = self.photon_spin.value()
        test_mode = self.test_mode_check.isChecked()
        
        custom_wls = None
        if mode == 'custom':
            try:
                text = self.custom_wl_input.toPlainText()
                custom_wls = [int(w.strip()) for w in text.split(',') if w.strip().isdigit()]
                if not custom_wls:
                    raise ValueError("No valid wavelengths")
            except Exception as e:
                QMessageBox.warning(self, "Input Error", "Invalid custom wavelengths. Use format: 660, 880, 940")
                self.run_btn.setEnabled(True)
                self.run_btn.setText("RUN SIMULATION")
                self.progress_bar.setRange(0, 100)
                return

        # Start thread
        self.worker = SimulationWorker(mode, nphoton, test_mode, custom_wls)
        self.worker.progress_signal.connect(self.log)
        self.worker.finished_signal.connect(self.on_simulation_finished)
        self.worker.error_signal.connect(self.on_simulation_error)
        self.worker.start()

    def on_simulation_finished(self, results):
        self.run_btn.setEnabled(True)
        self.run_btn.setText("RUN SIMULATION")
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(100)
        self.log("Simulation finished successfully!")
        
        self.results = results
        self.update_plots()

    def on_simulation_error(self, error_msg):
        self.run_btn.setEnabled(True)
        self.run_btn.setText("RUN SIMULATION")
        self.progress_bar.setRange(0, 100)
        self.log(f"ERROR: {error_msg}")
        QMessageBox.critical(self, "Simulation Error", error_msg)

    def run_optimization(self, opt_type):
        """Run optimization in background thread"""
        if not HAS_OPTIMIZATION:
            QMessageBox.warning(self, "Not Available", 
                "Optimization module not available. Please ensure ppg_optimization.py exists.")
            return
        
        # Disable buttons during optimization
        self.opt_wl_btn.setEnabled(False)
        self.opt_spo2_btn.setEnabled(False)
        self.opt_sdd_btn.setEnabled(False)
        self.opt_full_btn.setEnabled(False)
        self.progress_bar.setRange(0, 0)
        
        test_mode = self.test_mode_check.isChecked()
        
        self.log(f"Starting optimization: {opt_type} (test_mode={test_mode})")
        
        self.opt_worker = OptimizationWorker(opt_type=opt_type, test_mode=test_mode)
        self.opt_worker.progress_signal.connect(self.log)
        self.opt_worker.finished_signal.connect(self.on_optimization_finished)
        self.opt_worker.error_signal.connect(self.on_optimization_error)
        self.opt_worker.start()

    def on_optimization_finished(self, results):
        """Handle optimization completion"""
        self.opt_wl_btn.setEnabled(True)
        self.opt_spo2_btn.setEnabled(True)
        self.opt_sdd_btn.setEnabled(True)
        self.opt_full_btn.setEnabled(True)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(100)
        
        self.log("Optimization completed!")
        
        # Update results display
        self.update_optimization_results(results)

    def on_optimization_error(self, error_msg):
        """Handle optimization error"""
        self.opt_wl_btn.setEnabled(True)
        self.opt_spo2_btn.setEnabled(True)
        self.opt_sdd_btn.setEnabled(True)
        self.opt_full_btn.setEnabled(True)
        self.progress_bar.setRange(0, 100)
        
        self.log(f"Optimization ERROR: {error_msg}")
        QMessageBox.critical(self, "Optimization Error", error_msg)

    def update_optimization_results(self, results):
        """Update the optimization tab with results"""
        # Handle full optimization results (nested)
        if 'optimal_wavelength' in results and isinstance(results['optimal_wavelength'], dict):
            wl_data = results['optimal_wavelength']
            opt_wl = wl_data.get('optimal_wavelength', 'N/A')
            min_var = wl_data.get('min_variance', 0)
            self.opt_wl_result.setText(
                f"✓ Optimal Wavelength: {opt_wl} nm (PI Variance: {min_var:.6f})"
            )
            self.opt_wl_result.setStyleSheet("font-size: 14px; color: green; font-weight: bold;")
        elif 'optimal_wavelength' in results:
            opt_wl = results.get('optimal_wavelength', 'N/A')
            min_var = results.get('min_variance', 0)
            self.opt_wl_result.setText(
                f"✓ Optimal Wavelength: {opt_wl} nm (PI Variance: {min_var:.6f})"
            )
            self.opt_wl_result.setStyleSheet("font-size: 14px; color: green; font-weight: bold;")
        
        # SpO2 pair results
        if 'optimal_spo2_pair' in results:
            spo2_data = results['optimal_spo2_pair']
            pair = spo2_data.get('optimal_pair', (0, 0))
            improvement = spo2_data.get('improvement_percent', 0)
            self.opt_spo2_result.setText(
                f"✓ Optimal SpO2 Pair: {pair[0]} nm / {pair[1]} nm (Improvement: {improvement:.1f}%)"
            )
            self.opt_spo2_result.setStyleSheet("font-size: 14px; color: green; font-weight: bold;")
        elif 'optimal_pair' in results:
            pair = results.get('optimal_pair', (0, 0))
            improvement = results.get('improvement_percent', 0)
            self.opt_spo2_result.setText(
                f"✓ Optimal SpO2 Pair: {pair[0]} nm / {pair[1]} nm (Improvement: {improvement:.1f}%)"
            )
            self.opt_spo2_result.setStyleSheet("font-size: 14px; color: green; font-weight: bold;")
        
        # SDD results
        if 'sdd_optimization' in results:
            sdd_data = results['sdd_optimization']
            opt_sdd = sdd_data.get('optimal_sdd', 'N/A')
            wl = sdd_data.get('wavelength', 880)
            self.opt_sdd_result.setText(
                f"✓ Optimal SDD: {opt_sdd} mm (for λ={wl} nm)"
            )
            self.opt_sdd_result.setStyleSheet("font-size: 14px; color: green; font-weight: bold;")
        elif 'optimal_sdd' in results:
            opt_sdd = results.get('optimal_sdd', 'N/A')
            wl = results.get('wavelength', 880)
            self.opt_sdd_result.setText(
                f"✓ Optimal SDD: {opt_sdd} mm (for λ={wl} nm)"
            )
            self.opt_sdd_result.setStyleSheet("font-size: 14px; color: green; font-weight: bold;")
        
        # Co-optimization results
        if 'co_optimization' in results:
            co_data = results['co_optimization']
            opt_wl = co_data.get('optimal_wavelength', 'N/A')
            opt_sdd = co_data.get('optimal_sdd', 'N/A')
            score = co_data.get('combined_score', 0)
            self.opt_coopt_result.setText(
                f"★ CO-OPTIMIZED: λ = {opt_wl} nm, SDD = {opt_sdd} mm (Score: {score:.3f})"
            )
            self.opt_coopt_result.setStyleSheet(
                "font-size: 16px; color: #4CAF50; font-weight: bold; padding: 10px; "
                "background-color: #E8F5E9; border-radius: 5px;"
            )
        
        self.log("Optimization results displayed in Optimization tab")

    def check_existing_results(self):
        # Look for most recent results
        output_dir = 'ppg_results/data'
        if os.path.exists(output_dir):
            files = [f for f in os.listdir(output_dir) if f.startswith('simulation_results') and f.endswith('.json')]
            if files:
                latest = sorted(files)[-1]
                path = os.path.join(output_dir, latest)
                self.log(f"Loading existing results from: {latest}")
                try:
                    self.results = load_results(path)
                    self.update_plots()
                except Exception as e:
                    self.log(f"Failed to load existing results: {e}")

    def update_plots(self):
        if not self.results:
            return
        
        self.log("Updating visualizations...")
        
        try:
            # === Figure 1: Intensity vs Depth ===
            self.canvas1.fig.clear()
            wavelengths = sorted(self.results.keys())
            skin_types = sorted(self.results[wavelengths[0]].keys())
            
            # Use 2x3 layout
            axes = self.canvas1.fig.subplots(2, 3)
            # Create a shared legend handle list
            handles, labels = [], []
            
            colors = plt.cm.plasma(np.linspace(0, 0.9, len(wavelengths)))
            
            for idx, st in enumerate(skin_types):
                ax = axes.flatten()[idx]
                for w_idx, wl in enumerate(wavelengths):
                    data = self.results[wl][st]
                    line, = ax.plot(data['depth_mm'], data['intensity'], 
                                   color=colors[w_idx], label=f'{wl} nm' if idx==0 else "")
                    if idx == 0:
                        handles.append(line)
                        labels.append(f'{wl} nm')
                
                ax.set_title(f"Type {st}", fontsize=10)
                ax.set_ylim(0, 1.05)
                ax.grid(True, alpha=0.3)
                if idx >= 3: ax.set_xlabel("Depth (mm)")
                if idx % 3 == 0: ax.set_ylabel("Intensity")
            
            self.canvas1.fig.legend(handles, labels, loc='upper right', fontsize='small')
            self.canvas1.fig.suptitle("Intensity vs Depth by Skin Type", fontsize=12, fontweight='bold')
            self.canvas1.fig.tight_layout(rect=[0, 0, 1, 0.95])
            self.canvas1.draw()
            
            # === Figure 2: Signal Strength ===
            self.canvas2.fig.clear()
            ax2 = self.canvas2.fig.add_subplot(111)
            
            # Prepare data
            x = np.arange(len(skin_types))
            width = 0.8 / len(wavelengths)
            
            for i, wl in enumerate(wavelengths):
                baseline = self.results[wl]['I']['penetration_depth_mm']
                signals = []
                for st in skin_types:
                    val = self.results[wl][st]['penetration_depth_mm']
                    signals.append(val/baseline if baseline > 0 else 0)
                
                ax2.bar(x + (i - len(wavelengths)/2 + 0.5)*width, signals, width, 
                       label=f'{wl} nm', color=colors[i], alpha=0.8)
            
            ax2.set_xticks(x)
            ax2.set_xticklabels([f"Type {st}" for st in skin_types])
            ax2.set_ylabel("Relative Signal Strength")
            ax2.set_title("Signal Degradation with Skin Tone", fontsize=12, fontweight='bold')
            ax2.legend()
            ax2.grid(True, axis='y', alpha=0.3)
            self.canvas2.draw()
            
            # === Figure 3: Bias Heatmap ===
            self.canvas3.fig.clear()
            ax3 = self.canvas3.fig.add_subplot(111)
            
            bias_matrix = np.zeros((len(wavelengths), len(skin_types)))
            for i, wl in enumerate(wavelengths):
                baseline = self.results[wl]['I']['penetration_depth_mm']
                for j, st in enumerate(skin_types):
                    val = self.results[wl][st]['penetration_depth_mm']
                    bias = (1 - val/baseline) * 100 if baseline > 0 else 0
                    bias_matrix[i, j] = bias
            
            im = ax3.imshow(bias_matrix, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=50)
            
            ax3.set_xticks(np.arange(len(skin_types)))
            ax3.set_xticklabels([f"Type {st}" for st in skin_types])
            ax3.set_yticks(np.arange(len(wavelengths)))
            ax3.set_yticklabels([f"{wl} nm" for wl in wavelengths])
            
            # Annotate
            for i in range(len(wavelengths)):
                for j in range(len(skin_types)):
                    val = bias_matrix[i, j]
                    c = 'white' if val > 25 else 'black'
                    ax3.text(j, i, f"{val:.1f}%", ha="center", va="center", color=c)
            
            ax3.set_title("Bias Heatmap (% Signal Reduction)", fontsize=12, fontweight='bold')
            self.canvas3.fig.colorbar(im, ax=ax3, label="Signal Loss (%)")
            self.canvas3.draw()
            
            # === Tables ===
            self.update_tables(wavelengths, skin_types)
            
        except Exception as e:
            self.log(f"Error updating plots: {e}")
            import traceback
            traceback.print_exc()

    def update_tables(self, wavelengths, skin_types):
        # Configure table
        self.table_widget.clear()
        self.table_widget.setRowCount(len(wavelengths))
        self.table_widget.setColumnCount(len(skin_types) + 2) # +2 for Wavelength and Avg
        
        headers = ["Wavelength"] + [f"Type {st}" for st in skin_types] + ["Avg Pen (mm)"]
        self.table_widget.setHorizontalHeaderLabels(headers)
        
        for i, wl in enumerate(wavelengths):
            # Wavelength Input
            self.table_widget.setItem(i, 0, QTableWidgetItem(f"{wl} nm"))
            
            vals = []
            for j, st in enumerate(skin_types):
                pen = self.results[wl][st]['penetration_depth_mm']
                vals.append(pen)
                self.table_widget.setItem(i, j+1, QTableWidgetItem(f"{pen:.3f}"))
            
            # Average
            avg = sum(vals) / len(vals)
            self.table_widget.setItem(i, len(skin_types)+1, QTableWidgetItem(f"{avg:.3f}"))
            
        self.table_widget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    # Set style
    app.setStyle('Fusion')
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())
