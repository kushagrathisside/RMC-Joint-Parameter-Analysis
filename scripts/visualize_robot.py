import argparse
import os
import sys
import time
from pathlib import Path

import imageio.v2 as imageio

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Default to the desktop-friendly backend while still allowing manual override.
os.environ.setdefault("MUJOCO_GL", "glfw")

try:
    import mujoco
except ModuleNotFoundError as exc:
    raise SystemExit(
        "MuJoCo is not installed in the active environment. Run `pip install -r requirements.txt` first."
    ) from exc

import numpy as np
import yaml

try:
    from PyQt5 import QtCore, QtGui, QtWidgets
except ModuleNotFoundError as exc:
    raise SystemExit(
        "PyQt5 is not installed in the active environment. Run `pip install -r requirements.txt` first."
    ) from exc

try:
    from robot_descriptions.loaders.mujoco import load_robot_description
except ModuleNotFoundError as exc:
    raise SystemExit(
        "`robot-descriptions` is not installed in the active environment. Run `pip install -r requirements.txt` first."
    ) from exc

from controllers.nn_compensated_pd import NNCompensatedPD
from controllers.pd_controller import PDController
from controllers.rl_controller import RLController
from trajectories.neural_trajectory import NeuralTrajectoryGenerator
from trajectories.trajectory_generator import TrajectoryGenerator


def parse_args():
    parser = argparse.ArgumentParser(
        description="Qt5 robot visualizer for inspection, reference playback, and live simulation."
    )
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "config.yaml"),
        help="Path to the YAML config file. Defaults to configs/config.yaml.",
    )
    parser.add_argument(
        "--mode",
        choices=("inspect", "trajectory", "simulate"),
        default=None,
        help="Initial mode for the GUI.",
    )
    parser.add_argument(
        "--robot",
        default=None,
        help="Initial robot description name override.",
    )
    parser.add_argument(
        "--robot-variant",
        default=None,
        help="Initial robot variant override, for example `scene`.",
    )
    parser.add_argument(
        "--controller",
        choices=("pd", "rl", "nn_pd"),
        default=None,
        help="Initial controller override.",
    )
    parser.add_argument(
        "--trajectory",
        choices=("classic", "neural"),
        default=None,
        help="Initial trajectory override.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Initial duration in seconds. Use 0 for continuous playback.",
    )
    parser.add_argument(
        "--realtime-rate",
        type=float,
        default=1.0,
        help="Initial playback speed multiplier.",
    )
    parser.add_argument(
        "--camera-distance",
        type=float,
        default=None,
        help="Initial camera distance override.",
    )
    parser.add_argument(
        "--camera-azimuth",
        type=float,
        default=140.0,
        help="Initial camera azimuth in degrees.",
    )
    parser.add_argument(
        "--camera-elevation",
        type=float,
        default=-20.0,
        help="Initial camera elevation in degrees.",
    )
    return parser.parse_args()


def load_config(config_path):
    with open(config_path) as config_file:
        return yaml.safe_load(config_file) or {}


def build_controller(model, controller_type, controller_params):
    if controller_type == "pd":
        return PDController(model, **controller_params)
    if controller_type == "rl":
        return RLController(model)
    if controller_type == "nn_pd":
        return NNCompensatedPD(model, **controller_params)
    raise ValueError(f"Unknown controller type: {controller_type}")


def build_trajectory(trajectory_type, trajectory_params):
    if trajectory_type == "classic":
        return TrajectoryGenerator(**trajectory_params)
    if trajectory_type == "neural":
        return NeuralTrajectoryGenerator()
    raise ValueError(f"Unknown trajectory type: {trajectory_type}")


def control_dofs(model, data):
    return min(7, model.nu, data.ctrl.shape[0], data.qpos.shape[0], data.qvel.shape[0])


class RobotVisualizer(QtWidgets.QMainWindow):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.config_path = Path(args.config)
        self.config = load_config(self.config_path)

        self.model = None
        self.data = None
        self.renderer = None
        self.controller = None
        self.trajectory = None
        self.camera = None
        self.current_frame = None

        self.running = False
        self.sim_time = 0.0
        self.accumulator = 0.0
        self.last_wall_time = time.perf_counter()

        self.is_recording = False
        self.recording_log = None
        self.recording_writer = None
        self.recording_frame_count = 0
        self.recording_npz_path = None
        self.recording_mp4_path = None

        self.render_width = 1280
        self.render_height = 720

        self.setWindowTitle("RMC Robot Visualizer")
        self.resize(1600, 900)
        self._build_ui()
        self._apply_config_to_controls()
        self._load_robot()

        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(33)
        self.timer.timeout.connect(self._tick)
        self.timer.start()

        if self.mode_combo.currentText() != "inspect":
            self.running = True
            self.play_button.setText("Pause")

    def _build_ui(self):
        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)

        layout = QtWidgets.QHBoxLayout(central)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        controls_panel = QtWidgets.QFrame()
        controls_panel.setMinimumWidth(360)
        controls_panel.setMaximumWidth(420)
        controls_panel.setFrameShape(QtWidgets.QFrame.StyledPanel)
        controls_layout = QtWidgets.QVBoxLayout(controls_panel)
        controls_layout.setSpacing(10)

        title = QtWidgets.QLabel("Robot Visualizer")
        title_font = title.font()
        title_font.setPointSize(15)
        title_font.setBold(True)
        title.setFont(title_font)
        controls_layout.addWidget(title)

        subtitle = QtWidgets.QLabel(
            "Qt5 viewer for static inspection, reference motion playback, and live controller simulation."
        )
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet("color: #555;")
        controls_layout.addWidget(subtitle)

        form = QtWidgets.QFormLayout()
        form.setLabelAlignment(QtCore.Qt.AlignLeft)
        form.setFormAlignment(QtCore.Qt.AlignTop)
        form.setVerticalSpacing(8)

        self.robot_input = QtWidgets.QLineEdit()
        form.addRow("Robot", self.robot_input)

        self.robot_variant_input = QtWidgets.QLineEdit()
        form.addRow("Variant", self.robot_variant_input)

        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItems(["inspect", "trajectory", "simulate"])
        form.addRow("Mode", self.mode_combo)

        self.controller_combo = QtWidgets.QComboBox()
        self.controller_combo.addItems(["pd", "rl", "nn_pd"])
        form.addRow("Controller", self.controller_combo)

        self.trajectory_combo = QtWidgets.QComboBox()
        self.trajectory_combo.addItems(["classic", "neural"])
        form.addRow("Trajectory", self.trajectory_combo)

        self.duration_spin = QtWidgets.QDoubleSpinBox()
        self.duration_spin.setRange(0.0, 36000.0)
        self.duration_spin.setDecimals(1)
        self.duration_spin.setSingleStep(5.0)
        self.duration_spin.setSuffix(" s")
        self.duration_spin.setSpecialValueText("Continuous")
        form.addRow("Duration", self.duration_spin)

        self.rate_spin = QtWidgets.QDoubleSpinBox()
        self.rate_spin.setRange(0.1, 4.0)
        self.rate_spin.setDecimals(2)
        self.rate_spin.setSingleStep(0.1)
        self.rate_spin.setValue(1.0)
        form.addRow("Playback Rate", self.rate_spin)

        self.azimuth_spin = QtWidgets.QDoubleSpinBox()
        self.azimuth_spin.setRange(-180.0, 180.0)
        self.azimuth_spin.setDecimals(1)
        self.azimuth_spin.setSingleStep(5.0)
        form.addRow("Azimuth", self.azimuth_spin)

        self.elevation_spin = QtWidgets.QDoubleSpinBox()
        self.elevation_spin.setRange(-89.0, 89.0)
        self.elevation_spin.setDecimals(1)
        self.elevation_spin.setSingleStep(2.0)
        form.addRow("Elevation", self.elevation_spin)

        self.distance_spin = QtWidgets.QDoubleSpinBox()
        self.distance_spin.setRange(0.5, 20.0)
        self.distance_spin.setDecimals(2)
        self.distance_spin.setSingleStep(0.1)
        form.addRow("Distance", self.distance_spin)

        controls_layout.addLayout(form)

        buttons = QtWidgets.QHBoxLayout()
        self.load_button = QtWidgets.QPushButton("Load Robot")
        self.play_button = QtWidgets.QPushButton("Play")
        self.reset_button = QtWidgets.QPushButton("Reset")
        buttons.addWidget(self.load_button)
        buttons.addWidget(self.play_button)
        buttons.addWidget(self.reset_button)
        controls_layout.addLayout(buttons)

        secondary_buttons = QtWidgets.QHBoxLayout()
        self.reload_config_button = QtWidgets.QPushButton("Reload Config")
        self.record_button = QtWidgets.QPushButton("Start Recording")
        self.snapshot_button = QtWidgets.QPushButton("Save Snapshot")
        secondary_buttons.addWidget(self.reload_config_button)
        secondary_buttons.addWidget(self.record_button)
        secondary_buttons.addWidget(self.snapshot_button)
        controls_layout.addLayout(secondary_buttons)

        self.status_label = QtWidgets.QLabel("Ready")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("color: #333;")
        controls_layout.addWidget(self.status_label)

        note = QtWidgets.QLabel(
            "This project currently assumes a 7-joint arm-style control layout. The best fit is the default Panda robot."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: #666;")
        controls_layout.addWidget(note)
        controls_layout.addStretch(1)

        self.render_label = QtWidgets.QLabel()
        self.render_label.setMinimumSize(960, 720)
        self.render_label.setAlignment(QtCore.Qt.AlignCenter)
        self.render_label.setStyleSheet("background-color: #111; border: 1px solid #222;")

        layout.addWidget(controls_panel)
        layout.addWidget(self.render_label, 1)

        self.load_button.clicked.connect(self._load_robot)
        self.play_button.clicked.connect(self._toggle_playback)
        self.reset_button.clicked.connect(self._reset_runtime)
        self.reload_config_button.clicked.connect(self._reload_config)
        self.record_button.clicked.connect(self._toggle_recording)
        self.snapshot_button.clicked.connect(self._save_snapshot)
        self.mode_combo.currentTextChanged.connect(self._on_runtime_settings_changed)
        self.controller_combo.currentTextChanged.connect(self._on_runtime_settings_changed)
        self.trajectory_combo.currentTextChanged.connect(self._on_runtime_settings_changed)
        self.duration_spin.valueChanged.connect(self._on_runtime_settings_changed)
        self.rate_spin.valueChanged.connect(self._on_runtime_settings_changed)
        self.azimuth_spin.valueChanged.connect(self._render_current_frame)
        self.elevation_spin.valueChanged.connect(self._render_current_frame)
        self.distance_spin.valueChanged.connect(self._render_current_frame)

    def _apply_config_to_controls(self):
        controller_cfg = self.config.get("controller", {})
        trajectory_cfg = self.config.get("trajectory", {})
        simulation_cfg = self.config.get("simulation", {})

        robot_name = self.args.robot or self.config.get("robot", {}).get("name", "panda_mj_description")
        robot_variant = self.args.robot_variant
        if robot_variant is None:
            robot_variant = self.config.get("robot", {}).get("variant", "")
        mode = self.args.mode or "trajectory"
        controller_type = self.args.controller or controller_cfg.get("type", "pd")
        trajectory_type = self.args.trajectory or trajectory_cfg.get("type", "classic")
        duration = self.args.duration
        if duration is None:
            duration = float(simulation_cfg.get("duration", 20.0))
        if mode == "inspect":
            duration = 0.0

        self.robot_input.setText(robot_name)
        self.robot_variant_input.setText(robot_variant)
        self.mode_combo.setCurrentText(mode)
        self.controller_combo.setCurrentText(controller_type if controller_type in {"pd", "rl", "nn_pd"} else "pd")
        self.trajectory_combo.setCurrentText(
            trajectory_type if trajectory_type in {"classic", "neural"} else "classic"
        )
        self.duration_spin.setValue(max(0.0, duration))
        self.rate_spin.setValue(self.args.realtime_rate)
        self.azimuth_spin.setValue(self.args.camera_azimuth)
        self.elevation_spin.setValue(self.args.camera_elevation)
        self.distance_spin.setValue(2.5 if self.args.camera_distance is None else self.args.camera_distance)

    def _reload_config(self):
        if self.is_recording:
            self._stop_recording(show_dialog=False)
        try:
            self.config = load_config(self.config_path)
        except Exception as exc:
            self._show_error("Could not reload config", str(exc))
            return
        self._apply_config_to_controls()
        self._load_robot()

    def _load_robot(self):
        if self.is_recording:
            self._stop_recording(show_dialog=False)
        try:
            robot_name = self.robot_input.text().strip() or "panda_mj_description"
            robot_variant = self.robot_variant_input.text().strip() or None
            self.model = load_robot_description(robot_name, variant=robot_variant)
            self.data = mujoco.MjData(self.model)
            self.camera = mujoco.MjvCamera()
            mujoco.mjv_defaultCamera(self.camera)
            self._apply_camera_defaults()
            self._create_renderer()
            self._reset_runtime(autoplay=self.mode_combo.currentText() != "inspect")
            variant_suffix = f" ({robot_variant})" if robot_variant else ""
            self.status_label.setText(f"Loaded `{robot_name}`{variant_suffix}. Adjust the controls or press Play.")
        except Exception as exc:
            self._show_error("Could not load robot", str(exc))

    def _create_renderer(self):
        if self.renderer is not None and hasattr(self.renderer, "close"):
            self.renderer.close()
        self.renderer = mujoco.Renderer(self.model, height=self.render_height, width=self.render_width)

    def _apply_camera_defaults(self):
        self.camera.lookat[:] = self.model.stat.center
        model_distance = max(1.5, float(self.model.stat.extent) * 2.2)
        if self.args.camera_distance is None and self.distance_spin.value() == 2.5:
            self.distance_spin.blockSignals(True)
            self.distance_spin.setValue(model_distance)
            self.distance_spin.blockSignals(False)
        self.camera.distance = self.distance_spin.value()
        self.camera.azimuth = self.azimuth_spin.value()
        self.camera.elevation = self.elevation_spin.value()

    def _reset_runtime(self, autoplay=None):
        if self.model is None:
            return

        self.data = mujoco.MjData(self.model)
        dofs = control_dofs(self.model, self.data)
        if dofs == 0:
            self._show_error(
                "Unsupported model",
                "The loaded model has no controllable joints in the first 7 positions expected by this project.",
            )
            return

        try:
            self.trajectory = build_trajectory(
                self.trajectory_combo.currentText(),
                self.config.get("trajectory", {}).get("params", {}),
            )
            if self.mode_combo.currentText() == "simulate":
                self.controller = build_controller(
                    self.model,
                    self.controller_combo.currentText(),
                    self.config.get("controller", {}).get("params", {}),
                )
            else:
                self.controller = None
        except Exception as exc:
            self._show_error("Could not build runtime objects", str(exc))
            return

        self.sim_time = 0.0
        self.accumulator = 0.0
        self.last_wall_time = time.perf_counter()
        if autoplay is None:
            self.running = self.mode_combo.currentText() != "inspect"
        else:
            self.running = autoplay
        self.play_button.setText("Pause" if self.running else "Play")

        mujoco.mj_forward(self.model, self.data)
        self._render_current_frame()

    def _on_runtime_settings_changed(self):
        self._reset_runtime(autoplay=self.running)

    def _toggle_playback(self):
        self.running = not self.running
        self.last_wall_time = time.perf_counter()
        self.play_button.setText("Pause" if self.running else "Play")
        self._update_status()

    def _toggle_recording(self):
        if self.is_recording:
            self._stop_recording(show_dialog=True)
        else:
            self._start_recording()

    def _clear_recording_buffers(self):
        self.recording_log = {"time": [], "qpos": [], "qvel": [], "ctrl": []}
        self.recording_writer = None
        self.recording_frame_count = 0
        self.recording_npz_path = None
        self.recording_mp4_path = None

    def _build_recording_paths(self):
        results_dir = ROOT / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        base = results_dir / f"visualizer_recording_{timestamp}"
        suffix = 1
        while base.with_suffix(".npz").exists() or base.with_suffix(".mp4").exists():
            suffix += 1
            base = results_dir / f"visualizer_recording_{timestamp}_{suffix:02d}"
        return base.with_suffix(".npz"), base.with_suffix(".mp4")

    def _ensure_recording_writer(self):
        if self.recording_writer is not None or self.recording_mp4_path is None:
            return

        frame_interval_ms = max(1, self.timer.interval())
        fps = max(1, int(round(1000 / frame_interval_ms)))
        self.recording_writer = imageio.get_writer(
            str(self.recording_mp4_path),
            fps=fps,
            codec="libx264",
        )

    def _append_recording_state(self, current_time):
        if self.recording_log is None or self.data is None:
            return

        self.recording_log["time"].append(float(current_time))
        self.recording_log["qpos"].append(self.data.qpos.copy())
        self.recording_log["qvel"].append(self.data.qvel.copy())
        self.recording_log["ctrl"].append(self.data.ctrl.copy())

    def _append_recording_frame(self, frame):
        if self.recording_mp4_path is None:
            return

        self._ensure_recording_writer()
        if self.recording_writer is None:
            return

        self.recording_writer.append_data(np.ascontiguousarray(frame))
        self.recording_frame_count += 1

    def _start_recording(self):
        if self.model is None or self.data is None:
            self._show_error("No model loaded", "Load a robot before starting a recording.")
            return

        self._clear_recording_buffers()
        self.recording_npz_path, self.recording_mp4_path = self._build_recording_paths()
        self.is_recording = True
        self.record_button.setText("Stop Recording")
        self._update_status()

    def _stop_recording(self, show_dialog):
        if not self.is_recording and self.recording_log is None:
            return

        npz_path = self.recording_npz_path
        mp4_path = self.recording_mp4_path

        try:
            if self.recording_log is not None and not self.recording_log["time"] and self.data is not None:
                self._append_recording_state(self.sim_time)

            if self.recording_frame_count == 0:
                if self.current_frame is None:
                    self._render_current_frame()
                if self.current_frame is not None:
                    self._append_recording_frame(self.current_frame.copy())

            if self.recording_log is not None and npz_path is not None:
                np.savez(
                    npz_path,
                    **{key: np.asarray(values) for key, values in self.recording_log.items()},
                )
        except Exception as exc:
            self._clear_recording_state_after_save()
            self._show_error("Could not save recording", str(exc))
            return

        writer_error = None
        if self.recording_writer is not None:
            try:
                self.recording_writer.close()
            except Exception as exc:
                writer_error = exc

        self._clear_recording_state_after_save()

        if writer_error is not None:
            self._show_error("Could not finalize video", str(writer_error))
            return

        message = "Recording saved."
        if npz_path is not None and mp4_path is not None:
            message = f"Recording saved to:\n{npz_path}\n\n{mp4_path}"

        self.status_label.setText(message.replace("\n\n", " | ").replace("\n", " "))
        if show_dialog:
            QtWidgets.QMessageBox.information(self, "Recording Saved", message)
        else:
            self._update_status()

    def _clear_recording_state_after_save(self):
        if self.recording_writer is not None:
            try:
                self.recording_writer.close()
            except Exception:
                pass
        self.is_recording = False
        self.record_button.setText("Start Recording")
        self._clear_recording_buffers()

    def _tick(self):
        if self.model is None or self.data is None or self.renderer is None:
            return

        now = time.perf_counter()
        wall_dt = now - self.last_wall_time
        self.last_wall_time = now
        state_advanced = False

        if self.running:
            rate = self.rate_spin.value()
            mode = self.mode_combo.currentText()
            duration = self.duration_spin.value()

            if mode == "inspect":
                mujoco.mj_forward(self.model, self.data)
                state_advanced = True
                if self.is_recording:
                    self._append_recording_state(self.sim_time)
            elif mode == "trajectory":
                self.sim_time += wall_dt * rate
                self._apply_trajectory_pose(self.sim_time)
                state_advanced = True
            else:
                self.accumulator += wall_dt * rate
                timestep = max(float(self.model.opt.timestep), 1.0 / 240.0)
                while self.accumulator >= timestep:
                    self._step_simulation(self.sim_time)
                    self.sim_time += timestep
                    self.accumulator -= timestep
                    if self.is_recording:
                        self._append_recording_state(self.sim_time)
                    state_advanced = True

            if duration > 0.0 and self.sim_time >= duration:
                self.running = False
                self.play_button.setText("Play")

        self._render_current_frame()
        if self.is_recording and (state_advanced or self.mode_combo.currentText() == "inspect"):
            self._append_recording_frame(self.current_frame.copy())
        self._update_status()

    def _apply_trajectory_pose(self, current_time):
        target = np.asarray(self.trajectory.generate(current_time), dtype=np.float64)
        dofs = control_dofs(self.model, self.data)
        self.data.qpos[:dofs] = target[:dofs]
        self.data.qvel[:dofs] = 0.0
        self.data.ctrl[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        if self.is_recording:
            self._append_recording_state(current_time)

    def _step_simulation(self, current_time):
        target = np.asarray(self.trajectory.generate(current_time), dtype=np.float64)
        dofs = control_dofs(self.model, self.data)
        self.data.ctrl[:] = 0.0
        torque = np.asarray(self.controller.compute(self.data, target), dtype=np.float64)
        self.data.ctrl[:dofs] = torque[:dofs]
        mujoco.mj_step(self.model, self.data)

    def _render_current_frame(self):
        if self.model is None or self.data is None or self.renderer is None or self.camera is None:
            return

        self.camera.lookat[:] = self.model.stat.center
        self.camera.distance = self.distance_spin.value()
        self.camera.azimuth = self.azimuth_spin.value()
        self.camera.elevation = self.elevation_spin.value()

        self.renderer.update_scene(self.data, camera=self.camera)
        frame = np.ascontiguousarray(self.renderer.render())
        self.current_frame = frame

        image = QtGui.QImage(
            frame.data,
            frame.shape[1],
            frame.shape[0],
            frame.strides[0],
            QtGui.QImage.Format_RGB888,
        ).copy()
        pixmap = QtGui.QPixmap.fromImage(image)
        scaled = pixmap.scaled(
            self.render_label.size(),
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation,
        )
        self.render_label.setPixmap(scaled)

    def _update_status(self):
        mode = self.mode_combo.currentText()
        state = "running" if self.running else "paused"
        controller = self.controller_combo.currentText() if mode == "simulate" else "n/a"
        trajectory = self.trajectory_combo.currentText() if mode != "inspect" else "n/a"
        if self.is_recording and self.recording_log is not None:
            recording = (
                f"on ({len(self.recording_log['time'])} states, {self.recording_frame_count} frames)"
            )
        else:
            recording = "off"
        self.status_label.setText(
            f"Mode: {mode} | State: {state} | t={self.sim_time:0.2f}s | "
            f"Controller: {controller} | Trajectory: {trajectory} | Recording: {recording}"
        )

    def _save_snapshot(self):
        if self.current_frame is None:
            self._show_error("No frame available", "Load a robot first so there is something to save.")
            return

        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Snapshot",
            str(ROOT / "results" / "robot_snapshot.png"),
            "PNG Image (*.png)",
        )
        if not path:
            return

        image = QtGui.QImage(
            self.current_frame.data,
            self.current_frame.shape[1],
            self.current_frame.shape[0],
            self.current_frame.strides[0],
            QtGui.QImage.Format_RGB888,
        ).copy()
        image.save(path)

    def _show_error(self, title, message):
        self.running = False
        self.play_button.setText("Play")
        self.status_label.setText(message)
        QtWidgets.QMessageBox.critical(self, title, message)

    def closeEvent(self, event):
        if self.is_recording:
            self._stop_recording(show_dialog=False)
        if self.renderer is not None and hasattr(self.renderer, "close"):
            self.renderer.close()
        super().closeEvent(event)


def main():
    args = parse_args()
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("RMC Robot Visualizer")
    window = RobotVisualizer(args)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
